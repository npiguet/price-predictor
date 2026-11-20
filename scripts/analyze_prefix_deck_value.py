"""How does the deck built from a seat's pool grow, pick by pick?

Scores every prefix of a seat's drafted pool, so the whole draft can be read as
a curve rather than a single end score. Two regimes, and they mean different
things:

``picks 23-45`` -- the builder chooses 23 non-lands out of a larger pool, so a
    pick is worth something only if it displaces the deck's current worst card.
    Deck size is fixed at 23, so ``score(t) - score(t-1)`` is a clean marginal
    value.

``picks 1-22`` -- fewer cards than a deck needs, so there is nothing to choose:
    the deck *is* the pool and the pool is scored directly. At exactly 23 cards
    the builder has no choice either, and the two paths agree to 1e-6 on every
    seat tested, which is what licenses joining them.

**Levels below 23 picks are not comparable across ``t``.** The set the scorer
sees grows with every pick and the scorer never saw a sub-23-card deck in
training, so the absolute number drifts for reasons that have nothing to do with
the cards. What *is* comparable is two agents at the same ``t``: same set size,
same model. Read the gap columns, not the level, and treat pick 1 -- a one-card
set scored by a model trained on 23 -- as the most strained point on the curve.

For the same reason the report gives marginal values only from pick 24. Below
23 a difference mixes the card's own quality with the change in set size, which
is a different quantity from the displacement value measured above it.

A further limit on the whole method: this measures the pool a seat accumulated,
not the difficulty of the choices it faced. A seat in a weak pod collects a
better pool without picking better.

Raw per-prefix scores stream to ``--raw`` as JSONL, one record per
(draft, seat, prefix). Re-running skips prefixes already present, so an
interrupted run resumes and an existing t>=23 file is extended downward cheaply.
``--report-only`` rebuilds the tables without scoring anything.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/analyze_prefix_deck_value.py \\
        --drafts $G/lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl \\
        --raw $G/lr1e-5_t2all_decay0.3-prefix-deck-scores.jsonl

    # pool several corpora into one table
    python scripts/analyze_prefix_deck_value.py --report-only \\
        --title "v-forge" --raw ...-a.jsonl --raw ...-b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from _lib.markdown import banner
from draft_corpus_common import (
    iter_prefix_records,
    load_labeler_and_scorer,
    prefix_curves,
    read_prefix_scores,
    read_records,
)

DECK = 23           # NONLAND_DECK_SIZE: at and above this the builder must choose
AGENTS = ("forge-full", "gen1", "gen3", "gen4")


def _hms(seconds: float) -> str:
    """Durations here run to tens of hours, so h:mm:ss beats bare seconds."""
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def load_raw(path: Path, want: str = "greedy") -> set[tuple[str, int, int]]:
    """(draft_id, seat, n_picks) already scored by a previous run, if by ``want``."""
    done: set[tuple[str, int, int]] = set()
    methods: set[str] = set()
    if not path.exists():
        return done
    for r in iter_prefix_records(path):
        done.add((r["draft_id"], r["seat"], r["n_picks"]))
        methods.add(r.get("build_method", "picker"))
    if len(methods) > 1:
        raise SystemExit(f"{path} mixes builders {sorted(methods)}; scores are not comparable")
    if methods and want not in methods:
        raise SystemExit(
            f"{path} was built with {methods.pop()!r} but --build-method is {want!r}. "
            "Delete the file or pass the matching builder; the two are not comparable.")
    return done


def _seat_jobs(record, geo, index: int, done: set[tuple[str, int, int]]) -> Iterator[tuple]:
    """The not-yet-scored prefixes of one seat's pool, shortest first."""
    pool = geo.drafted_pool(record, index)
    agent = record.seats[index].agent
    for t in range(1, len(pool) + 1):
        if (record.draft_id, index, t) not in done:
            yield (record.draft_id, index, agent, t, pool[:t])


def _pending_jobs(corpus: Path, done: set[tuple[str, int, int]]) -> tuple[list, list]:
    """The prefixes of ``corpus`` not yet scored: ``(early, late)``, below and at/above DECK."""
    early, late = [], []
    for record, geo in read_records(corpus):
        for i in range(len(record.seats)):
            for job in _seat_jobs(record, geo, i, done):
                (late if job[3] >= DECK else early).append(job)
    return early, late


def _announce_pending(done: set[tuple[str, int, int]], early: list, late: list) -> None:
    # Spell out the split: "already scored" counts BOTH regimes, so on a resume it
    # sits confusingly close to the build total and reads as though nothing resumed.
    done_direct = sum(1 for _, _, t in done if t < DECK)
    done_built = len(done) - done_direct
    print(f"{len(done)} prefixes already scored "
          f"({done_direct} direct + {done_built} built); "
          f"remaining: {len(early)} to score directly, {len(late)} to build "
          f"(of {done_built + len(late)} builds for this corpus)", flush=True)


@dataclass(frozen=True)
class Phase:
    """One scoring phase: its jobs, the batch size and how a chunk of jobs is scored."""

    label: str
    jobs: list
    batch: int
    score: Callable[[list], list[float]]


def _write_chunk(out, build_method: str, chunk: list, scores: list[float]) -> None:
    for (draft_id, seat, agent, t, _), score in zip(chunk, scores):
        out.write(json.dumps({"draft_id": draft_id, "seat": seat, "agent": agent,
                              "n_picks": t, "deck_score": score,
                              "build_method": build_method}) + "\n")
    out.flush()          # a kill mid-run costs one batch, not the run


def _progress_line(label: str, n: int, total: int, elapsed: float) -> str:
    rate = n / max(elapsed, 1e-9)
    eta = (total - n) / rate if rate else 0.0
    return (f"  {label} {n}/{total} ({100 * n / total:.1f} %)  "
            f"{rate:.1f}/s  elapsed {_hms(elapsed)}  eta {_hms(eta)}")


def _run_phase(phase: Phase, out, args) -> None:
    jobs = phase.jobs
    if not jobs:
        return
    # Time-based, not batch-based: greedy builds run at a few per second,
    # so a fixed batch stride would go quiet for many minutes at a time.
    phase_start = last_tick = time.time()
    for s in range(0, len(jobs), phase.batch):
        chunk = jobs[s:s + phase.batch]
        _write_chunk(out, args.build_method, chunk, phase.score(chunk))
        n = s + len(chunk)
        now = time.time()
        if now - last_tick >= args.progress_every or n == len(jobs):
            last_tick = now
            print(_progress_line(phase.label, n, len(jobs), now - phase_start), flush=True)


def build_missing(corpus: Path, raw_path: Path, args) -> None:
    from sealed.application.evaluate_scorer import score_decks

    done = load_raw(raw_path, args.build_method)
    early, late = _pending_jobs(corpus, done)
    _announce_pending(done, early, late)
    if not early and not late:
        return

    # GreedyDeckBuilder shuffles with the *global* stdlib RNG, which nothing in
    # src/ ever seeds -- the online trainer seeds torch and numpy and keeps its own
    # random.Random(seed) instance, so the corpora were built unseeded and their
    # deck_scores are not reproducible. Seeding here makes THIS run repeatable; it
    # does not recover the corpora's exact values, only the same distribution.
    random.seed(args.seed)

    labeler, scorer, locator = load_labeler_and_scorer(
        args.cards_path, args.scorer, args.picker, args.build_method)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def pools_of(chunk: list) -> list:
        return [p for *_, p in chunk]

    def score_direct(chunk: list) -> list[float]:
        return score_decks(scorer, pools_of(chunk), locator)

    def score_built(chunk: list) -> list[float]:
        return [sc for _, sc in labeler.build_and_score_many(pools_of(chunk))]

    with raw_path.open("a", encoding="utf-8") as out:
        _run_phase(Phase("direct", early, args.score_batch, score_direct), out, args)
        _run_phase(Phase("build", late, args.batch, score_built), out, args)


@dataclass(frozen=True)
class Curves:
    """The per-prefix curves a report reads: levels, steps and the agents present."""

    level: dict
    delta: dict
    present: list[str]

    def cell(self, a: str, t: int) -> str:
        lv, dv = self.level[(a, t)], self.delta[(a, t)]
        row = f"{st.fmean(lv):>+11.3f}" if lv else f"{'-':>11}"
        row += f"{st.fmean(dv):>+11.3f}" if dv else f"{'-':>11}"
        return row + f"{len(lv):>7}"

    def print_table(self, lo: int, hi: int) -> None:
        print(f"{'picks':>6}" + "".join(f"{a:>29}" for a in self.present))
        print(f"{'':>6}" + "".join(f"{'level':>11}{'added':>11}{'seats':>7}"
                                   for _ in self.present))
        for t in range(lo, hi + 1):
            row = "".join(self.cell(a, t) for a in self.present)
            mark = "   <- pool reaches deck size" if t == DECK else ""
            print(f"{t:>6}{row}{mark}")


def _print_preamble() -> None:
    print("")
    print("level = score of the deck the builder makes from the first t picks.")
    print("added = level(t) - level(t-1), the same seat against itself.")
    print("")
    print("Below 23 picks there is no deck to build: the pool IS the deck, so every card")
    print("drafted must be played and the set the scorer sees grows by one each pick. Both")
    print("the level and the step therefore mix card quality with set size, and the scorer")
    print("never saw a sub-23-card deck in training. From 23 on the deck is a fixed 23")
    print("non-lands chosen out of a growing pool, so a step is a clean displacement value.")
    print("")
    print("Watch the seats column. It falls as the shorter drafts run out, so late rows")
    print("average a different, smaller population than early ones.")
    print("")


def _print_beyond(curves: Curves, lengths: Counter, n_seats: int, cap: int) -> None:
    beyond = sum(n for L, n in lengths.items() if L > cap)
    if not beyond:
        return
    print("")
    print(f"Picks past {cap} exist only for {beyond} of {n_seats} seats "
          f"({100 * beyond / n_seats:.1f} %), drafted from sets with more or larger")
    print("packs. Levels and gaps below are a different population from the table above,")
    print("not a continuation of it: the jump at the boundary is the sample changing.")
    print("Only the added column stays within-seat and comparable.")
    print("")
    curves.print_table(cap + 1, max(lengths))


def _print_lead(level: dict, refs: list[str], cap: int) -> None:
    if not refs:
        return
    print("")
    print(f"gen-4's lead at the same prefix length, picks 1-{cap}")
    print(f"{'picks':>6}" + "".join(f"{'vs ' + a:>16}" for a in refs))
    for t in range(1, cap + 1, 4):
        row = ""
        for a in refs:
            g, o = level[("gen4", t)], level[(a, t)]
            row += f"{st.fmean(g) - st.fmean(o):>+16.3f}" if g and o else f"{'-':>16}"
        print(f"{t:>6}{row}")


def _print_marginal(curves: Curves, cap: int) -> None:
    print("")
    print(f"marginal value over picks {DECK + 1}-{cap}, where deck size is fixed")
    for a in curves.present:
        allv = [x for t in range(DECK + 1, cap + 1) for x in curves.delta[(a, t)]]
        pos = [x for x in allv if x > 0]
        if not allv:
            continue
        print(f"  {a:<12} mean {st.fmean(allv):+.4f}  improves {100*len(pos)/len(allv):.0f}% "
              f"of the time by {st.fmean(pos):+.3f} when it does  n={len(allv)}")


def report(raw_paths: list[Path], title: str, cap: int | None) -> None:
    rows, agent_of = read_prefix_scores(raw_paths)
    level, delta = prefix_curves(rows, agent_of)

    present = [a for a in AGENTS if any(level[(a, t)] for t in range(1, 200))]
    refs = [a for a in present if a != "gen4"]
    lengths = Counter(max(by_t) for by_t in rows.values())
    modal = lengths.most_common(1)[0][0]
    if cap is None:
        cap = modal
    curves = Curves(level, delta, present)

    banner(f"{title}   {len(rows)} seats over {len(raw_paths)} corpora")
    _print_preamble()
    curves.print_table(1, cap)
    _print_beyond(curves, lengths, len(rows), cap)
    _print_lead(level, refs, cap)
    _print_marginal(curves, cap)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", type=Path, help="corpus to score; omit with --report-only")
    ap.add_argument("--raw", type=Path, action="append", required=True,
                    help="JSONL of per-prefix scores; appended to and resumed from. "
                         "Repeat to pool corpora in the report; the run writes to the first.")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--title", default="")
    ap.add_argument("--max-picks", type=int, default=None,
                    help="last pick in the main table; default is the modal pool length")
    ap.add_argument("--batch", type=int, default=96, help="pools per picker build")
    ap.add_argument("--score-batch", type=int, default=2048, help="pools per direct scoring")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the builder's global RNG so the run is repeatable")
    ap.add_argument("--progress-every", type=float, default=60.0,
                    help="seconds between progress lines (default: 60)")
    ap.add_argument("--build-method", choices=("greedy", "picker"), default="greedy",
                    help="Pool -> deck before scoring. MUST match the builder that produced "
                         "the corpus's deck_score, which for every online-trained corpus is "
                         "greedy. GenerateDraftDataConfig defaults to picker, so leaving this "
                         "unset silently measures a different pipeline.")
    ap.add_argument("--cards-path", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--picker", type=Path, required=True)
    args = ap.parse_args()

    if not args.report_only:
        if args.drafts is None:
            ap.error("--drafts is required unless --report-only is given")
        build_missing(args.drafts, args.raw[0], args)
    report(args.raw, args.title, args.max_picks)


if __name__ == "__main__":
    main()
