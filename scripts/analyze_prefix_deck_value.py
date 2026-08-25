"""How does the best deck buildable from a seat's pool grow, pick by pick?

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
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from draft_corpus_common import read_records, seat_pool  # noqa: E402

DECK = 23           # NONLAND_DECK_SIZE: at and above this the builder must choose
AGENTS = ("forge-full", "gen1", "gen3", "gen4")


def load_raw(path: Path) -> set[tuple[str, int, int]]:
    """(draft_id, seat, n_picks) already scored by a previous run."""
    done: set[tuple[str, int, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue        # a torn final line from an interrupted run
            done.add((r["draft_id"], r["seat"], r["n_picks"]))
    return done


def make_models(args):
    """The picker-based labeler for full pools, and the bare scorer for short ones."""
    import torch

    from draft.application.generate_draft_data import GenerateDraftDataConfig, build_labeler
    from sealed.domain.scorer_model import SetTransformerScorer
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
    from sealed.infrastructure.scorer_store import ScorerStore

    locator = ConvertedCardLocator(args.cards_path)
    cfg = GenerateDraftDataConfig(
        n_drafts=0, agent_mix=[], scorer_checkpoint=args.scorer,
        picker_checkpoint=args.picker, cards_path=args.cards_path,
    )
    labeler = build_labeler(cfg, locator=locator)
    ckpt = ScorerStore().load_checkpoint(args.scorer)
    scorer = SetTransformerScorer(ckpt.config)
    scorer.load_state_dict(ckpt.model_state_dict)
    scorer.eval()
    scorer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return labeler, scorer, locator


def build_missing(corpus: Path, raw_path: Path, args) -> None:
    from sealed.application.evaluate_scorer import score_decks

    done = load_raw(raw_path)
    early, late = [], []
    for record, geo in read_records(corpus):
        for i, seat in enumerate(record.seats):
            pool = seat_pool(record, geo, i)
            for t in range(1, len(pool) + 1):
                if (record.draft_id, i, t) in done:
                    continue
                job = (record.draft_id, i, seat.agent, t, pool[:t])
                (late if t >= DECK else early).append(job)
    print(f"{len(done)} prefixes already scored; {len(early)} to score directly, "
          f"{len(late)} to build", flush=True)
    if not early and not late:
        return

    labeler, scorer, locator = make_models(args)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with raw_path.open("a", encoding="utf-8") as out:
        def flush(chunk, scores):
            for (draft_id, seat, agent, t, _), score in zip(chunk, scores):
                out.write(json.dumps({"draft_id": draft_id, "seat": seat, "agent": agent,
                                      "n_picks": t, "deck_score": score}) + "\n")
            out.flush()          # a kill mid-run costs one batch, not the run

        for label, jobs, batch in (("direct", early, args.score_batch),
                                   ("build", late, args.batch)):
            for s in range(0, len(jobs), batch):
                chunk = jobs[s:s + batch]
                pools = [p for *_, p in chunk]
                if label == "direct":
                    flush(chunk, score_decks(scorer, pools, locator))
                else:
                    flush(chunk, [sc for _, sc in labeler.build_and_score_many(pools)])
                if s % (batch * 20) == 0 or s + len(chunk) == len(jobs):
                    n = s + len(chunk)
                    rate = n / max(time.time() - start, 1e-9)
                    print(f"  {label} {n}/{len(jobs)}  {rate:.0f}/s", flush=True)


def report(raw_paths: list[Path], title: str, cap: int | None) -> None:
    from collections import Counter

    rows: dict = defaultdict(dict)
    agent_of: dict = {}
    for raw_path in raw_paths:
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["draft_id"], r["seat"])
                rows[key][r["n_picks"]] = r["deck_score"]
                agent_of[key] = r["agent"]

    level: dict = defaultdict(list)
    delta: dict = defaultdict(list)
    for key, by_t in rows.items():
        a = agent_of[key]
        for t, score in by_t.items():
            if score is None:
                continue
            level[(a, t)].append(score)
            prev = by_t.get(t - 1)
            if prev is not None:
                delta[(a, t)].append(score - prev)

    present = [a for a in AGENTS if any(level[(a, t)] for t in range(1, 200))]
    refs = [a for a in present if a != "gen4"]
    lengths = Counter(max(by_t) for by_t in rows.values())
    modal = lengths.most_common(1)[0][0]
    if cap is None:
        cap = modal
    last = max(lengths)

    def table(lo: int, hi: int) -> None:
        print(f"{'picks':>6}" + "".join(f"{a:>29}" for a in present))
        print(f"{'':>6}" + "".join(f"{'level':>11}{'added':>11}{'seats':>7}" for _ in present))
        for t in range(lo, hi + 1):
            row = ""
            for a in present:
                lv, dv = level[(a, t)], delta[(a, t)]
                row += f"{st.fmean(lv):>+11.3f}" if lv else f"{'-':>11}"
                row += f"{st.fmean(dv):>+11.3f}" if dv else f"{'-':>11}"
                row += f"{len(lv):>7}"
            mark = "   <- pool reaches deck size" if t == DECK else ""
            print(f"{t:>6}{row}{mark}")

    print("")
    print("=" * 78)
    print(f"{title}   {len(rows)} seats over {len(raw_paths)} corpora")
    print("")
    print("level = score of the best deck buildable from the first t picks.")
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
    table(1, cap)

    beyond = sum(n for L, n in lengths.items() if L > cap)
    if beyond:
        print("")
        print(f"Picks past {cap} exist only for {beyond} of {len(rows)} seats "
              f"({100 * beyond / len(rows):.1f} %), drafted from sets with more or larger")
        print("packs. Levels and gaps below are a different population from the table above,")
        print("not a continuation of it: the jump at the boundary is the sample changing.")
        print("Only the added column stays within-seat and comparable.")
        print("")
        table(cap + 1, last)

    if refs:
        print("")
        print(f"gen-4's lead at the same prefix length, picks 1-{cap}")
        print(f"{'picks':>6}" + "".join(f"{'vs ' + a:>16}" for a in refs))
        for t in range(1, cap + 1, 4):
            row = ""
            for a in refs:
                g, o = level[("gen4", t)], level[(a, t)]
                row += f"{st.fmean(g) - st.fmean(o):>+16.3f}" if g and o else f"{'-':>16}"
            print(f"{t:>6}{row}")

    print("")
    print(f"marginal value over picks {DECK + 1}-{cap}, where deck size is fixed")
    for a in present:
        allv = [x for t in range(DECK + 1, cap + 1) for x in delta[(a, t)]]
        pos = [x for x in allv if x > 0]
        if not allv:
            continue
        print(f"  {a:<12} mean {st.fmean(allv):+.4f}  improves {100*len(pos)/len(allv):.0f}% "
              f"of the time by {st.fmean(pos):+.3f} when it does  n={len(allv)}")


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
    ap.add_argument("--cards-path", type=Path, default=Path("output/cardsfolder-512"))
    ap.add_argument("--scorer", type=Path, default=Path(
        "models/sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt"))
    ap.add_argument("--picker", type=Path, default=Path(
        "models/sealed/picker/best_20260524_203230-4l-8h-4top256-1e-5lr.pt"))
    args = ap.parse_args()

    if not args.report_only:
        if args.drafts is None:
            ap.error("--drafts is required unless --report-only is given")
        build_missing(args.drafts, args.raw[0], args)
    report(args.raw, args.title, args.max_picks)


if __name__ == "__main__":
    main()
