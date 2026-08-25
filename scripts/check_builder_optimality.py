"""Is the prefix-deck-value curve an upper bound on the pool, or one builder's output?

Adding a card to a pool cannot lower the *best* deck obtainable from it, yet 15 %
of the steps past deck size in the prefix-deck-value data fall. This script asks
why, and answers it in two parts.

``--report`` reads the raw prefix scores and breaks the fall rate down by pick, by
the level the step started from, and by agent, then asks whether a fall is undone
by the next pick.

``--probe`` settles the cause at the one pick where the optimum is cheap to
compute. At pick 24 the builder must keep 23 of 24 cards, so scoring all 24
leave-one-out decks gives the true best deck exactly. Comparing it with what the
builder returned separates a hard search from a builder that is not searching for
this quantity at all.

The builder is a frozen picker network followed by a simulated-annealing pass, and
the scorer that grades its output is a separate model the picker was never trained
to maximise, so nothing makes its output monotone in the pool.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/check_builder_optimality.py --report \\
        --raw $G/lr1e-5_t2all_decay0.3-prefix-deck-scores.jsonl

    python scripts/check_builder_optimality.py --probe \\
        --drafts $G/lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl --n-drafts 60
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from draft_corpus_common import read_records, seat_pool  # noqa: E402

DECK = 23       # NONLAND_DECK_SIZE
TOL = 1e-3      # below this a score difference cannot change which deck is better


def report(raw_paths: list[Path], last: int) -> None:
    """Where do the falls sit, and do they stick?"""
    rows: dict = defaultdict(dict)
    agent_of: dict = {}
    for path in raw_paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue        # a torn final line from an interrupted run
                key = (path.name, r["draft_id"], r["seat"])
                rows[key][r["n_picks"]] = r["deck_score"]
                agent_of[key] = r["agent"]

    by_pick: dict = defaultdict(lambda: [0, 0])
    by_agent: dict = defaultdict(lambda: [0, 0])
    by_level: dict = defaultdict(lambda: [0, 0])
    regained = [0, 0]
    drops: list[float] = []
    for key, by_t in rows.items():
        agent = agent_of[key]
        for t in range(DECK + 1, last + 1):
            cur, prev = by_t.get(t), by_t.get(t - 1)
            if cur is None or prev is None:
                continue
            fell = cur - prev < -TOL
            for acc in (by_pick[t], by_agent[agent], by_level[round(prev)]):
                acc[0] += fell
                acc[1] += 1
            if fell:
                drops.append(prev - cur)
                nxt = by_t.get(t + 1)
                if nxt is not None:
                    regained[0] += nxt >= prev - TOL
                    regained[1] += 1

    print("")
    print(f"{len(rows)} seats, steps past deck size; a fall is a drop of more than {TOL}")
    print("")
    print("by pick")
    for t in range(DECK + 1, last + 1):
        n_fell, n = by_pick[t]
        if not n:
            continue
        print(f"  {t:>2} {100 * n_fell / n:>5.1f} %  {'#' * round(50 * n_fell / n)}")
    print("")
    print("by the level the step started from")
    for lv in sorted(by_level):
        n_fell, n = by_level[lv]
        if n < 2000:
            continue
        print(f"  {lv:>+3} {100 * n_fell / n:>5.1f} %   n={n:,}")
    print("")
    print("by agent")
    for agent, (n_fell, n) in sorted(by_agent.items()):
        print(f"  {agent:<12}{100 * n_fell / n:>6.1f} %   n={n:,}")
    print("")
    print(f"fall size: median {st.median(drops):.3f}  mean {st.fmean(drops):.3f}  "
          f"max {max(drops):.2f}")
    print(f"the next pick regains the pre-fall level: "
          f"{100 * regained[0] / regained[1]:.1f} %  (n={regained[1]:,})")


def probe(corpus: Path, n_drafts: int, args) -> None:
    """At pick 24 the optimum is 24 decks away. Score them all and compare."""
    import torch

    from draft.application.generate_draft_data import GenerateDraftDataConfig, build_labeler
    from sealed.application.evaluate_scorer import score_decks
    from sealed.domain.scorer_model import SetTransformerScorer
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
    from sealed.infrastructure.scorer_store import ScorerStore

    locator = ConvertedCardLocator(args.cards_path)
    labeler = build_labeler(GenerateDraftDataConfig(
        n_drafts=0, agent_mix=[], scorer_checkpoint=args.scorer,
        picker_checkpoint=args.picker, cards_path=args.cards_path), locator=locator)
    ckpt = ScorerStore().load_checkpoint(args.scorer)
    scorer = SetTransformerScorer(ckpt.config)
    scorer.load_state_dict(ckpt.model_state_dict)
    scorer.eval()
    scorer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    size = DECK + 1
    pools = []
    for n, (record, geo) in enumerate(read_records(corpus)):
        if n >= n_drafts:
            break
        for i in range(geo.pod_size):
            pools.append(seat_pool(record, geo, i)[:size])

    built = [sc for _deck, sc in labeler.build_and_score_many(pools)]
    best = [max(score_decks(scorer, [p[:j] + p[j + 1:] for j in range(size)], locator))
            for p in pools]
    gap = [b - s for b, s in zip(best, built) if s is not None]
    kept = [s for s in built if s is not None]

    print("")
    print(f"{len(gap)} seats at pick {size}, every one of the {size} leave-one-out decks scored")
    print("")
    print(f"the builder is beaten by the best of them: "
          f"{100 * sum(g > TOL for g in gap) / len(gap):.0f} % of seats")
    print(f"shortfall: mean {st.fmean(gap):+.3f}  median {st.median(gap):+.3f}  "
          f"max {max(gap):+.2f}")
    print(f"mean level: builder {st.fmean(kept):+.3f}  best available {st.fmean(best):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="break the falls down by pick")
    ap.add_argument("--probe", action="store_true", help="exhaustive search at pick 24")
    ap.add_argument("--raw", type=Path, action="append", default=[],
                    help="prefix-deck-scores JSONL; repeat to pool corpora")
    ap.add_argument("--drafts", type=Path, help="corpus to probe")
    ap.add_argument("--n-drafts", type=int, default=60)
    ap.add_argument("--last-pick", type=int, default=45)
    ap.add_argument("--cards-path", type=Path, default=Path("output/cardsfolder-512"))
    ap.add_argument("--scorer", type=Path, default=Path(
        "models/sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt"))
    ap.add_argument("--picker", type=Path, default=Path(
        "models/sealed/picker/best_20260524_203230-4l-8h-4top256-1e-5lr.pt"))
    args = ap.parse_args()

    if not (args.report or args.probe):
        ap.error("give --report, --probe, or both")
    if args.report:
        if not args.raw:
            ap.error("--report needs at least one --raw")
        report(args.raw, args.last_pick)
    if args.probe:
        if args.drafts is None:
            ap.error("--probe needs --drafts")
        probe(args.drafts, args.n_drafts, args)


if __name__ == "__main__":
    main()
