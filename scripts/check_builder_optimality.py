"""Is the prefix-deck-value curve an upper bound on the pool, or one builder's output?

Adding a card to a pool cannot lower the *best* deck obtainable from it, yet
6.6 % of the steps past deck size in the prefix-deck-value data fall. This script
asks why, and answers it in two parts.

``--report`` reads the raw prefix scores and breaks the fall rate down by pick, by
the level the step started from, and by agent, then asks whether a fall is undone
by the next pick.

``--probe`` settles the cause at the one pick where the optimum is cheap to
compute. At pick 24 the builder keeps 23 spells out of the pool, so enumerating
every legal deck gives the true optimum exactly.

Enumerate LEGAL decks, not leave-one-out over the raw pool. A legal deck is 23
spells (by ``is_land_embedding``) plus any subset of the pool's drafted lands.
Dropping one card of 24 instead yields 22-spell decks whenever the pool holds a
land, and since ``score_decks`` strips only *basic* lands the scorer then sees a
different number of cards than it does for the builder's own deck. That
comparison inflates the apparent optimum by ~0.27 and is how an earlier pass
wrongly concluded the builder was leaving 0.33 on the table. It is not: on the
legal enumeration the builder is the argmax on 99 of 100 pools at pick 24.

The two builders are alternatives, not stages. ``greedy`` is a simulated-annealing
search that maximises the scorer directly, and is what every online-trained corpus
used; ``picker`` is a one-shot network that was never retrained for draft pools and
is scored afterwards by a model it does not optimise. Neither is exhaustive, so
neither is monotone in the pool -- but they are not comparable to each other, and
``--build-method`` must match the corpus or the levels mean nothing.

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


def _legal_decks(pool: list[str], locator, scorer) -> list[list[str]] | None:
    """Every deck the builder could legally return: 23 spells + any land subset."""
    import itertools

    from sealed.application.deck_assembly import assemble_full_deck, load_pool_embeddings
    from sealed.domain.greedy_deck_builder import GreedyDeckBuilder

    embeddings, valid = load_pool_embeddings(pool, locator)
    if len(valid) < DECK:
        return None
    spells, lands = GreedyDeckBuilder(scorer, embeddings)._partition_pool(valid)
    if len(spells) < DECK:
        return None            # builder falls back to the whole pool; not a choice
    land_subsets = [list(c) for r in range(len(lands) + 1)
                    for c in itertools.combinations(lands, r)]
    spell_subsets = list(itertools.combinations(spells, DECK))
    if len(spell_subsets) * len(land_subsets) > 40_000:
        return None
    return [assemble_full_deck([valid[i] for i in list(sp) + ld], locator)
            for sp in spell_subsets for ld in land_subsets]


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
        picker_checkpoint=args.picker, cards_path=args.cards_path,
        build_method=args.build_method), locator=locator)
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

    pools = [p for p in pools if _legal_decks(p, locator, scorer) is not None]
    built = [sc for _deck, sc in labeler.build_and_score_many(pools)]
    best = [max(score_decks(scorer, _legal_decks(p, locator, scorer), locator))
            for p in pools]
    gap = [b - s for b, s in zip(best, built) if s is not None]
    kept = [s for s in built if s is not None]

    print("")
    print(f"{len(gap)} seats at pick {size}, every legal deck enumerated and scored")
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
    ap.add_argument("--build-method", choices=("greedy", "picker"), default="greedy",
                    help="Pool -> deck before scoring; greedy is what every corpus used.")
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
