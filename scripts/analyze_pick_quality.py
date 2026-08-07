"""Was an off-lane pick worth it? Card-quality diagnostics over draft corpora.

``analyze_draft_lanes.py`` shows *how often* an agent drafts outside its colours
and whether it had a choice. This script asks what it got for it: when an agent
declines an on-colour card for an off-colour one, is the off-colour card
actually better?

Per agent it reports

- **best-card rate** — share of picks taking the highest-quality card in the pack;
- **off-lane premium** — for every voluntary off-lane pick (outside the seat's
  eventual top-2 colours, made with an on-colour card still available), the
  taken card's quality minus the best available on-colour card's, summarised as
  mean, median, and the share above zero.

A premium near zero with a share near 50 % means the agent's off-colour picks
are indifferent to card quality — it is not trading colour discipline for
power, it has simply stopped weighing colour.

Two quality scales, because neither alone is conclusive:

``--quality winrates`` (default, preferred)
    ``shrunk_score_play`` from a sealed ``cards-win-rates.txt``. Derived from
    real game outcomes, so it is independent of any agent's pick behaviour.

``--quality pickrate``
    P(taken | available), estimated from ``--reference-agents`` seats only. Needs
    no external file, but the scale is defined by the references' own choices —
    so they score high on *best-card rate* by construction and that column is
    not comparable between a reference and a candidate. The premium column is
    unaffected, since it compares two cards on one scale.

Usage
-----
    G=models/draft/agent/gen3

    python scripts/analyze_pick_quality.py --cards-path output/cardsfolder-512 \\
        --drafts "field at argmax, T3=$G/lr1e-5_t3_20260806_144756-yardstick-drafts.jsonl" \\
        --winrates output/sealed/cards-win-rates.txt

    # no win-rate labels on hand:
    python scripts/analyze_pick_quality.py --drafts corpus.jsonl --quality pickrate

``cards-win-rates.txt`` is rewritten by every ``python -m sealed train-encoder``
run; point ``--winrates`` at whichever snapshot matches the corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from draft_corpus_common import (
    ColourResolver,
    parse_drafts_arg,
    read_records,
    seat_pool,
    top_two_colours,
)

DEFAULT_CARDS_PATH = Path("output/cardsfolder-512")
DEFAULT_WINRATES = Path("output/sealed/cards-win-rates.txt")
QUALITY_COLUMN = "shrunk_score_play"


def load_winrate_quality(path: Path, column: str = QUALITY_COLUMN) -> dict[str, float]:
    """card name -> ``shrunk_score_play``; blank cells (no signal) are skipped."""
    quality: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split(";")
        try:
            index = header.index(column)
        except ValueError as exc:
            raise SystemExit(f"{path} has no '{column}' column") from exc
        for line in handle:
            fields = line.rstrip("\r\n").split(";")
            if len(fields) <= index or not fields[index]:
                continue
            try:
                quality[fields[0]] = float(fields[index])
            except ValueError:
                continue
    return quality


def build_pickrate_quality(
    corpora: list[tuple[str, Path]], references: tuple[str, ...], min_obs: int
) -> dict[str, float]:
    """P(taken | available) over reference seats, pooled across every corpus."""
    seen: Counter = Counter()
    taken: Counter = Counter()
    for _, path in corpora:
        for record, geometry in read_records(path):
            for index, seat in enumerate(record.seats):
                if seat.agent not in references:
                    continue
                for pack in range(1, geometry.packs + 1):
                    for pick in range(1, geometry.pack_size + 1):
                        legal = geometry.legal_actions(record, index, pack, pick)
                        taken[legal[0]] += 1
                        for card in legal:
                            seen[card] += 1
    return {c: taken[c] / n for c, n in seen.items() if n >= min_obs}


def analyze(
    label: str,
    path: Path,
    colours: ColourResolver,
    quality: dict[str, float],
    agents: tuple[str, ...] | None,
) -> None:
    best_hit: Counter = Counter()
    best_total: Counter = Counter()
    premiums: dict[str, list[float]] = defaultdict(list)
    scored_slots = [0, 0]

    for record, geometry in read_records(path):
        for booster in record.boosters:
            for card in booster.picks:
                scored_slots[1] += 1
                if card in quality:
                    scored_slots[0] += 1
        for index, seat in enumerate(record.seats):
            if agents is not None and seat.agent not in agents:
                continue
            top2 = top_two_colours(seat_pool(record, geometry, index), colours)
            for pack in range(1, geometry.packs + 1):
                for pick in range(1, geometry.pack_size + 1):
                    legal = geometry.legal_actions(record, index, pack, pick)
                    got = legal[0]
                    if got not in quality:
                        continue
                    scored = [quality[c] for c in legal if c in quality]
                    if len(scored) >= 2:
                        best_total[seat.agent] += 1
                        if quality[got] >= max(scored):
                            best_hit[seat.agent] += 1
                    taken_colours = colours(got)
                    if not taken_colours or taken_colours <= top2:
                        continue
                    on_lane = [
                        quality[card]
                        for card in legal[1:]
                        if card in quality
                        and (c := colours(card))
                        and c <= top2
                    ]
                    if on_lane:
                        premiums[seat.agent].append(quality[got] - max(on_lane))

    coverage = 100 * scored_slots[0] / scored_slots[1] if scored_slots[1] else 0
    print("=" * 96)
    print(f"{label}   ({coverage:.1f}% of drafted card slots carry a quality label)")
    print(
        f"     {'agent':<12} {'best-card':>10} {'n':>8} {'mean prem':>11} "
        f"{'median':>9} {'share>0':>9}"
    )
    for agent in sorted(set(best_total) | set(premiums)):
        values = premiums[agent]
        rate = 100 * best_hit[agent] / best_total[agent] if best_total[agent] else 0
        if values:
            above = 100 * sum(1 for v in values if v > 0) / len(values)
            print(
                f"     {agent:<12} {rate:9.1f}% {len(values):8d} "
                f"{mean(values):+11.4f} {median(values):+9.4f} {above:8.1f}%"
            )
        else:
            print(f"     {agent:<12} {rate:9.1f}% {0:8d} {'-':>11} {'-':>9} {'-':>9}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--drafts", action="append", required=True, metavar="LABEL=PATH",
        help="corpus to analyze; repeatable. Bare PATH labels by file stem.",
    )
    parser.add_argument("--cards-path", type=Path, default=DEFAULT_CARDS_PATH)
    parser.add_argument("--quality", choices=("winrates", "pickrate"), default="winrates")
    parser.add_argument("--winrates", type=Path, default=DEFAULT_WINRATES)
    parser.add_argument(
        "--reference-agents", default="gen1,forge-full",
        help="seats defining the pickrate scale (--quality pickrate only)",
    )
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument(
        "--agents", default=None,
        help="comma-separated mix labels to report (default: all seen)",
    )
    args = parser.parse_args()

    corpora = [parse_drafts_arg(raw) for raw in args.drafts]
    agents = tuple(a.strip() for a in args.agents.split(",")) if args.agents else None

    if args.quality == "winrates":
        if not args.winrates.exists():
            raise SystemExit(
                f"{args.winrates} not found — pass --winrates PATH, or use "
                f"--quality pickrate to derive a scale from the corpus itself"
            )
        quality = load_winrate_quality(args.winrates)
        values = sorted(quality.values())
        print(
            f"quality: {QUALITY_COLUMN} for {len(quality)} cards "
            f"(range {values[0]:+.3f} .. {values[-1]:+.3f}, "
            f"median {values[len(values) // 2]:+.3f})\n"
        )
    else:
        references = tuple(a.strip() for a in args.reference_agents.split(","))
        quality = build_pickrate_quality(corpora, references, args.min_obs)
        print(
            f"quality: P(taken | available) over {'/'.join(references)} seats, "
            f"{len(quality)} cards with >={args.min_obs} observations\n"
        )

    colours = ColourResolver(args.cards_path)
    for label, path in corpora:
        analyze(label, path, colours, quality, agents)


if __name__ == "__main__":
    main()
