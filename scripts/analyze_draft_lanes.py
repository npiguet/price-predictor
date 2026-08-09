"""Colour-discipline diagnostics over one or more ``drafts.jsonl`` corpora.

Answers "does this agent commit to a lane, and when?" by reconstructing every
seat's 45 picks from the recorded booster geometry. Three reports per corpus,
each broken down per agent so a candidate is read against the frozen references
sitting in the same pods (a paired comparison — pod luck cancels):

A. **Mana-base width** — distinct basic land types in the seat's *built* deck,
   with each bucket's mean ``deck_score``. A four- or five-colour mana base is
   the visible end of a pool that never found a deep colour pair.

B. **Pool concentration by pack** — after each pack, the effective colour count
   of the cards drafted so far (smallest set of colours covering 80 % of the
   coloured cards) and the share of the pool sitting in the seat's eventual
   top-2 colours. Shows whether a lane converges as the draft goes on.

C. **Off-lane picks** — per pack and per pick window, the share of coloured
   picks falling outside the seat's eventual top-2 colours, and — of those —
   the share made while an on-colour card was *still in the pack*. That second
   number separates a forced off-colour pick (nothing on-colour left) from a
   voluntary one (declined an on-colour card), which is the distinction that
   tells a starved drafter from an undisciplined one.

D. **Colour of off-lane picks** — when a seat breaks lane, which colour it
   breaks into, measured against the colour mix of the off-lane cards actually
   available in that pack at that pick. The availability baseline carries the
   report: off-lane is defined against the seat's own top-2, so a black-green
   seat can never make a black off-lane pick, and raw colour counts would only
   re-describe the lane distribution. A lift above 1 means the agent takes that
   colour more often than the supply of it predicts. Unlike C, D runs over every
   pick rather than the three windows.

Two windows carry no signal by construction and are useful as a control: at the
last picks of a pack every agent is forced off-lane, and at the first picks
every off-lane pick is voluntary because a full pack nearly always still holds
an on-colour card.

Usage
-----
    G=models/draft/agent/gen3
    A=$G/temperature-on-all-agents

    python scripts/analyze_draft_lanes.py --cards-path output/cardsfolder-512 \\
        --drafts "field at T, T2=$A/lr1e-5_t2_20260805_221050-yardstick-drafts.jsonl" \\
        --drafts "field at argmax, T3=$G/lr1e-5_t3_20260806_144756-yardstick-drafts.jsonl"

``--drafts`` repeats, takes ``LABEL=PATH`` (bare ``PATH`` labels by file stem),
and ``--agents`` selects which mix labels to report (default: every label seen).
Expect a few minutes per 500-draft corpus — the work is reconstructing ~180k
picks and resolving each card's colour identity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from draft_corpus_common import (
    ColourResolver,
    effective_colour_count,
    mana_base_width,
    parse_drafts_arg,
    read_records,
    seat_pool,
    top_two_colours,
)

DEFAULT_CARDS_PATH = Path("output/cardsfolder-512")
WINDOWS = (("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15))
WUBRG = "WUBRG"
LEAN_COLOURS = frozenset("GBW")


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:5.1f}%" if denominator else "    -"


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Sample mean and its standard error; ``(nan, nan)`` below two samples."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (variance / n) ** 0.5


def analyze(
    label: str,
    path: Path,
    colours: ColourResolver,
    agents: tuple[str, ...] | None,
) -> None:
    width_counts: dict[str, Counter] = defaultdict(Counter)
    width_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    eff_counts: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    top2_share: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    off_lane: dict[str, dict[tuple[int, str], list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    voluntary: dict[str, dict[tuple[int, str], list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    lean_taken: dict[str, Counter] = defaultdict(Counter)
    lean_avail: dict[str, Counter] = defaultdict(Counter)
    lean_diff: dict[str, list[float]] = defaultdict(list)
    seats_seen: Counter = Counter()

    for record, geometry in read_records(path):
        for index, seat in enumerate(record.seats):
            if agents is not None and seat.agent not in agents:
                continue
            seats_seen[seat.agent] += 1
            pool = seat_pool(record, geometry, index)
            top2 = top_two_colours(pool, colours)

            if seat.deck:
                width = mana_base_width(seat.deck)
                width_counts[seat.agent][width] += 1
                if seat.deck_score is not None:
                    width_scores[seat.agent][width].append(seat.deck_score)

            for cut in range(1, geometry.packs + 1):
                accumulated: Counter = Counter()
                for offset, card in enumerate(pool):
                    if offset // geometry.pack_size < cut:
                        for colour in colours(card):
                            accumulated[colour] += 1
                eff_counts[seat.agent][cut][effective_colour_count(accumulated)] += 1
                total = sum(accumulated.values())
                if total:
                    top2_share[seat.agent][cut].append(
                        sum(n for c, n in accumulated.items() if c in top2) / total
                    )

            for pack in range(1, geometry.packs + 1):
                for pick in range(1, geometry.pack_size + 1):
                    legal = geometry.legal_actions(record, index, pack, pick)
                    taken_colours = colours(legal[0])
                    if not taken_colours:
                        continue

                    # D runs over every pick, not only the three windows.
                    if not taken_colours <= top2:
                        options = [
                            c for card in legal
                            if (c := colours(card)) and not c <= top2
                        ]
                        if options:
                            weight = 1.0 / len(options)
                            for c in taken_colours:
                                lean_taken[seat.agent][c] += 1.0 / len(taken_colours)
                            for option in options:
                                for c in option:
                                    lean_avail[seat.agent][c] += weight / len(option)
                            lean_diff[seat.agent].append(
                                sum(1.0 for c in taken_colours if c in LEAN_COLOURS)
                                / len(taken_colours)
                                - sum(
                                    weight
                                    * sum(1.0 for c in option if c in LEAN_COLOURS)
                                    / len(option)
                                    for option in options
                                )
                            )

                    window = next(
                        (name for name, lo, hi in WINDOWS if lo <= pick <= hi), None
                    )
                    if window is None:
                        continue
                    slot = off_lane[seat.agent][(pack, window)]
                    slot[1] += 1
                    if taken_colours <= top2:
                        continue
                    slot[0] += 1
                    declined = voluntary[seat.agent][(pack, window)]
                    declined[1] += 1
                    if any(
                        (c := colours(card)) and c <= top2 for card in legal[1:]
                    ):
                        declined[0] += 1

    reported = sorted(seats_seen)
    print("=" * 96)
    print(f"{label}   seats: " + ", ".join(f"{a}={seats_seen[a]}" for a in reported))

    print("\n  A. mana-base width of the built deck (share of decks, mean deck_score)")
    print(f"     {'agent':<12} " + "  ".join(f"{n} types" for n in range(1, 6)) + "   >=4")
    for agent in reported:
        counts = width_counts[agent]
        total = sum(counts.values())
        if not total:
            continue
        cells = []
        for n in range(1, 6):
            scores = width_scores[agent][n]
            share = f"{100 * counts.get(n, 0) / total:4.1f}%"
            mean = f"{sum(scores) / len(scores):+5.2f}" if scores else "    -"
            cells.append(f"{share} {mean}")
        wide = sum(v for k, v in counts.items() if k >= 4)
        print(f"     {agent:<12} " + " ".join(cells) + f"  {_pct(wide, total)}")

    print("\n  B. accumulated pool after each pack: effective colour count (80% coverage)")
    counts_header = "".join(f"{n:>8}" for n in range(1, 6))
    print(f"     {'agent':<12} {'after':<8} " + counts_header + "   top-2 share")
    for agent in reported:
        for cut in sorted(eff_counts[agent]):
            counts = eff_counts[agent][cut]
            total = sum(counts.values())
            row = "".join(f"{100 * counts.get(n, 0) / total:7.1f}%" for n in range(1, 6))
            shares = top2_share[agent][cut]
            mean = 100 * sum(shares) / len(shares) if shares else float("nan")
            head = agent if cut == 1 else ""
            print(f"     {head:<12} pack {cut:<3} {row}     {mean:5.1f}%")
        print()

    print("  C. off-lane rate (coloured picks outside the eventual top-2 colours), and")
    print("     of those, the share made with an on-colour card still in the pack")
    header = "  ".join(f"{name:>14}" for name, _, _ in WINDOWS)
    print(f"     {'agent':<12} {'pack':<5} {header}")
    for agent in reported:
        for pack in sorted({p for p, _ in off_lane[agent]}):
            cells = []
            for name, _, _ in WINDOWS:
                off, total = off_lane[agent][(pack, name)]
                vol, vol_total = voluntary[agent][(pack, name)]
                rate = f"{100 * off / total:4.1f}%" if total else "   - "
                share = f"{100 * vol / vol_total:4.1f}%" if vol_total else "   - "
                cells.append(f"{rate} ({share})")
            head = agent if pack == 1 else ""
            print(f"     {head:<12} {pack:<5} " + "  ".join(f"{c:>14}" for c in cells))
        print()

    print("  D. colour of off-lane picks: share taken vs share available in the pack,")
    print("     with the lift (taken/available). Availability is per decision, so a")
    print("     15-card pack and a 2-card pack weigh the same.")
    print(f"     {'agent':<12} " + "".join(f"{c:>15}" for c in WUBRG))
    for agent in reported:
        taken_total = sum(lean_taken[agent].values())
        avail_total = sum(lean_avail[agent].values())
        if not taken_total or not avail_total:
            continue
        cells = []
        for colour in WUBRG:
            took = lean_taken[agent][colour] / taken_total
            supply = lean_avail[agent][colour] / avail_total
            lift = f"{took / supply:4.2f}" if supply else "   -"
            cells.append(f"{100 * took:4.1f}/{100 * supply:4.1f} {lift}")
        print(f"     {agent:<12} " + "".join(f"{c:>15}" for c in cells))

    print()
    print(f"     lean towards {''.join(sorted(LEAN_COLOURS))}: mean(taken - available)")
    print(f"     {'agent':<12} {'diff':>9} {'se':>7} {'z':>7} {'picks':>8}")
    for agent in reported:
        mean, se = _mean_se(lean_diff[agent])
        if se != se:  # nan — too few off-lane picks to say anything
            continue
        z = mean / se if se else float("nan")
        print(f"     {agent:<12} {100 * mean:+8.2f}pp {100 * se:6.2f} {z:7.1f} "
              f"{len(lean_diff[agent]):8d}")
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
    parser.add_argument(
        "--agents", default=None,
        help="comma-separated mix labels to report (default: all seen)",
    )
    args = parser.parse_args()

    agents = tuple(a.strip() for a in args.agents.split(",")) if args.agents else None
    colours = ColourResolver(args.cards_path)
    for raw in args.drafts:
        label, path = parse_drafts_arg(raw)
        analyze(label, path, colours, agents)

    if colours.unresolved:
        print(
            f"unresolved card names: {len(colours.unresolved)} distinct "
            f"({sum(colours.unresolved.values())} lookups) — treated as colourless"
        )


if __name__ == "__main__":
    main()
