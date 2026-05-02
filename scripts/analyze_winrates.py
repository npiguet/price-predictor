"""Quick diagnostic: win rates by generation method, with Bo1/Bo3/Bo5/Bo7 simulation.

For each match, the games string (e.g. ``ABABBAB``) records every game's winner
in order. We can therefore "rewind" the match to what the result would have
been at Bo1/Bo3/Bo5 by walking the prefix until one side reaches the required
number of wins. Bo7 is the actual recorded outcome.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from price_predictor.domain.entities import Card
from price_predictor.infrastructure.converted_card_parser import parse_converted_file
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator


DEFAULT_PATH = Path("output/sealed/match-outcomes.txt")
DEFAULT_CARDS_PATH = Path("output/cardsfolder")
BEST_OFS = [1, 3, 5, 7]  # match-length variants to simulate
COLOR_BUCKETS = (1, 2, 3, 4, 5)
WUBRG = ("W", "U", "B", "R", "G")


def winner_at_threshold(games: str, threshold: int) -> str:
    """Return 'A' or 'B' — winner of a best-of-(2*threshold-1) match.

    Walks the games string and returns whichever side first reaches
    ``threshold`` game wins. Always reachable for valid Bo7 data because
    one side always reaches 4 wins, which exceeds every threshold <= 4.
    """
    a = b = 0
    for ch in games:
        if ch == "A":
            a += 1
        else:
            b += 1
        if a == threshold:
            return "A"
        if b == threshold:
            return "B"
    return "A" if a > b else "B"  # fallback for malformed data


def deck_colors(
    card_names: list[str],
    card_cache: dict[str, Card | None],
    locator: ConvertedCardLocator,
) -> frozenset[str]:
    """Set of distinct WUBRG colors across nonland mana costs in the deck."""
    colors: set[str] = set()
    for name in card_names:
        if name not in card_cache:
            path = locator.text_path(name)
            card_cache[name] = parse_converted_file(path) if path else None
        card = card_cache[name]
        if card is None or card.mana_cost is None:
            continue
        mc = card.mana_cost
        if mc.w > 0:
            colors.add("W")
        if mc.u > 0:
            colors.add("U")
        if mc.b > 0:
            colors.add("B")
        if mc.r > 0:
            colors.add("R")
        if mc.g > 0:
            colors.add("G")
    return frozenset(colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to match-outcomes file (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--cards-path",
        type=Path,
        default=DEFAULT_CARDS_PATH,
        help=(
            "Path to converted cardsfolder, used for the per-color-count "
            f"win-rate table (default: {DEFAULT_CARDS_PATH})"
        ),
    )
    args = parser.parse_args()
    path: Path = args.path

    if not path.exists():
        print(f"Not found: {path}")
        return

    color_table_enabled = args.cards_path.exists()
    locator = ConvertedCardLocator(args.cards_path) if color_table_enabled else None
    card_cache: dict[str, Card | None] = {}
    # (method, n_colors) -> {matches, wins}
    color_stats: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"matches": 0, "wins": 0}
    )
    # (method, color_letter) -> {matches, wins}
    color_presence_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"matches": 0, "wins": 0}
    )

    # method -> {best_of -> {matches, match_wins}}
    method_stats: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: {bo: {"matches": 0, "match_wins": 0} for bo in BEST_OFS}
    )
    # best_of -> (first, second) -> {matches, wins_for_first}
    h2h: dict[int, dict[tuple[str, str], dict[str, int]]] = {
        bo: defaultdict(lambda: {"matches": 0, "wins_for_first": 0})
        for bo in BEST_OFS
    }
    flip_counts = {bo: 0 for bo in BEST_OFS if bo != 7}
    games_length_counts: dict[int, int] = defaultdict(int)
    n = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 10:
                continue
            method_a, method_b = parts[3], parts[4]
            games = parts[7]
            if not games:
                continue
            n += 1
            games_length_counts[len(games)] += 1

            first, second = sorted([method_a, method_b])
            h2h_key = (first, second)
            bo7_winner = winner_at_threshold(games, 4)

            for bo in BEST_OFS:
                threshold = (bo + 1) // 2  # Bo1 -> 1, Bo3 -> 2, Bo5 -> 3, Bo7 -> 4
                w = winner_at_threshold(games, threshold)
                method_stats[method_a][bo]["matches"] += 1
                method_stats[method_b][bo]["matches"] += 1
                if w == "A":
                    method_stats[method_a][bo]["match_wins"] += 1
                else:
                    method_stats[method_b][bo]["match_wins"] += 1
                if bo != 7 and w != bo7_winner:
                    flip_counts[bo] += 1

                # Head-to-head for this Bo. Mirror matches arbitrarily credit
                # the A side so the cell reads near the ~50% baseline.
                h2h[bo][h2h_key]["matches"] += 1
                if method_a == method_b:
                    if w == "A":
                        h2h[bo][h2h_key]["wins_for_first"] += 1
                else:
                    won_by_first = (
                        (w == "A" and method_a == first)
                        or (w == "B" and method_b == first)
                    )
                    if won_by_first:
                        h2h[bo][h2h_key]["wins_for_first"] += 1

            if color_table_enabled and locator is not None:
                deck_a = parts[5].split("|") if parts[5] else []
                deck_b = parts[6].split("|") if parts[6] else []
                colors_a = deck_colors(deck_a, card_cache, locator)
                colors_b = deck_colors(deck_b, card_cache, locator)
                color_stats[(method_a, len(colors_a))]["matches"] += 1
                if bo7_winner == "A":
                    color_stats[(method_a, len(colors_a))]["wins"] += 1
                color_stats[(method_b, len(colors_b))]["matches"] += 1
                if bo7_winner == "B":
                    color_stats[(method_b, len(colors_b))]["wins"] += 1
                for c in colors_a:
                    color_presence_stats[(method_a, c)]["matches"] += 1
                    if bo7_winner == "A":
                        color_presence_stats[(method_a, c)]["wins"] += 1
                for c in colors_b:
                    color_presence_stats[(method_b, c)]["matches"] += 1
                    if bo7_winner == "B":
                        color_presence_stats[(method_b, c)]["wins"] += 1

    print(f"Total matches parsed: {n}")
    print()

    print("=== Bo7 match length distribution ===")
    print("(how many games each match took - shorter = more lopsided, longer = closer)")
    for length in sorted(games_length_counts):
        count = games_length_counts[length]
        pct = 100 * count / max(n, 1)
        loser_wins = length - 4  # Bo7 ends as soon as one side reaches 4 wins
        print(f"  {length} games (4-{loser_wins}): {pct:5.2f}%  ({count:>6} of {n})")
    print()

    print("=== Match win rate by method, simulated across best-of-N ===")
    print(f"{'method':<14} {'instances':>10}"
          + "".join(f"  {'Bo' + str(bo):>8}" for bo in BEST_OFS))
    print("-" * (14 + 10 + len(BEST_OFS) * 10))
    methods_sorted = sorted(
        method_stats.keys(),
        key=lambda m: -method_stats[m][7]["match_wins"] / max(method_stats[m][7]["matches"], 1)
    )
    for method in methods_sorted:
        stats = method_stats[method]
        instances = stats[7]["matches"]
        cells = [f"{method:<14} {instances:>10}"]
        for bo in BEST_OFS:
            wr = 100 * stats[bo]["match_wins"] / max(stats[bo]["matches"], 1)
            cells.append(f"  {wr:>7.1f}%")
        print("".join(cells))
    print()

    print("=== Bo-N vs Bo7 flip rate ===")
    print("(% of matches where the simulated Bo-N winner differs from the actual Bo7 winner)")
    for bo in BEST_OFS:
        if bo == 7:
            continue
        pct = 100 * flip_counts[bo] / max(n, 1)
        print(f"  Bo{bo} vs Bo7: {pct:5.2f}% flipped ({flip_counts[bo]} of {n})")
    print()

    methods = sorted(method_stats.keys())
    print("=== Head-to-head match win rate (Bo7, row beats column) ===")
    print(f"{'row \\ col':<14}" + "".join(f"{m:>17}" for m in methods))
    print("-" * (14 + 17 * len(methods)))
    bo_h2h = h2h[7]
    for row in methods:
        cells = [f"{row:<14}"]
        for col in methods:
            first, second = sorted([row, col])
            key = (first, second)
            s = bo_h2h.get(key)
            if s is None or s["matches"] == 0:
                cells.append(f"{'-':>17}")
                continue
            wins_for_row = (
                s["wins_for_first"] if first == row
                else s["matches"] - s["wins_for_first"]
            )
            wr = 100 * wins_for_row / s["matches"]
            cells.append(f"{wr:>6.1f}% (n={s['matches']:>5})")
        print("".join(cells))
    print()

    if color_table_enabled and color_stats:
        print("=== Win rate by method by deck color count (Bo7) ===")
        print(
            f"{'method':<14}"
            + "".join(f"{f'{c}-color':>17}" for c in COLOR_BUCKETS)
        )
        print("-" * (14 + 17 * len(COLOR_BUCKETS)))
        for method in methods:
            cells = [f"{method:<14}"]
            for c in COLOR_BUCKETS:
                s = color_stats.get((method, c))
                if s is None or s["matches"] == 0:
                    cells.append(f"{'-':>17}")
                    continue
                wr = 100 * s["wins"] / s["matches"]
                cells.append(f"{wr:>6.1f}% (n={s['matches']:>5})")
            print("".join(cells))
        print("-" * (14 + 17 * len(COLOR_BUCKETS)))
        cells = [f"{'Overall':<14}"]
        for c in COLOR_BUCKETS:
            tot_m = sum(color_stats.get((m, c), {}).get("matches", 0) for m in methods)
            tot_w = sum(color_stats.get((m, c), {}).get("wins", 0) for m in methods)
            if tot_m == 0:
                cells.append(f"{'-':>17}")
                continue
            wr = 100 * tot_w / tot_m
            cells.append(f"{wr:>6.1f}% (n={tot_m:>5})")
        print("".join(cells))
        print()

    if color_table_enabled and color_presence_stats:
        print("=== Win rate by method by color presence (Bo7) ===")
        print(
            "  (each cell counts every match where the deck contained that "
            "color; rows can sum to >100% of matches)"
        )
        print(f"{'method':<14}" + "".join(f"{c:>17}" for c in WUBRG))
        print("-" * (14 + 17 * len(WUBRG)))
        for method in methods:
            cells = [f"{method:<14}"]
            for c in WUBRG:
                s = color_presence_stats.get((method, c))
                if s is None or s["matches"] == 0:
                    cells.append(f"{'-':>17}")
                    continue
                wr = 100 * s["wins"] / s["matches"]
                cells.append(f"{wr:>6.1f}% (n={s['matches']:>5})")
            print("".join(cells))
        print("-" * (14 + 17 * len(WUBRG)))
        cells = [f"{'Overall':<14}"]
        for c in WUBRG:
            tot_m = sum(color_presence_stats.get((m, c), {}).get("matches", 0) for m in methods)
            tot_w = sum(color_presence_stats.get((m, c), {}).get("wins", 0) for m in methods)
            if tot_m == 0:
                cells.append(f"{'-':>17}")
                continue
            wr = 100 * tot_w / tot_m
            cells.append(f"{wr:>6.1f}% (n={tot_m:>5})")
        print("".join(cells))
        print()


if __name__ == "__main__":
    main()
