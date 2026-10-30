"""Quick diagnostic: win rates by generation method, with Bo1/Bo3/Bo5/Bo7 simulation.

For each match, the games string (e.g. ``ABABBAB``) records every game's winner
in order. We can therefore "rewind" the match to what the result would have
been at Bo1/Bo3/Bo5 by walking the prefix until one side reaches the required
number of wins. Bo7 is the actual recorded outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from _lib.corpora import iter_matches

from price_predictor.domain.entities import Card
from price_predictor.infrastructure.converted_card_parser import parse_converted_file
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

DEFAULT_PATH = Path("output/sealed/match-outcomes.txt")
DEFAULT_CARDS_PATH = Path("output/cardsfolder")
BEST_OFS = [1, 3, 5, 7]  # match-length variants to simulate
COLOR_BUCKETS = (1, 2, 3, 4, 5)
WUBRG = ("W", "U", "B", "R", "G")
CREATURE_BUCKETS = ("<=13", "14", "15", "16", "17", "18", "19", ">=20")
MV_BUCKETS = (
    "<2.5", "2.5-2.8", "2.8-3.1", "3.1-3.4",
    "3.4-3.7", "3.7-4.0", ">=4.0",
)
CELL = 17


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


def creature_bucket(count: int) -> str:
    if count <= 13:
        return "<=13"
    if count >= 20:
        return ">=20"
    return str(count)


def mv_bucket(avg_mv: float) -> str:
    for upper, label in ((2.5, "<2.5"), (2.8, "2.5-2.8"), (3.1, "2.8-3.1"),
                         (3.4, "3.1-3.4"), (3.7, "3.4-3.7")):
        if avg_mv < upper:
            return label
    return "3.7-4.0" if avg_mv <= 4.0 else ">=4.0"


def deck_features(names: tuple[str, ...], card: Callable[[str], Card | None]) -> dict:
    """The deck's bucket under each per-deck table: colour count, the colours
    present (across mana costs), creature-count bucket and average nonland mana
    value bucket (None when no nonland has a cost)."""
    cards = [c for c in map(card, names) if c is not None]
    costs = [c.mana_cost for c in cards if c.mana_cost is not None]
    colors = frozenset(col for col in WUBRG for mc in costs if getattr(mc, col.lower()) > 0)
    mvs = [c.mana_cost.total_mana_value for c in cards
           if not c.is_land() and c.mana_cost is not None]
    return {
        "color_count": len(colors),
        "color_presence": colors,
        "creatures": creature_bucket(sum("Creature" in c.types for c in cards)),
        "mv": mv_bucket(sum(mvs) / len(mvs)) if mvs else None,
    }


def rate_cell(matches: int, wins: int) -> str:
    if matches == 0:
        return f"{'-':>{CELL}}"
    return f"{100 * wins / matches:>6.1f}% (n={matches:>5})"


def print_bucket_table(title: str, tally: dict, methods: list[str], buckets, labels,
                       note: str | None = None) -> None:
    """Win rate per (method, bucket), then an overall row pooling the methods."""
    print(title)
    if note:
        print(note)
    rule = "-" * (14 + CELL * len(buckets))
    print(f"{'method':<14}" + "".join(f"{label:>{CELL}}" for label in labels))
    print(rule)
    for method in methods:
        print(f"{method:<14}" + "".join(rate_cell(*tally[(method, b)]) for b in buckets))
    print(rule)
    overall = [[sum(tally[(m, b)][i] for m in methods) for i in (0, 1)] for b in buckets]
    print(f"{'Overall':<14}" + "".join(rate_cell(*cell) for cell in overall))
    print()


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

    deck_tables_enabled = args.cards_path.exists()
    locator = ConvertedCardLocator(args.cards_path) if deck_tables_enabled else None
    card_cache: dict[str, Card | None] = {}

    def card(name: str) -> Card | None:
        if name not in card_cache:
            text = locator.text_path(name)
            card_cache[name] = parse_converted_file(text) if text else None
        return card_cache[name]

    # feature -> (method, bucket) -> [matches, wins]
    deck_tallies: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # method -> best_of -> [matches, match_wins]
    method_stats: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: {bo: [0, 0] for bo in BEST_OFS}
    )
    # (first, second) -> [matches, wins_for_first], Bo7 only
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    flip_counts = {bo: 0 for bo in BEST_OFS if bo != 7}
    games_length_counts: dict[int, int] = defaultdict(int)
    n = 0

    for match in iter_matches(path):
        method_a, method_b, games = match.method_a, match.method_b, match.games
        if not games:
            continue
        n += 1
        games_length_counts[len(games)] += 1
        bo7_winner = winner_at_threshold(games, 4)

        for bo in BEST_OFS:
            w = winner_at_threshold(games, (bo + 1) // 2)  # Bo1 -> 1 ... Bo7 -> 4
            method_stats[method_a][bo][0] += 1
            method_stats[method_b][bo][0] += 1
            method_stats[method_a if w == "A" else method_b][bo][1] += 1
            if bo != 7 and w != bo7_winner:
                flip_counts[bo] += 1

        # Mirror matches arbitrarily credit the A side so the cell reads near
        # the ~50% baseline.
        first = min(method_a, method_b)
        cell = h2h[(first, max(method_a, method_b))]
        cell[0] += 1
        winning_method = method_a if bo7_winner == "A" else method_b
        if (bo7_winner == "A") if method_a == method_b else winning_method == first:
            cell[1] += 1

        if deck_tables_enabled:
            for side, deck in (("A", match.deck_a), ("B", match.deck_b)):
                method = method_a if side == "A" else method_b
                won = int(bo7_winner == side)
                features = deck_features(deck.cards, card)
                for feature, value in features.items():
                    buckets = value if feature == "color_presence" else (value,)
                    for bucket in buckets:
                        if bucket is not None:
                            tally = deck_tallies[feature][(method, bucket)]
                            tally[0] += 1
                            tally[1] += won

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
        method_stats, key=lambda m: -method_stats[m][7][1] / max(method_stats[m][7][0], 1)
    )
    for method in methods_sorted:
        stats = method_stats[method]
        print(f"{method:<14} {stats[7][0]:>10}" + "".join(
            f"  {100 * stats[bo][1] / max(stats[bo][0], 1):>7.1f}%" for bo in BEST_OFS))
    print()

    print("=== Bo-N vs Bo7 flip rate ===")
    print("(% of matches where the simulated Bo-N winner differs from the actual Bo7 winner)")
    for bo, flips in flip_counts.items():
        print(f"  Bo{bo} vs Bo7: {100 * flips / max(n, 1):5.2f}% flipped ({flips} of {n})")
    print()

    methods = sorted(method_stats)
    print("=== Head-to-head match win rate (Bo7, row beats column) ===")
    print(f"{'row \\ col':<14}" + "".join(f"{m:>{CELL}}" for m in methods))
    print("-" * (14 + CELL * len(methods)))
    for row in methods:
        cells = [f"{row:<14}"]
        for col in methods:
            first = min(row, col)
            matches, wins_for_first = h2h.get((first, max(row, col)), (0, 0))
            wins_for_row = wins_for_first if first == row else matches - wins_for_first
            cells.append(rate_cell(matches, wins_for_row))
        print("".join(cells))
    print()

    if not deck_tables_enabled:
        return
    for feature, title, buckets, labels, note in (
        ("color_count", "=== Win rate by method by deck color count (Bo7) ===",
         COLOR_BUCKETS, [f"{c}-color" for c in COLOR_BUCKETS], None),
        ("color_presence", "=== Win rate by method by color presence (Bo7) ===",
         WUBRG, WUBRG,
         "  (each cell counts every match where the deck contained that "
         "color; rows can sum to >100% of matches)"),
        ("creatures", "=== Win rate by method by creature count (Bo7) ===",
         CREATURE_BUCKETS, CREATURE_BUCKETS, None),
        ("mv", "=== Win rate by method by avg nonland mana value (Bo7) ===",
         MV_BUCKETS, MV_BUCKETS, None),
    ):
        if deck_tallies[feature]:
            print_bucket_table(title, deck_tallies[feature], methods, buckets, labels, note)


if __name__ == "__main__":
    main()
