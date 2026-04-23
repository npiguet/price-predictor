"""Quick diagnostic: win rates by generation method, with Bo1/Bo3/Bo5/Bo7 simulation.

For each match, the games string (e.g. ``ABABBAB``) records every game's winner
in order. We can therefore "rewind" the match to what the result would have
been at Bo1/Bo3/Bo5 by walking the prefix until one side reaches the required
number of wins. Bo7 is the actual recorded outcome.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


PATH = Path("output/sealed/match-outcomes.txt")
BEST_OFS = [1, 3, 5, 7]  # match-length variants to simulate


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


def main() -> None:
    if not PATH.exists():
        print(f"Not found: {PATH}")
        return

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

    with open(PATH, encoding="utf-8") as f:
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
    for bo in BEST_OFS:
        print(f"=== Head-to-head match win rate (Bo{bo}, row beats column) ===")
        print(f"{'row \\ col':<14}" + "".join(f"{m:>16}" for m in methods))
        print("-" * (14 + 16 * len(methods)))
        bo_h2h = h2h[bo]
        for row in methods:
            cells = [f"{row:<14}"]
            for col in methods:
                first, second = sorted([row, col])
                key = (first, second)
                s = bo_h2h.get(key)
                if s is None or s["matches"] == 0:
                    cells.append(f"{'-':>16}")
                    continue
                wins_for_row = (
                    s["wins_for_first"] if first == row
                    else s["matches"] - s["wins_for_first"]
                )
                wr = 100 * wins_for_row / s["matches"]
                cells.append(f"{wr:>6.1f}% (n={s['matches']:>4})")
            print("".join(cells))
        print()


if __name__ == "__main__":
    main()
