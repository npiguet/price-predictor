"""Quick diagnostic: win rates by generation method, plus head-to-head matrix."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


PATH = Path("output/sealed/match-outcomes.txt")


def main() -> None:
    if not PATH.exists():
        print(f"Not found: {PATH}")
        return

    method_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"matches": 0, "match_wins": 0, "games": 0, "game_wins": 0}
    )
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"matches": 0, "wins_for_first": 0,
                 "games": 0, "game_wins_for_first": 0}
    )

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
            wa = games.count("A")
            wb = games.count("B")
            total = wa + wb
            if total == 0:
                continue
            n += 1

            method_stats[method_a]["matches"] += 1
            method_stats[method_a]["games"] += total
            method_stats[method_a]["game_wins"] += wa
            if wa > wb:
                method_stats[method_a]["match_wins"] += 1

            method_stats[method_b]["matches"] += 1
            method_stats[method_b]["games"] += total
            method_stats[method_b]["game_wins"] += wb
            if wb > wa:
                method_stats[method_b]["match_wins"] += 1

            # Head-to-head: always store as (sorted) pair, with stats from
            # the perspective of the first method in sorted order.
            if method_a == method_b:
                key = (method_a, method_b)
                h2h[key]["matches"] += 1
                h2h[key]["games"] += total
                # Mirror match — credit half to "first"
                h2h[key]["wins_for_first"] += 1 if wa > wb else 0
                h2h[key]["game_wins_for_first"] += wa
            else:
                first, second = sorted([method_a, method_b])
                key = (first, second)
                h2h[key]["matches"] += 1
                h2h[key]["games"] += total
                if first == method_a:
                    h2h[key]["wins_for_first"] += 1 if wa > wb else 0
                    h2h[key]["game_wins_for_first"] += wa
                else:
                    h2h[key]["wins_for_first"] += 1 if wb > wa else 0
                    h2h[key]["game_wins_for_first"] += wb

    print(f"Total matches parsed: {n}")
    print()

    print("=== Overall win rates by method ===")
    print(f"{'method':<14} {'matches':>8} {'match WR':>9} "
          f"{'games':>8} {'game WR':>9}")
    print("-" * 60)
    for method, s in sorted(
        method_stats.items(),
        key=lambda x: -x[1]["match_wins"] / max(x[1]["matches"], 1)
    ):
        match_wr = 100 * s["match_wins"] / max(s["matches"], 1)
        game_wr = 100 * s["game_wins"] / max(s["games"], 1)
        print(f"{method:<14} {s['matches']:>8} {match_wr:>8.1f}% "
              f"{s['games']:>8} {game_wr:>8.1f}%")
    print()

    print("=== Head-to-head match win rate (row beats column) ===")
    methods = sorted(method_stats.keys())
    print(f"{'row \\ col':<14}" + "".join(f"{m:>16}" for m in methods))
    print("-" * (14 + 16 * len(methods)))
    for row in methods:
        cells = [f"{row:<14}"]
        for col in methods:
            first, second = sorted([row, col])
            key = (first, second)
            s = h2h.get(key)
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


if __name__ == "__main__":
    main()
