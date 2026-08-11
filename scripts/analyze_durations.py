"""Quick diagnostic: duration_s and throughput trends in match-outcomes.txt."""

from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path

PATH = Path("output/sealed/match-outcomes.txt")
CHUNK_SIZE = 100  # matches per trend bucket


def main() -> None:
    if not PATH.exists():
        print(f"Not found: {PATH}")
        return

    timestamps: list[datetime] = []
    durations: list[int] = []
    games_lengths: list[int] = []

    with open(PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 10:
                continue
            try:
                ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                games = parts[7]
                d = int(parts[9])
            except (ValueError, IndexError):
                continue
            timestamps.append(ts)
            durations.append(d)
            games_lengths.append(len(games))

    n = len(durations)
    if n == 0:
        print("No matches in file.")
        return

    print(f"Total matches: {n}")
    print()
    print("=== Duration distribution (seconds) ===")
    sd = sorted(durations)
    print(f"  mean:   {statistics.mean(durations):.1f}")
    print(f"  median: {statistics.median(durations):.0f}")
    print(f"  p90:    {sd[int(n * 0.90)]}")
    print(f"  p95:    {sd[int(n * 0.95)]}")
    print(f"  p99:    {sd[min(int(n * 0.99), n - 1)]}")
    print(f"  max:    {max(durations)}")
    print()

    print("=== Duration histogram ===")
    buckets = [(0, 10), (10, 20), (20, 30), (30, 60),
               (60, 120), (120, 300), (300, 10**9)]
    for lo, hi in buckets:
        count = sum(1 for d in durations if lo <= d < hi)
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        label = f"{lo:>3}-{(str(hi) if hi < 10**9 else 'inf'):<4}s"
        print(f"  {label}: {count:>5} ({pct:5.1f}%) {bar}")
    print()

    print("=== Games-per-match histogram (Bo7 = 4-7) ===")
    for length in range(2, 8):
        count = sum(1 for g in games_lengths if g == length)
        pct = 100 * count / n
        bar = "#" * int(pct / 2)
        print(f"  {length} games: {count:>5} ({pct:5.1f}%) {bar}")
    print()

    print(f"=== Mean duration per chunk of {CHUNK_SIZE} matches (chronological) ===")
    for i in range(0, n, CHUNK_SIZE):
        chunk = durations[i:i + CHUNK_SIZE]
        if not chunk:
            continue
        chunk_mean = statistics.mean(chunk)
        chunk_med = statistics.median(chunk)
        print(f"  matches {i:>5}-{i + len(chunk) - 1:>5}: "
              f"mean {chunk_mean:5.1f}s  median {chunk_med:5.0f}s  (n={len(chunk)})")
    print()

    print(f"=== Throughput per chunk of {CHUNK_SIZE} matches ===")
    for i in range(0, n, CHUNK_SIZE):
        chunk_ts = timestamps[i:i + CHUNK_SIZE]
        if len(chunk_ts) < 2:
            continue
        elapsed_s = (chunk_ts[-1] - chunk_ts[0]).total_seconds()
        if elapsed_s <= 0:
            continue
        rate = len(chunk_ts) / elapsed_s * 60
        print(f"  matches {i:>5}-{i + len(chunk_ts) - 1:>5}: "
              f"{rate:6.1f} matches/min over {elapsed_s/60:5.1f} min")
    print()

    # Cumulative throughput from start to each chunk boundary
    print(f"=== Cumulative throughput from start (every {CHUNK_SIZE} matches) ===")
    t0 = timestamps[0]
    for i in range(CHUNK_SIZE - 1, n, CHUNK_SIZE):
        elapsed_s = (timestamps[i] - t0).total_seconds()
        if elapsed_s <= 0:
            continue
        cum_rate = (i + 1) / elapsed_s * 60
        print(f"  through match {i:>5}: cumulative {cum_rate:6.2f} matches/min "
              f"over {elapsed_s/60:6.1f} min")
    # And final
    if (n - 1) % CHUNK_SIZE != 0:
        elapsed_s = (timestamps[-1] - t0).total_seconds()
        cum_rate = n / elapsed_s * 60
        print(f"  through match {n - 1:>5}: cumulative {cum_rate:6.2f} matches/min "
              f"over {elapsed_s/60:6.1f} min")


if __name__ == "__main__":
    main()
