"""Aggregate round-robin evaluation outcomes into a typed result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundRobinResults:
    """Per-pool and aggregate win rates for an N×N round-robin evaluation."""

    n_pools: int
    n_matches: int
    total_games: int
    a_win_rates: list[float]
    b_win_rates: list[float]
    pool_deltas: list[float]
    a_aggregate_win_rate: float
    b_aggregate_win_rate: float

    @classmethod
    def empty(cls, n_pools: int, n_matches: int, total_games: int) -> RoundRobinResults:
        return cls(
            n_pools=n_pools,
            n_matches=n_matches,
            total_games=total_games,
            a_win_rates=[],
            b_win_rates=[],
            pool_deltas=[],
            a_aggregate_win_rate=0.0,
            b_aggregate_win_rate=0.0,
        )

    def format_report(self) -> str:
        lines = [
            "",
            "=== Evaluation Results ===",
            f"Pools: {self.n_pools}  |  Matches: {self.n_matches}  |  Games: {self.total_games}",
            f"Scorer aggregate win rate: {self.a_aggregate_win_rate:.1%}",
            f"Forge  aggregate win rate: {self.b_aggregate_win_rate:.1%}",
        ]

        if self.a_win_rates:
            lines.append("")
            lines.append("Per-pool comparison (scorer vs Forge from same pool):")
            for i, (ar, br, delta) in enumerate(
                zip(self.a_win_rates, self.b_win_rates, self.pool_deltas)
            ):
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"  Pool {i + 1:2d}: scorer {ar:.1%}  forge {br:.1%}  delta {sign}{delta:.1%}"
                )
            mean_delta = sum(self.pool_deltas) / len(self.pool_deltas)
            sign = "+" if mean_delta >= 0 else ""
            lines.append(f"Mean delta: {sign}{mean_delta:.1%}")

        return "\n".join(lines)


def aggregate_results(outcome_files: list[Path], n_pools: int) -> RoundRobinResults:
    """Aggregate round-robin evaluation outcomes from per-worker outcome files.

    Outcomes are written in row-major order: match index ``k`` corresponds to
    A deck ``k // n_pools`` vs B deck ``k % n_pools``.
    """
    outcomes = _read_outcomes(outcome_files)
    n_matches = len(outcomes)
    total_games = sum(wa + wb for wa, wb in outcomes)

    if n_pools == 0 or n_matches == 0:
        return RoundRobinResults.empty(n_pools, n_matches, total_games)

    a_win_rates, b_win_rates = _per_pool_win_rates(outcomes, n_pools)
    pool_deltas = [a - b for a, b in zip(a_win_rates, b_win_rates)]

    total_a_wins = sum(wa for wa, _ in outcomes)
    total_b_wins = sum(wb for _, wb in outcomes)
    a_aggregate = total_a_wins / max(total_games, 1)
    b_aggregate = total_b_wins / max(total_games, 1)

    return RoundRobinResults(
        n_pools=n_pools,
        n_matches=n_matches,
        total_games=total_games,
        a_win_rates=a_win_rates,
        b_win_rates=b_win_rates,
        pool_deltas=pool_deltas,
        a_aggregate_win_rate=a_aggregate,
        b_aggregate_win_rate=b_aggregate,
    )


def _read_outcomes(outcome_files: list[Path]) -> list[tuple[int, int]]:
    outcomes: list[tuple[int, int]] = []
    for f in outcome_files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(";")
            outcomes.append((int(parts[0]), int(parts[1])))
    return outcomes


def _per_pool_win_rates(
    outcomes: list[tuple[int, int]], n_pools: int,
) -> tuple[list[float], list[float]]:
    a_wins = [0] * n_pools
    a_games = [0] * n_pools
    b_wins = [0] * n_pools
    b_games = [0] * n_pools

    for k, (wa, wb) in enumerate(outcomes):
        i = k // n_pools
        j = k % n_pools
        games = wa + wb
        a_wins[i] += wa
        a_games[i] += games
        b_wins[j] += wb
        b_games[j] += games

    a_rates = [a_wins[i] / max(a_games[i], 1) for i in range(n_pools)]
    b_rates = [b_wins[j] / max(b_games[j], 1) for j in range(n_pools)]
    return a_rates, b_rates
