"""Aggregate round-robin evaluation outcomes into a typed result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundRobinOutcome:
    """One match in a round-robin: A's wins and B's wins (best-of-N)."""

    wins_a: int
    wins_b: int

    @property
    def total_games(self) -> int:
        return self.wins_a + self.wins_b


@dataclass(frozen=True)
class RoundRobinResults:
    """Per-pool and aggregate win rates for an N×N round-robin evaluation.

    Each rate is computed at two granularities:
    - **Per-game**: fraction of individual games won (e.g. 4 of 7 games in a
      Bo7 match).
    - **Per-match**: fraction of matches won outright (A wins a match iff
      `wins_a > wins_b`).
    """

    n_pools: int
    n_matches: int
    total_games: int

    a_win_rates: list[float]
    b_win_rates: list[float]
    pool_deltas: list[float]
    a_aggregate_win_rate: float
    b_aggregate_win_rate: float

    a_match_win_rates: list[float]
    b_match_win_rates: list[float]
    pool_match_deltas: list[float]
    a_aggregate_match_win_rate: float
    b_aggregate_match_win_rate: float

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
            a_match_win_rates=[],
            b_match_win_rates=[],
            pool_match_deltas=[],
            a_aggregate_match_win_rate=0.0,
            b_aggregate_match_win_rate=0.0,
        )

    def format_report(self) -> str:
        lines = [
            "",
            "=== Evaluation Results ===",
            f"Pools: {self.n_pools}  |  Matches: {self.n_matches}  |  Games: {self.total_games}",
            "",
            f"{'':<28}{'Per-game':>10}{'Per-match':>12}",
            f"{'Scorer aggregate win rate:':<28}"
            f"{self.a_aggregate_win_rate:>10.1%}"
            f"{self.a_aggregate_match_win_rate:>12.1%}",
            f"{'Forge  aggregate win rate:':<28}"
            f"{self.b_aggregate_win_rate:>10.1%}"
            f"{self.b_aggregate_match_win_rate:>12.1%}",
        ]

        if self.a_win_rates:
            lines.append("")
            lines.append("Per-pool comparison (scorer vs Forge from same pool):")
            lines.append(
                f"{'':<11}{'----- per-game -----':<24}{'----- per-match -----'}"
            )
            lines.append(
                f"{'':<11}"
                f"{'scorer':>7}{'forge':>7}{'delta':>8}   "
                f"{'scorer':>7}{'forge':>7}{'delta':>8}"
            )
            for i in range(len(self.a_win_rates)):
                ar = self.a_win_rates[i]
                br = self.b_win_rates[i]
                d = self.pool_deltas[i]
                am = self.a_match_win_rates[i]
                bm = self.b_match_win_rates[i]
                dm = self.pool_match_deltas[i]
                lines.append(
                    f"  Pool {i + 1:2d}: "
                    f"{ar:>7.1%}{br:>7.1%}{_signed_pct(d):>8}   "
                    f"{am:>7.1%}{bm:>7.1%}{_signed_pct(dm):>8}"
                )
            mean_delta = sum(self.pool_deltas) / len(self.pool_deltas)
            mean_match_delta = (
                sum(self.pool_match_deltas) / len(self.pool_match_deltas)
            )
            lines.append(
                f"Mean delta:  per-game {_signed_pct(mean_delta)}"
                f"   per-match {_signed_pct(mean_match_delta)}"
            )

        return "\n".join(lines)


def _signed_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1%}"


def aggregate_results(outcome_files: list[Path], n_pools: int) -> RoundRobinResults:
    """Aggregate round-robin evaluation outcomes from per-worker outcome files.

    Outcomes are written in row-major order: match index ``k`` corresponds to
    A deck ``k // n_pools`` vs B deck ``k % n_pools``.
    """
    outcomes = _read_outcomes(outcome_files)
    n_matches = len(outcomes)
    total_games = sum(o.total_games for o in outcomes)

    if n_pools == 0 or n_matches == 0:
        return RoundRobinResults.empty(n_pools, n_matches, total_games)

    a_win_rates, b_win_rates = _per_pool_win_rates(outcomes, n_pools)
    pool_deltas = [a - b for a, b in zip(a_win_rates, b_win_rates)]

    total_a_wins = sum(o.wins_a for o in outcomes)
    total_b_wins = sum(o.wins_b for o in outcomes)
    a_aggregate = total_a_wins / max(total_games, 1)
    b_aggregate = total_b_wins / max(total_games, 1)

    a_match_rates, b_match_rates = _per_pool_match_win_rates(outcomes, n_pools)
    pool_match_deltas = [a - b for a, b in zip(a_match_rates, b_match_rates)]
    a_match_wins = sum(1 for o in outcomes if o.wins_a > o.wins_b)
    b_match_wins = sum(1 for o in outcomes if o.wins_b > o.wins_a)
    a_match_aggregate = a_match_wins / max(n_matches, 1)
    b_match_aggregate = b_match_wins / max(n_matches, 1)

    return RoundRobinResults(
        n_pools=n_pools,
        n_matches=n_matches,
        total_games=total_games,
        a_win_rates=a_win_rates,
        b_win_rates=b_win_rates,
        pool_deltas=pool_deltas,
        a_aggregate_win_rate=a_aggregate,
        b_aggregate_win_rate=b_aggregate,
        a_match_win_rates=a_match_rates,
        b_match_win_rates=b_match_rates,
        pool_match_deltas=pool_match_deltas,
        a_aggregate_match_win_rate=a_match_aggregate,
        b_aggregate_match_win_rate=b_match_aggregate,
    )


def _read_outcomes(outcome_files: list[Path]) -> list[RoundRobinOutcome]:
    outcomes: list[RoundRobinOutcome] = []
    for f in outcome_files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(";")
            outcomes.append(RoundRobinOutcome(int(parts[0]), int(parts[1])))
    return outcomes


def _per_pool_win_rates(
    outcomes: list[RoundRobinOutcome], n_pools: int,
) -> tuple[list[float], list[float]]:
    a_wins = [0] * n_pools
    a_games = [0] * n_pools
    b_wins = [0] * n_pools
    b_games = [0] * n_pools

    for k, outcome in enumerate(outcomes):
        i = k // n_pools
        j = k % n_pools
        a_wins[i] += outcome.wins_a
        a_games[i] += outcome.total_games
        b_wins[j] += outcome.wins_b
        b_games[j] += outcome.total_games

    a_rates = [a_wins[i] / max(a_games[i], 1) for i in range(n_pools)]
    b_rates = [b_wins[j] / max(b_games[j], 1) for j in range(n_pools)]
    return a_rates, b_rates


def _per_pool_match_win_rates(
    outcomes: list[RoundRobinOutcome], n_pools: int,
) -> tuple[list[float], list[float]]:
    a_wins = [0] * n_pools
    a_matches = [0] * n_pools
    b_wins = [0] * n_pools
    b_matches = [0] * n_pools

    for k, outcome in enumerate(outcomes):
        i = k // n_pools
        j = k % n_pools
        a_matches[i] += 1
        b_matches[j] += 1
        if outcome.wins_a > outcome.wins_b:
            a_wins[i] += 1
        elif outcome.wins_b > outcome.wins_a:
            b_wins[j] += 1

    a_rates = [a_wins[i] / max(a_matches[i], 1) for i in range(n_pools)]
    b_rates = [b_wins[j] / max(b_matches[j], 1) for j in range(n_pools)]
    return a_rates, b_rates
