"""Human-readable per-card score table from ``cards-played.txt``.

For every card name observed in the cards-played log, count:
  * wins_when_played   — winning side played at least one copy
  * wins_when_in_deck  — card was in the winning side's deck
  * losses_when_played — losing side played at least one copy
  * losses_when_in_deck — card was in the losing side's deck

The per-card raw score is

    (wins_played - losses_played) / (wins_in_deck + losses_in_deck)

bounded between -1 and +1. The numerator captures whether playing the
card tends to coincide with winning or losing (positive vs negative
influence); the denominator down-weights cards that are rarely actually
played, so a card's magnitude reflects both its effect and how often it
shows up.

The adjusted score applies Bayesian shrinkage with a pseudocount ``k``:

    adj = (wins_played - losses_played) / (wins_in_deck + losses_in_deck + k)

Low-n cards get pulled toward 0; cards with many observations stay
nearly identical to their raw score. ``k`` is configurable via
``--shrinkage-k`` (default 20).

The quantile column reports each card's rank-quantile of the adjusted
score, with 100.0% = best, 0.0% = worst. Tied cards share the average
rank, so the column is suitable as a regression target.

Output is sorted by adjusted score descending, with raw wins-played
desc, losses-played asc, and unplayed count asc as tiebreakers. The
final tiebreaker surfaces cards that were drafted into decks but never
actually played (e.g. ``Shackles`` had 637 in-deck observations and 0
plays) — within the score=0 tier they sort to the bottom, separating
them from score=0 cards that do see occasional play.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sealed.infrastructure.cards_played_reader import iter_rows
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

DEFAULT_CARDS_PLAYED = Path("output/sealed/cards-played.txt")
DEFAULT_CARDS_FOLDER = Path("output/cardsfolder")
_MANA_COST_PREFIX = "mana cost:"
_MANA_COST_LOOKUP_WORKERS = 32


@dataclass
class _Counts:
    wins_when_played: int = 0
    wins_when_in_deck: int = 0
    losses_when_played: int = 0
    losses_when_in_deck: int = 0
    mana_cost: Optional[str] = None

    @property
    def score(self) -> float:
        """Net-influence score in ``[-1, +1]`` per the module docstring."""
        in_deck_total = self.wins_when_in_deck + self.losses_when_in_deck
        if in_deck_total == 0:
            return 0.0
        return (self.wins_when_played - self.losses_when_played) / in_deck_total

    @property
    def score_display(self) -> str:
        return f"{self.score:+.4f}"

    @property
    def unplayed_count(self) -> int:
        """Times the card was in *some* deck but never played in the game.

        ``(wins_in_deck + losses_in_deck) - (wins_played + losses_played)``.
        Useful for surfacing dead cards: a card with high in-deck count
        and zero plays is one the AI consistently chooses to draft but
        never to cast.
        """
        in_deck = self.wins_when_in_deck + self.losses_when_in_deck
        played = self.wins_when_played + self.losses_when_played
        return in_deck - played

    def lift(self, k: float) -> float | None:
        """Shrunk winrate-when-played minus winrate-when-not-played.

        Adds ``k/2`` pseudo-wins and ``k/2`` pseudo-losses to each bucket
        (Beta prior pulling both rates toward 50%), so the lift shrinks
        toward 0 for low-observation cards.

        Returns ``None`` when either raw denominator is zero.
        """
        played_total = self.wins_when_played + self.losses_when_played
        if played_total == 0:
            return None
        not_played_wins = self.wins_when_in_deck - self.wins_when_played
        not_played_losses = self.losses_when_in_deck - self.losses_when_played
        not_played_total = not_played_wins + not_played_losses
        if not_played_total == 0:
            return None
        half_k = k / 2.0
        return (
            (self.wins_when_played + half_k) / (played_total + k)
            - (not_played_wins + half_k) / (not_played_total + k)
        )

    def adjusted_score(self, k: float) -> float:
        """Shrunk score with pseudocount ``k`` in the denominator.

        Low-observation cards are pulled toward 0; high-observation cards
        are nearly unchanged.
        """
        denom = self.wins_when_in_deck + self.losses_when_in_deck + k
        if denom == 0:
            return 0.0
        return (self.wins_when_played - self.losses_when_played) / denom


def _aggregate(cards_played_path: Path) -> dict[str, _Counts]:
    counts: dict[str, _Counts] = {}
    for row in iter_rows(cards_played_path):
        if row.winner == "A":
            winner_played, winner_deck = row.cards_played_a, row.cards_not_played_a
            loser_played, loser_deck = row.cards_played_b, row.cards_not_played_b
        else:
            winner_played, winner_deck = row.cards_played_b, row.cards_not_played_b
            loser_played, loser_deck = row.cards_played_a, row.cards_not_played_a

        winner_played_set = set(winner_played)
        winner_in_deck_set = winner_played_set | set(winner_deck)
        loser_played_set = set(loser_played)
        loser_in_deck_set = loser_played_set | set(loser_deck)

        for name in winner_in_deck_set:
            entry = counts.setdefault(name, _Counts())
            entry.wins_when_in_deck += 1
            if name in winner_played_set:
                entry.wins_when_played += 1
        for name in loser_in_deck_set:
            entry = counts.setdefault(name, _Counts())
            entry.losses_when_in_deck += 1
            if name in loser_played_set:
                entry.losses_when_played += 1

    return counts


def _read_mana_cost(path: Path) -> Optional[str]:
    """Read just the ``mana cost:`` line of a converted card file.

    Iterates lines and breaks at the first match — avoids loading the full
    file when only one line is needed. Returns ``None`` for cards whose
    converted text does not declare a mana cost (e.g. lands).
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip().lower()
            if stripped.startswith(_MANA_COST_PREFIX):
                value = line.split(":", 1)[1].strip()
                return value or None
    return None


def _attach_mana_costs(
    counts: dict[str, _Counts], locator: ConvertedCardLocator,
) -> int:
    """Resolve each card's mana cost in parallel. Returns the number of
    cards for which no converted .txt was found."""
    paths: dict[str, Path] = {}
    missing = 0
    for name in counts:
        path = locator.text_path(name)
        if path is None:
            missing += 1
            continue
        paths[name] = path

    if not paths:
        return missing

    with ThreadPoolExecutor(max_workers=_MANA_COST_LOOKUP_WORKERS) as pool:
        names = list(paths.keys())
        for name, cost in zip(
            names, pool.map(_read_mana_cost, (paths[n] for n in names)),
        ):
            counts[name].mana_cost = cost
    return missing


def _average_rank_quantiles(values: list[float]) -> list[float]:
    """Return rank-quantiles in ``[0, 1]`` matching the input order.

    Highest input value gets ``1.0``, lowest gets ``0.0``. Tied values
    share the average rank's quantile — necessary for using the quantile
    as a regression target where ties shouldn't be split arbitrarily.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    quantiles = [0.0] * n
    i = 0
    while i < n:
        j = i
        while (
            j + 1 < n
            and values[sorted_idx[j + 1]] == values[sorted_idx[i]]
        ):
            j += 1
        avg_rank = (i + j) / 2.0
        q = avg_rank / (n - 1)
        for pos in range(i, j + 1):
            quantiles[sorted_idx[pos]] = q
        i = j + 1
    return quantiles


def _format_table(counts: dict[str, _Counts], shrinkage_k: float) -> str:
    names = list(counts.keys())
    adj_scores = {name: counts[name].adjusted_score(shrinkage_k) for name in names}
    lifts = {name: counts[name].lift(shrinkage_k) for name in names}
    quantiles = dict(zip(
        names,
        _average_rank_quantiles([adj_scores[n] for n in names]),
    ))

    def sort_key(item: tuple[str, _Counts]) -> tuple[float, int, int, int]:
        name, c = item
        # adj score desc, wins_played desc, losses_played asc, unplayed asc.
        return (
            -adj_scores[name],
            -c.wins_when_played,
            c.losses_when_played,
            c.unplayed_count,
        )

    sorted_items = sorted(counts.items(), key=sort_key)
    data_rows = [
        (
            name,
            c.mana_cost or "",
            str(c.wins_when_played),
            str(c.wins_when_in_deck),
            str(c.losses_when_played),
            str(c.losses_when_in_deck),
            str(c.unplayed_count),
            c.score_display,
            f"{adj_scores[name]:+.4f}",
            f"{quantiles[name] * 100:.1f}%",
            f"{lifts[name] * 100:+.2f}%" if lifts[name] is not None else "",
        )
        for name, c in sorted_items
    ]

    headers = (
        "Card",
        "Mana Cost",
        "Wins Played",
        "Wins In Deck",
        "Losses Played",
        "Losses In Deck",
        "Unplayed",
        "Score",
        f"Adj Score (k={shrinkage_k:g})",
        "Quantile",
        "Lift",
    )
    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    fmt = "  ".join(
        f"{{:<{w}}}" if i < 2 else f"{{:>{w}}}"
        for i, w in enumerate(widths)
    )
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt.format(*row) for row in data_rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 on stdout/stderr so card names with macrons, accents,
    # and other non-Latin-1 characters round-trip when redirected on
    # Windows (where the default codec is cp1252).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Print a per-card score table from cards-played.txt. Sorted by "
            "the Bayesian-shrunk adjusted score descending; also reports the "
            "raw score and the rank quantile of the adjusted score."
        ),
    )
    parser.add_argument(
        "cards_played", nargs="?", type=Path, default=DEFAULT_CARDS_PLAYED,
        help=f"Path to cards-played.txt (default: {DEFAULT_CARDS_PLAYED})",
    )
    parser.add_argument(
        "--cards-folder", type=Path, default=DEFAULT_CARDS_FOLDER,
        help=(
            "Converted card corpus root for mana-cost lookup "
            f"(default: {DEFAULT_CARDS_FOLDER})"
        ),
    )
    parser.add_argument(
        "--shrinkage-k", type=float, default=20.0,
        help=(
            "Pseudocount added to the adjusted-score denominator "
            "(default: 20). Higher values pull low-observation cards "
            "more aggressively toward 0."
        ),
    )
    args = parser.parse_args(argv)

    if not args.cards_played.exists():
        print(f"Error: {args.cards_played} does not exist", file=sys.stderr)
        return 1

    counts = _aggregate(args.cards_played)
    if not counts:
        print(f"No card-play data found in {args.cards_played}", file=sys.stderr)
        return 0

    locator = ConvertedCardLocator(args.cards_folder)
    missing = _attach_mana_costs(counts, locator)
    if missing:
        print(
            f"Note: {missing}/{len(counts)} card(s) had no converted .txt "
            f"under {args.cards_folder}; mana cost left blank for those.",
            file=sys.stderr,
        )

    print(_format_table(counts, args.shrinkage_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
