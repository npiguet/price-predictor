"""Human-readable per-card score table from ``cards-played.txt``.

For every card name observed in the cards-played log, count:
  * wins_when_played   — winning side played at least one copy
  * wins_when_in_deck  — card was in the winning side's deck
  * losses_when_played — losing side played at least one copy
  * losses_when_in_deck — card was in the losing side's deck

The per-card score is

    (wins_played - losses_played) / (wins_in_deck + losses_in_deck)

bounded between -1 and +1. The numerator captures whether playing the
card tends to coincide with winning or losing (positive vs negative
influence); the denominator down-weights cards that are rarely actually
played, so a card's magnitude reflects both its effect and how often it
shows up. Output is sorted by score descending.
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

    def sort_key(self) -> tuple[float, int, int]:
        # score desc, wins_played desc, losses_played asc — encoded as an
        # ascending tuple so callers can sort with a single key function.
        return (-self.score, -self.wins_when_played, self.losses_when_played)


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


def _format_table(counts: dict[str, _Counts]) -> str:
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1].sort_key())
    data_rows = [
        (
            name,
            c.mana_cost or "",
            str(c.wins_when_played),
            str(c.wins_when_in_deck),
            str(c.losses_when_played),
            str(c.losses_when_in_deck),
            c.score_display,
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
        "Score",
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
    parser = argparse.ArgumentParser(
        description=(
            "Print a per-card score table from cards-played.txt, sorted by "
            "(wins_played - losses_played) / (wins_in_deck + losses_in_deck) "
            "descending."
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

    print(_format_table(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
