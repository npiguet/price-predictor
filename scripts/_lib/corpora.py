"""Tolerant readers of the corpus files the analysis scripts consume.

Each wraps the typed parser ``src`` already owns. The scripts read corpora
still being appended to, often gzipped, so a malformed or truncated line is
skipped rather than raised.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from price_predictor.application.ridge_probes import load_labels
from sealed.domain.match import Side, match_winner
from sealed.infrastructure.match_data_loader import MatchOutcome, parse_match_outcome

__all__ = ["MatchOutcome", "Side", "card_column", "decided_winner", "iter_matches",
           "load_win_rates", "open_text"]


def open_text(path: Path):
    """Open a UTF-8 text file for reading, gunzipping a ``.gz`` one."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_matches(path: Path, limit: int | None = None) -> Iterator[MatchOutcome]:
    """The well-formed rows of a match-outcome file (plain or ``.gz``), in order."""

    def parsed():
        with open_text(path) as handle:
            for line in handle:
                try:
                    yield parse_match_outcome(line)
                except ValueError:
                    continue

    return islice(parsed(), limit)


def decided_winner(match: MatchOutcome) -> Side | None:
    """The side that won more games; None for a drawn or empty match."""
    return match_winner(match.wins_a, match.wins_b)


def load_win_rates(path: Path, min_in_deck: int = 0) -> dict[str, dict]:
    """``cards-win-rates.txt`` as ``name -> {column: value}`` (see ``load_labels``),
    keeping only cards seen in at least ``min_in_deck`` decks."""
    labels = load_labels(path)
    if min_in_deck <= 0:
        return labels
    return {name: row for name, row in labels.items()
            if row["wins_when_in_deck"] + row["losses_when_in_deck"] >= min_in_deck}


def card_column(path: Path, key: str, sign: float = 1.0) -> dict[str, float]:
    """One numeric column of a per-card CSV, keyed by its name column
    (``name``, ``card_name`` or ``card``). Empty when the file or column is
    missing; rows whose cell does not parse as a number are skipped."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        name_col = next((c for c in fields if c.lower() in ("name", "card_name", "card")), None)
        if name_col is None or key not in fields:
            return out
        for row in reader:
            try:
                out[row[name_col]] = sign * float(row[key])
            except (TypeError, ValueError):
                continue
    return out
