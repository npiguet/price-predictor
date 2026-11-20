"""Per-card win/loss counters for ``train-encoder``: the single streaming
pass over ``cards-played.txt`` that fills them (FR-010), the color identity
each deck is sliced by, and the drop of cards with no converted text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from sealed.application.encoder_training_common import _log
from sealed.domain.encoder_model import COLOR_ORDER
from sealed.domain.match import Side
from sealed.infrastructure.cards_played_reader import CardsPlayedRow
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

# ── Per-card counters and labels (replaces v1 CardLabel) ────────────────

# Letter → index into the per-color counter lists (WUBRG order). Note
# that for a color-lift head, ``head_idx - _COLOR_LIFT_OFFSET`` already
# equals this index.
_COLOR_INDEX: dict[str, int] = {color: i for i, color in enumerate(COLOR_ORDER)}
_N_COLORS: int = len(COLOR_ORDER)


def _empty_color_counts() -> list[int]:
    return [0] * _N_COLORS


@dataclass
class CardCounters:
    """All raw counters maintained per card during aggregation.

    The four primary counters and the four ``@play`` subset counters
    track win/loss × played/in-deck; the four per-color counter lists
    hold the same, sliced by each of the WUBRG colors (indexed by
    ``_COLOR_INDEX``). The ``@draw`` counterparts are derivable by
    subtraction; the cast-lift "not played" counterparts are derivable
    from ``wins_when_in_deck - wins_when_played`` (and the loss analog).
    """

    wins_when_played: int = 0
    losses_when_played: int = 0
    wins_when_in_deck: int = 0
    losses_when_in_deck: int = 0
    wins_when_played_at_play: int = 0
    losses_when_played_at_play: int = 0
    wins_when_in_deck_at_play: int = 0
    losses_when_in_deck_at_play: int = 0
    wins_when_played_with: list[int] = field(default_factory=_empty_color_counts)
    losses_when_played_with: list[int] = field(default_factory=_empty_color_counts)
    wins_when_in_deck_with: list[int] = field(default_factory=_empty_color_counts)
    losses_when_in_deck_with: list[int] = field(default_factory=_empty_color_counts)


# ── Color extraction helper (FR-010b, Decision D-16) ────────────────────

_MANA_SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")
_COLOR_LETTERS: frozenset[str] = frozenset(COLOR_ORDER)


def _colors_from_mana_cost(line: str | None) -> set[str]:
    """Return the WUBRG colors present in a converted ``mana cost:`` line.

    Hybrid (``{W/U}``), Phyrexian (``{W/P}``), and mono-hybrid (``{2/W}``)
    symbols all contribute their color letter. Generic (``{2}``),
    colorless (``{C}``), and `X` (``{X}``) contribute nothing. Cards with
    no ``mana cost:`` line (``line is None``) contribute no colors.
    """
    if not line:
        return set()
    colors: set[str] = set()
    for symbol in _MANA_SYMBOL_PATTERN.findall(line):
        for ch in symbol.upper():
            if ch in _COLOR_LETTERS:
                colors.add(ch)
    return colors


# ── Aggregation (FR-010: primary + @play + per-color counters) ─────────


def _record_side(
    counters: dict[str, CardCounters],
    deck_played: list[str],
    deck_not_played: list[str],
    deck_colors: frozenset[int],
    *,
    won: bool,
    at_play: bool,
) -> None:
    """Fold one side's deck of one game into ``counters``.

    ``deck_played`` lists distinct card names that entered the
    battlefield/stack; ``deck_not_played`` lists distinct names that
    stayed in the deck (the two sets are disjoint per FR-004).
    ``deck_colors`` is the deck's WUBRG color set as ``_COLOR_INDEX``
    values. For each card the four primary counters, the four ``@play``
    subset counters, and the four per-color counter slots (one per
    color in ``deck_colors``) are incremented in a single pass over the
    deck.
    """
    played_set = set(deck_played)
    in_deck_set = played_set | set(deck_not_played)
    for name in in_deck_set:
        c = counters.get(name)
        if c is None:
            c = counters[name] = CardCounters()
        played = name in played_set
        if won:
            c.wins_when_in_deck += 1
            if played:
                c.wins_when_played += 1
            if at_play:
                c.wins_when_in_deck_at_play += 1
                if played:
                    c.wins_when_played_at_play += 1
            for color in deck_colors:
                c.wins_when_in_deck_with[color] += 1
                if played:
                    c.wins_when_played_with[color] += 1
        else:
            c.losses_when_in_deck += 1
            if played:
                c.losses_when_played += 1
            if at_play:
                c.losses_when_in_deck_at_play += 1
                if played:
                    c.losses_when_played_at_play += 1
            for color in deck_colors:
                c.losses_when_in_deck_with[color] += 1
                if played:
                    c.losses_when_played_with[color] += 1


def _aggregate(
    rows: Iterable[CardsPlayedRow], locator: ConvertedCardLocator,
) -> dict[str, CardCounters]:
    """Single streaming pass over ``cards-played.txt`` (FR-010).

    For every card observed in either side's deck, fills the four
    primary, four ``@play``, and 4×5 per-color counters. Each card's
    color identity (as ``_COLOR_INDEX`` values) is resolved lazily from
    its converted ``mana cost:`` line and cached per card; each game's
    deck-color set is the union over its contents. Cards with no
    converted text contribute an empty color set (and are pruned
    afterwards by :func:`_drop_missing_cards`).
    """
    counters: dict[str, CardCounters] = {}
    color_cache: dict[str, frozenset[int]] = {}

    def colors_of(name: str) -> frozenset[int]:
        cached = color_cache.get(name)
        if cached is not None:
            return cached
        converted = locator.load_text(name)
        cost_line = converted.mana_cost_line() if converted is not None else None
        result = frozenset(_COLOR_INDEX[c] for c in _colors_from_mana_cost(cost_line))
        color_cache[name] = result
        return result

    def deck_color_set(played: list[str], not_played: list[str]) -> frozenset[int]:
        colors: set[int] = set()
        for name in played:
            colors |= colors_of(name)
        for name in not_played:
            colors |= colors_of(name)
        return frozenset(colors)

    for row in rows:
        a_won = row.winner is Side.A
        a_at_play = row.starter is Side.A
        _record_side(
            counters, row.cards_played_a, row.cards_not_played_a,
            deck_color_set(row.cards_played_a, row.cards_not_played_a),
            won=a_won, at_play=a_at_play,
        )
        _record_side(
            counters, row.cards_played_b, row.cards_not_played_b,
            deck_color_set(row.cards_played_b, row.cards_not_played_b),
            won=not a_won, at_play=not a_at_play,
        )
    return counters


# ── Missing-card check (FR-023d) ────────────────────────────────────────

_MISSING_CARD_DISPLAY_CAP: int = 20


def _drop_missing_cards(
    counters: dict[str, CardCounters], locator: ConvertedCardLocator,
) -> int:
    """Run after aggregation, before label-map construction.

    Cards referenced by ``cards-played.txt`` that have no converted
    ``.txt`` under the cards folder cannot be tokenized, so they're
    dropped from ``counters`` (and therefore from the label map, the
    train/val split, and the dataset). Missing cards are *reported* via
    a log warning naming up to ``_MISSING_CARD_DISPLAY_CAP`` of them
    plus the total — but they do not block the run. Returns the number
    of cards dropped.

    Cards with zero observations in both sides' decks won't be in
    ``counters`` at all (aggregation only inserts on first sighting), so
    this only ever drops cards the model would otherwise have trained
    on. Any per-color counters accrued for a dropped card are discarded
    with its ``CardCounters``; since :func:`_aggregate`'s ``colors_of``
    returns ``∅`` for cards with no converted text, a missing card never
    contributed a color to any deck's color set in the first place.
    """
    missing = sorted(
        name for name in counters if locator.text_path(name) is None
    )
    if not missing:
        return 0
    for name in missing:
        del counters[name]
    head = missing[:_MISSING_CARD_DISPLAY_CAP]
    suffix = (
        f" (and {len(missing) - _MISSING_CARD_DISPLAY_CAP} more)"
        if len(missing) > _MISSING_CARD_DISPLAY_CAP else ""
    )
    _log(
        f"WARNING: {len(missing)} card(s) referenced by cards-played.txt "
        f"have no converted .txt under the cards folder; dropping them from "
        f"training. First {len(head)}: " + ", ".join(head) + suffix
        + ". Run python -m price_predictor convert to add them."
    )
    return len(missing)
