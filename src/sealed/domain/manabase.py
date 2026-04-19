"""Compute basic-land distribution for a 40-card sealed deck."""

from __future__ import annotations

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.domain.value_objects import WUBRG, ManaCost
from price_predictor.infrastructure.converted_card_parser import convert_mana_cost

DECK_SIZE = 40

COLOR_TO_LAND: dict[str, str] = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}


def compute_basic_lands(nonland_texts: list[ConvertedCardText]) -> dict[str, int]:
    """Compute basic land distribution for a 40-card deck.

    Args:
        nonland_texts: card text for each nonland slot, one entry per card
            (duplicates appear multiple times).

    Returns:
        Mapping ``basic land name → count`` summing to ``40 - len(nonland_texts)``.
        Each color with at least one pip gets at least 2 basics when there is
        room; the remainder is distributed proportional to per-color pip counts.
        Colorless decks get an even five-color split.
    """
    n_basics = DECK_SIZE - len(nonland_texts)
    pips = _count_color_pips(nonland_texts)

    if sum(pips.values()) == 0:
        pips = dict.fromkeys(WUBRG, 1)

    return _distribute_basics(pips, n_basics)


def _count_color_pips(nonland_texts: list[ConvertedCardText]) -> dict[str, int]:
    pips: dict[str, int] = dict.fromkeys(WUBRG, 0)
    for converted in nonland_texts:
        cost = _parse_card_mana_cost(converted)
        if cost is None:
            continue
        pips["W"] += cost.w
        pips["U"] += cost.u
        pips["B"] += cost.b
        pips["R"] += cost.r
        pips["G"] += cost.g
    return pips


def _parse_card_mana_cost(converted: ConvertedCardText) -> ManaCost | None:
    raw = converted.mana_cost_line()
    if raw is None:
        return None
    return ManaCost.parse(convert_mana_cost(raw))


def _distribute_basics(pips: dict[str, int], n_basics: int) -> dict[str, int]:
    active_colors = [c for c in WUBRG if pips[c] > 0]
    n_active = len(active_colors)

    min_per_color = 2 if 2 * n_active <= n_basics else 0
    remaining = n_basics - min_per_color * n_active
    remaining_pips = sum(pips[c] for c in active_colors)

    lands: dict[str, int] = {}
    for color in active_colors:
        if remaining_pips == pips[color]:
            extra = remaining
        else:
            extra = round(remaining * pips[color] / remaining_pips)
        extra = max(0, min(extra, remaining))
        count = min_per_color + extra
        if count > 0:
            lands[COLOR_TO_LAND[color]] = count
        remaining -= extra
        remaining_pips -= pips[color]

    return lands
