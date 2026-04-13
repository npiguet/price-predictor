"""Parse 32 deterministic game features from converted card text.

The feature layout is defined by named slot constants in
``sealed.domain.card_embedding_layout`` — edit those constants if you need
to reorganise the feature block, not the literal indices here.
"""

from __future__ import annotations

import numpy as np

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import WUBRG, ManaCost
from price_predictor.infrastructure.converted_card_parser import (
    MANA_BRACE_RE,
    extract_mana_cost_line,
    parse_converted_text,
)
from sealed.domain.card_embedding_layout import (
    COLOR_FLAGS,
    COLOR_PIPS,
    COLORLESS_PIP,
    FEATURE_COUNT,
    GENERIC,
    IS_COLORLESS,
    IS_LAND,
    LOYALTY,
    MANA_PRODUCED,
    MANA_VALUE,
    POWER,
    PRODUCES_COLORLESS,
    PRODUCES_COLORS,
    TOUGHNESS,
    X_COUNT,
)

_COLOR_PIP_BY_COLOR: dict[str, int] = dict(
    zip(WUBRG, range(COLOR_PIPS.start, COLOR_PIPS.stop))
)
_COLOR_FLAG_BY_COLOR: dict[str, int] = dict(
    zip(WUBRG, range(COLOR_FLAGS.start, COLOR_FLAGS.stop))
)
_PRODUCE_BY_COLOR: dict[str, int] = dict(
    zip(WUBRG, range(PRODUCES_COLORS.start, PRODUCES_COLORS.stop))
)


def parse_deterministic_features(card_text: str) -> np.ndarray:
    """Parse converted card text and return a float32 feature array of length ``FEATURE_COUNT``.

    Returns the zero vector for inputs that aren't well-formed converted card
    text (no ``name:``/``types:`` line). The encoder feeds short test snippets
    through here, so this contract has to stay lenient.
    """
    feats = np.zeros(FEATURE_COUNT, dtype=np.float32)

    try:
        card = parse_converted_text(card_text)
    except ValueError:
        return feats

    if _is_land(card):
        feats[IS_LAND] = 1.0

    _fill_mana_cost_features(feats, card.mana_cost, card_text)
    _fill_color_flags(feats, card)
    _fill_mana_production(feats, card_text)
    _fill_combat_stats(feats, card)

    return feats


def _is_land(card: Card) -> bool:
    return any(t.lower() == "land" for t in card.types)


def _fill_mana_cost_features(
    feats: np.ndarray, mana_cost: ManaCost | None, card_text: str,
) -> None:
    if mana_cost is None:
        return
    pip_counts = (mana_cost.w, mana_cost.u, mana_cost.b, mana_cost.r, mana_cost.g)
    for color, count in zip(WUBRG, pip_counts):
        feats[_COLOR_PIP_BY_COLOR[color]] = float(count)
    feats[COLORLESS_PIP] = float(mana_cost.colorless_mana)
    feats[GENERIC] = float(mana_cost.generic_mana)
    feats[X_COUNT] = float(_count_x_in_cost(card_text))
    feats[MANA_VALUE] = float(mana_cost.total_mana_value)


def _count_x_in_cost(card_text: str) -> int:
    cost_line = extract_mana_cost_line(card_text)
    if cost_line is None:
        return 0
    return sum(1 for m in MANA_BRACE_RE.finditer(cost_line) if m.group(1) == "X")


def _fill_color_flags(feats: np.ndarray, card: Card) -> None:
    mana_cost = card.mana_cost
    if _has_devoid(card) or mana_cost is None:
        feats[IS_COLORLESS] = 1.0
        return

    pip_counts = (mana_cost.w, mana_cost.u, mana_cost.b, mana_cost.r, mana_cost.g)
    has_color = False
    for color, pips in zip(WUBRG, pip_counts):
        if pips > 0:
            feats[_COLOR_FLAG_BY_COLOR[color]] = 1.0
            has_color = True
    if not has_color:
        feats[IS_COLORLESS] = 1.0


def _has_devoid(card: Card) -> bool:
    return "devoid" in (card.oracle_text or "").lower()


def _fill_mana_production(feats: np.ndarray, card_text: str) -> None:
    for activated_line in _activated_lines(card_text):
        add_clause = _extract_add_clause(activated_line)
        if add_clause is None:
            continue

        mana_count = 0
        for match in MANA_BRACE_RE.finditer(add_clause):
            symbol = match.group(1).upper()
            if symbol in _PRODUCE_BY_COLOR:
                feats[_PRODUCE_BY_COLOR[symbol]] = 1.0
                mana_count += 1
            elif symbol == "C":
                feats[PRODUCES_COLORLESS] = 1.0
                mana_count += 1

        if " or " in add_clause and mana_count > 1:
            mana_count = 1

        feats[MANA_PRODUCED] = max(feats[MANA_PRODUCED], float(mana_count))


def _activated_lines(card_text: str) -> list[str]:
    return [
        line.strip()
        for line in card_text.splitlines()
        if line.strip().startswith("activated")
    ]


def _extract_add_clause(activated_line: str) -> str | None:
    colon_idx = activated_line.find(":")
    if colon_idx < 0:
        return None
    ability_text = activated_line[colon_idx + 1:].lower()
    if "add" not in ability_text:
        return None
    return ability_text[ability_text.index("add"):]


def _fill_combat_stats(feats: np.ndarray, card: Card) -> None:
    feats[POWER] = _parse_stat(card.power) if card.power else 0.0
    feats[TOUGHNESS] = _parse_stat(card.toughness) if card.toughness else 0.0
    feats[LOYALTY] = _parse_stat(card.loyalty) if card.loyalty else 0.0


def _parse_stat(value: str) -> float:
    """Parse a power, toughness, or loyalty value. * and X → 0."""
    if value in ("*", "X", "x"):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
