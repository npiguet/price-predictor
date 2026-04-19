"""Parse 32 deterministic game features from converted card text.

The feature layout is defined by named slot constants in
``sealed.domain.card_embedding_layout`` — edit those constants if you need
to reorganise the feature block, not the literal indices here.
"""

from __future__ import annotations

import numpy as np

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.domain.entities import Card
from price_predictor.domain.power_toughness import parse_combat_stat
from price_predictor.domain.value_objects import WUBRG, ManaCost
from price_predictor.infrastructure.converted_card_parser import (
    MANA_BRACE_RE,
    parse_converted_text,
    parse_mana_production,
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


def parse_deterministic_features(converted: ConvertedCardText) -> np.ndarray:
    """Parse converted card text and return a float32 feature array of length ``FEATURE_COUNT``.

    Returns the zero vector for inputs that aren't well-formed converted card
    text (no ``name:``/``types:`` line). The encoder feeds short test snippets
    through here, so this contract has to stay lenient.
    """
    feats = np.zeros(FEATURE_COUNT, dtype=np.float32)

    try:
        card = parse_converted_text(converted)
    except ValueError:
        return feats

    if card.is_land():
        feats[IS_LAND] = 1.0

    _fill_mana_cost_features(feats, card.mana_cost, converted)
    _fill_color_flags(feats, card)
    _fill_mana_production(feats, converted)
    _fill_combat_stats(feats, card)

    return feats


def _fill_mana_cost_features(
    feats: np.ndarray, mana_cost: ManaCost | None, converted: ConvertedCardText,
) -> None:
    if mana_cost is None:
        return
    pip_counts = (mana_cost.w, mana_cost.u, mana_cost.b, mana_cost.r, mana_cost.g)
    for color, count in zip(WUBRG, pip_counts):
        feats[_COLOR_PIP_BY_COLOR[color]] = float(count)
    feats[COLORLESS_PIP] = float(mana_cost.colorless_mana)
    feats[GENERIC] = float(mana_cost.generic_mana)
    feats[X_COUNT] = float(_count_x_in_cost(converted))
    feats[MANA_VALUE] = float(mana_cost.total_mana_value)


def _count_x_in_cost(converted: ConvertedCardText) -> int:
    cost_line = converted.mana_cost_line()
    if cost_line is None:
        return 0
    return sum(1 for m in MANA_BRACE_RE.finditer(cost_line) if m.group(1) == "X")


def _fill_color_flags(feats: np.ndarray, card: Card) -> None:
    if card.is_colorless():
        feats[IS_COLORLESS] = 1.0
        return

    mana_cost = card.mana_cost
    pip_counts = (mana_cost.w, mana_cost.u, mana_cost.b, mana_cost.r, mana_cost.g)
    for color, pips in zip(WUBRG, pip_counts):
        if pips > 0:
            feats[_COLOR_FLAG_BY_COLOR[color]] = 1.0


def _fill_mana_production(feats: np.ndarray, converted: ConvertedCardText) -> None:
    production = parse_mana_production(converted)
    for color in production.colors:
        feats[_PRODUCE_BY_COLOR[color]] = 1.0
    if production.produces_colorless:
        feats[PRODUCES_COLORLESS] = 1.0
    feats[MANA_PRODUCED] = float(production.max_mana_count)


def _fill_combat_stats(feats: np.ndarray, card: Card) -> None:
    feats[POWER] = parse_combat_stat(card.power)
    feats[TOUGHNESS] = parse_combat_stat(card.toughness)
    feats[LOYALTY] = parse_combat_stat(card.loyalty)
