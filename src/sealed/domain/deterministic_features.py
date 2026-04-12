"""Parse 32 deterministic game features from converted card text.

Feature index mapping (within the 32-element array):
  0       is_land
  1-5     W/U/B/R/G pip counts
  6       colorless pip count (C)
  7       generic mana
  8       X pip count
  9       mana value
  10-14   is_white..is_green (binary, zeroed if devoid)
  15      is_colorless
  16-20   produces W/U/B/R/G (binary, from activated abilities)
  21      produces C (binary)
  22      mana_count (total mana produced per activation)
  23      power
  24      toughness
  25      loyalty
  26-31   zero padding
"""

from __future__ import annotations

import re

import numpy as np

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import WUBRG, ManaCost
from price_predictor.infrastructure.converted_card_parser import parse_converted_text

_COLOR_FLAGS = {c: i + 10 for i, c in enumerate(WUBRG)}     # W=10, U=11, B=12, R=13, G=14
_PRODUCE_SYMBOLS = {c: i + 16 for i, c in enumerate(WUBRG)} # W=16, U=17, B=18, R=19, G=20

_MANA_BRACE_RE = re.compile(r"\{([^}]+)\}")
_DEVOID_LINE = "static: devoid"


def parse_deterministic_features(card_text: str) -> np.ndarray:
    """Parse converted card text and return a 32-element float32 feature array.

    Returns the zero vector for inputs that aren't well-formed converted card
    text (no ``name:``/``types:`` line). The encoder feeds short test snippets
    through here, so this contract has to stay lenient.
    """
    feats = np.zeros(32, dtype=np.float32)

    try:
        card = parse_converted_text(card_text)
    except ValueError:
        return feats

    if _is_land(card):
        feats[0] = 1.0

    _fill_mana_cost_features(feats, card.mana_cost, card_text)
    _fill_color_flags(feats, card.mana_cost, _has_devoid(card_text))
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
    feats[1] = float(mana_cost.w)
    feats[2] = float(mana_cost.u)
    feats[3] = float(mana_cost.b)
    feats[4] = float(mana_cost.r)
    feats[5] = float(mana_cost.g)
    feats[6] = float(mana_cost.colorless_mana)
    feats[7] = float(mana_cost.generic_mana)
    feats[8] = float(_count_x_in_cost(card_text))
    feats[9] = feats[1] + feats[2] + feats[3] + feats[4] + feats[5] + feats[6] + feats[7]


def _count_x_in_cost(card_text: str) -> int:
    """Count X braces in the mana cost line (preserves the original feature semantics)."""
    for line in card_text.splitlines():
        if line.strip().lower().startswith("mana cost:"):
            return sum(1 for m in _MANA_BRACE_RE.finditer(line) if m.group(1) == "X")
    return 0


def _fill_color_flags(
    feats: np.ndarray, mana_cost: ManaCost | None, is_devoid: bool,
) -> None:
    if is_devoid or mana_cost is None:
        feats[15] = 1.0
        return

    color_pips = (mana_cost.w, mana_cost.u, mana_cost.b, mana_cost.r, mana_cost.g)
    has_color = False
    for color, pips in zip(WUBRG, color_pips):
        if pips > 0:
            feats[_COLOR_FLAGS[color]] = 1.0
            has_color = True
    if not has_color:
        feats[15] = 1.0


def _has_devoid(card_text: str) -> bool:
    return any(line.strip().lower() == _DEVOID_LINE for line in card_text.splitlines())


def _fill_mana_production(feats: np.ndarray, card_text: str) -> None:
    for activated_line in _activated_lines(card_text):
        add_clause = _extract_add_clause(activated_line)
        if add_clause is None:
            continue

        mana_count = 0
        for match in _MANA_BRACE_RE.finditer(add_clause):
            symbol = match.group(1).upper()
            if symbol in _PRODUCE_SYMBOLS:
                feats[_PRODUCE_SYMBOLS[symbol]] = 1.0
                mana_count += 1
            elif symbol == "C":
                feats[21] = 1.0
                mana_count += 1

        if " or " in add_clause and mana_count > 1:
            mana_count = 1

        feats[22] = max(feats[22], float(mana_count))


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
    feats[23] = _parse_stat(card.power) if card.power else 0.0
    feats[24] = _parse_stat(card.toughness) if card.toughness else 0.0
    feats[25] = _parse_stat(card.loyalty) if card.loyalty else 0.0


def _parse_stat(value: str) -> float:
    """Parse a power, toughness, or loyalty value. * and X → 0."""
    if value in ("*", "X", "x"):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
