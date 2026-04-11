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

_MANA_SYMBOLS = {"W": 1, "U": 2, "B": 3, "R": 4, "G": 5}
_PRODUCE_SYMBOLS = {"W": 16, "U": 17, "B": 18, "R": 19, "G": 20}
_COLOR_FLAGS = {"W": 10, "U": 11, "B": 12, "R": 13, "G": 14}

_MANA_SYMBOL_RE = re.compile(r"\{([^}]+)\}")


def parse_deterministic_features(card_text: str) -> np.ndarray:
    """Parse converted card text and return a 32-element float32 feature array."""
    feats = np.zeros(32, dtype=np.float32)

    lines: dict[str, str] = {}
    activated_lines: list[str] = []

    for line in card_text.splitlines():
        line = line.strip()
        if line.startswith("activated"):
            activated_lines.append(line)
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip().lower()
            val = line[colon_idx + 1:].strip()
            if key not in lines:
                lines[key] = val

    # is_land (index 0)
    types_line = lines.get("types", "")
    if "land" in types_line.lower().split():
        feats[0] = 1.0

    # Mana cost parsing (indices 1-9)
    mana_cost = lines.get("mana cost", "")
    if mana_cost:
        for match in _MANA_SYMBOL_RE.finditer(mana_cost):
            symbol = match.group(1)
            if symbol in _MANA_SYMBOLS:
                feats[_MANA_SYMBOLS[symbol]] += 1.0
            elif symbol == "C":
                feats[6] += 1.0  # colorless pip
            elif symbol == "X":
                feats[8] += 1.0  # X count
            else:
                try:
                    feats[7] += float(symbol)  # generic mana
                except ValueError:
                    pass

        # Mana value = sum of colored + colorless + generic (X counts as 0)
        feats[9] = feats[1] + feats[2] + feats[3] + feats[4] + feats[5] + feats[6] + feats[7]

    # Devoid detection
    is_devoid = False
    for line in card_text.splitlines():
        if line.strip().lower() == "static: devoid":
            is_devoid = True
            break

    # Color flags (indices 10-15)
    if is_devoid:
        feats[15] = 1.0  # is_colorless
    elif mana_cost:
        has_color = False
        for sym, idx in _COLOR_FLAGS.items():
            if feats[_MANA_SYMBOLS[sym]] > 0:
                feats[idx] = 1.0
                has_color = True
        if not has_color:
            feats[15] = 1.0  # no colored pips → colorless
    else:
        feats[15] = 1.0  # no mana cost → colorless

    # Mana production from activated abilities (indices 16-22)
    for act_line in activated_lines:
        # Look for "add" in the activated ability text
        colon_idx = act_line.find(":")
        if colon_idx < 0:
            continue
        ability_text = act_line[colon_idx + 1:].lower()
        if "add" not in ability_text:
            continue

        # Find the add clause
        add_idx = ability_text.index("add")
        add_clause = ability_text[add_idx:]

        mana_count = 0
        for match in _MANA_SYMBOL_RE.finditer(add_clause):
            symbol = match.group(1).upper()
            if symbol in _PRODUCE_SYMBOLS:
                feats[_PRODUCE_SYMBOLS[symbol]] = 1.0
                mana_count += 1
            elif symbol == "C":
                feats[21] = 1.0  # produces C
                mana_count += 1

        # "or" in add clause means one mana of choice, not multiple
        if " or " in add_clause and mana_count > 1:
            mana_count = 1

        feats[22] = max(feats[22], float(mana_count))

    # Power/toughness (indices 23-24)
    pt_line = lines.get("power toughness", "")
    if pt_line:
        parts = pt_line.split("/")
        if len(parts) == 2:
            feats[23] = _parse_stat(parts[0].strip())
            feats[24] = _parse_stat(parts[1].strip())

    # Loyalty (index 25)
    loyalty_str = lines.get("loyalty", "")
    if loyalty_str:
        try:
            feats[25] = float(loyalty_str)
        except ValueError:
            pass

    # Indices 26-31 remain zero (padding)
    return feats


def _parse_stat(value: str) -> float:
    """Parse a power, toughness, or loyalty value. * and X → 0."""
    if value in ("*", "X", "x"):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
