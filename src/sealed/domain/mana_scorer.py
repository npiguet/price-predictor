"""mana_scorer: pure-domain mana-base quality scoring for sealed deck analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass

# ─── Value objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipCounts:
    """Per-color tally of mana requirements across a set of cards."""
    counts: dict[str, float]


@dataclass(frozen=True)
class IdealDistribution:
    """Target mana source counts per color, derived from pip counts."""
    ideal: dict[str, float]


@dataclass(frozen=True)
class ActualSourceCounts:
    """Actual mana-producing sources per color in the deck."""
    sources: dict[str, float]


@dataclass(frozen=True)
class ManaScore:
    """Scalar quality measure of a deck's mana base."""
    score: float    # [0.0, 1.0]
    reward: float   # 2*score - 1, in [-1.0, 1.0]
    l1_error: float
    n_lands: int


# ─── Regex patterns ───────────────────────────────────────────────────────────

# Match a single mana symbol in a mana cost string.
# Groups: (color | phyrexian color/P | hybrid color1/color2 | generic | colorless C | variable)
_MANA_SYMBOL_RE = re.compile(
    r"\{([WUBRGC])"              # single colored or colorless: {W}, {U}, {C}
    r"|([WUBRGC])/P"             # Phyrexian: {W/P}, {R/P}
    r"|([WUBRGC])/([WUBRGC])"   # Hybrid: {G/R}, {W/U}
    r"|\d+"                      # generic: {1}, {2}, {3}...
    r"|X"                        # variable: {X}
    r"}"                         # close brace (consumed as part of alternation anchor)
)

# Match tap-for-mana ability line: "activated[N]: {T}: add <clause>"
_MANA_ABILITY_RE = re.compile(
    r"activated\[\d+\]:\s*\{T\}:\s*add\s+(.+)",
    re.IGNORECASE,
)

# Extract individual color symbols from an "add" clause
_ADD_COLOR_RE = re.compile(r"\{([WUBRGC])\}")


# ─── T012: count_pips() ───────────────────────────────────────────────────────

def count_pips(card_texts: list[str]) -> PipCounts:
    """Count total mana pips from all provided card texts (non-land spells).

    One 'mana cost:' line is parsed per card face. Faces are separated by an
    'ALTERNATE' section marker (used by split, adventure, and similar layouts).
    This avoids summing duplicate colour variants on Baldur's Gate companion
    cards, which carry six 'mana cost:' lines for the same face.
    """
    totals: dict[str, float] = {}

    for text in card_texts:
        seen_cost_in_face = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper() == "ALTERNATE":
                seen_cost_in_face = False  # new face — allow next mana cost line
                continue
            if seen_cost_in_face:
                continue
            low = stripped.lower()
            if not low.startswith("mana cost:"):
                continue
            cost_part = stripped[len("mana cost:"):].strip()
            _accumulate_pips(cost_part, totals)
            seen_cost_in_face = True  # skip any further cost lines in this face

    return PipCounts(counts=totals)


def _accumulate_pips(cost_str: str, totals: dict[str, float]) -> None:
    """Parse a mana cost string and add pip values to totals."""
    pos = 0
    while pos < len(cost_str):
        if cost_str[pos] != "{":
            pos += 1
            continue
        # Find matching close brace
        end = cost_str.find("}", pos)
        if end == -1:
            break
        symbol = cost_str[pos + 1: end]  # content between braces
        pos = end + 1

        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2:
                left, right = parts[0].upper(), parts[1].upper()
                if right == "P":
                    # Phyrexian: {W/P} → +0.5 to left color
                    if left in "WUBRGC":
                        totals[left] = totals.get(left, 0.0) + 0.5
                else:
                    # Hybrid: {G/R} → +0.5 to each color
                    for c in (left, right):
                        if c in "WUBRGC":
                            totals[c] = totals.get(c, 0.0) + 0.5
        else:
            upper = symbol.upper()
            if upper in ("W", "U", "B", "R", "G"):
                totals[upper] = totals.get(upper, 0.0) + 1.0
            elif upper == "C":
                totals["C"] = totals.get("C", 0.0) + 1.0
            # else: generic digit or X → ignored


# ─── T013: compute_ideal_distribution() ──────────────────────────────────────

def compute_ideal_distribution(pip_counts: PipCounts) -> IdealDistribution:
    """Compute ideal mana source distribution from pip counts.

    Formula (FR-007):
        colors_present = {c : pip_counts[c] > 0}
        n_colors = len(colors_present)
        total_pips = sum(pip_counts[c] for c in colors_present)
        ideal[c] = 2 + (17 - 2 * n_colors) * pip_counts[c] / total_pips
    """
    colors_present = {c: v for c, v in pip_counts.counts.items() if v > 0}
    if not colors_present:
        return IdealDistribution(ideal={})

    n_colors = len(colors_present)
    total_pips = sum(colors_present.values())
    remaining = 17 - 2 * n_colors

    ideal: dict[str, float] = {
        c: 2.0 + remaining * v / total_pips
        for c, v in colors_present.items()
    }
    return IdealDistribution(ideal=ideal)


# ─── T014: count_actual_sources() ────────────────────────────────────────────

def count_actual_sources(land_texts: list[str]) -> ActualSourceCounts:
    """Count actual mana sources from land card texts.

    Scans each land for 'activated[N]: {T}: add ...' lines (FR-008).
    Extracts distinct color symbols {W/U/B/R/G/C} per ability line.
    Each distinct symbol → +1 to that color.
    """
    sources: dict[str, float] = {}

    for text in land_texts:
        for line in text.splitlines():
            m = _MANA_ABILITY_RE.match(line.strip())
            if not m:
                continue
            add_clause = m.group(1)
            # Extract distinct color symbols from the add clause
            colors_found = set(_ADD_COLOR_RE.findall(add_clause))
            for c in colors_found:
                sources[c.upper()] = sources.get(c.upper(), 0.0) + 1.0

    return ActualSourceCounts(sources=sources)


# ─── 014: compute_mana_value() ───────────────────────────────────────────────

def compute_mana_value(cost_str: str) -> float:
    """Return the mana value (CMC) of a brace-format cost string.

    Rules (FR-007 / data-model.md):
      {W},{U},{B},{R},{G},{C}  → +1
      {N} generic digit        → +N
      {W/P} phyrexian          → +1
      {G/R} hybrid             → +1
      {X}                      → +0
    """
    total = 0.0
    pos = 0
    while pos < len(cost_str):
        if cost_str[pos] != "{":
            pos += 1
            continue
        end = cost_str.find("}", pos)
        if end == -1:
            break
        symbol = cost_str[pos + 1: end]
        pos = end + 1

        if "/" in symbol:
            # Phyrexian or hybrid — counts as 1 pip regardless
            total += 1.0
        elif symbol.upper() in ("W", "U", "B", "R", "G", "C"):
            total += 1.0
        elif symbol.upper() == "X":
            pass  # X contributes 0
        else:
            # Generic integer
            try:
                total += float(symbol)
            except ValueError:
                pass
    return total


# ─── Explicit mana feature extraction ───────────────────────────────────────

def count_generic(cost_str: str) -> float:
    """Return the total generic mana in a brace-format cost string.

    {3}{W}{W} → 3.0   {R} → 0.0   {2}{2} → 4.0   '' → 0.0
    """
    total = 0.0
    pos = 0
    while pos < len(cost_str):
        if cost_str[pos] != "{":
            pos += 1
            continue
        end = cost_str.find("}", pos)
        if end == -1:
            break
        symbol = cost_str[pos + 1: end]
        pos = end + 1
        if "/" not in symbol and symbol.upper() not in ("W", "U", "B", "R", "G", "C", "X"):
            try:
                total += float(symbol)
            except ValueError:
                pass
    return total


def count_x(cost_str: str) -> int:
    """Return the number of {X} symbols in a brace-format cost string.

    {X}{R} → 1   {X}{X}{G} → 2   {3}{W} → 0
    """
    count = 0
    pos = 0
    while pos < len(cost_str):
        if cost_str[pos] != "{":
            pos += 1
            continue
        end = cost_str.find("}", pos)
        if end == -1:
            break
        symbol = cost_str[pos + 1: end]
        pos = end + 1
        if symbol == "X":
            count += 1
    return count


def count_mana_produced(card_text: str) -> dict[str, float]:
    """Count mana pips produced per color by a single card's tap abilities.

    Unlike count_actual_sources, this counts actual pips rather than
    deduplicating per ability. So {T}: add {C}{C} → {"C": 2.0}.

    Scans for 'activated[N]: {T}: add ...' lines and tallies all color
    symbols found in the add clause (without dedup).
    """
    sources: dict[str, float] = {}
    for line in card_text.splitlines():
        m = _MANA_ABILITY_RE.match(line.strip())
        if not m:
            continue
        add_clause = m.group(1)
        for c in _ADD_COLOR_RE.findall(add_clause):
            key = c.upper()
            sources[key] = sources.get(key, 0.0) + 1.0
    return sources


_PIP_COLORS = ("W", "U", "B", "R", "G", "C")
_PIP_NORM = {"W": 8.0, "U": 8.0, "B": 8.0, "R": 8.0, "G": 8.0, "C": 3.0}
_GENERIC_NORM = 15.0
_X_NORM = 3.0
_MV_NORM = 16.0
_PRODUCED_NORM = 3.0


def extract_mana_features(card_text: str) -> list[float]:
    """Return the 15-element raw mana feature vector for a single card.

    Layout:
      0–5:  pip counts W/U/B/R/G/C  (from count_pips)
      6:    generic mana count       (from count_generic)
      7:    X count                  (from count_x)
      8:    mana value               (from compute_mana_value)
      9–14: mana produced W/U/B/R/G/C (from count_mana_produced)
    """
    pips = count_pips([card_text])
    produced = count_mana_produced(card_text)

    # Extract mana cost line (face-aware: stop after first per face)
    cost_str = ""
    seen_cost = False
    for line in card_text.splitlines():
        stripped = line.strip()
        if stripped.upper() == "ALTERNATE":
            seen_cost = False
            continue
        if seen_cost:
            continue
        if stripped.lower().startswith("mana cost:"):
            cost_str = stripped[len("mana cost:"):].strip()
            seen_cost = True

    features: list[float] = []
    for color in _PIP_COLORS:
        features.append(float(pips.counts.get(color, 0.0)))
    features.append(count_generic(cost_str))
    features.append(float(count_x(cost_str)))
    features.append(compute_mana_value(cost_str))
    for color in _PIP_COLORS:
        features.append(produced.get(color, 0.0))

    return features


def normalize_mana_features(raw: list[float]) -> list[float]:
    """Return the 15-element normalized mana feature vector (approx [0, 1]).

    Uses the same layout as extract_mana_features.
    """
    norm = []
    # 0–4: pip counts WUBRG ÷8
    for i in range(5):
        norm.append(min(raw[i] / 8.0, 1.0))
    # 5: pip count C ÷3
    norm.append(min(raw[5] / 3.0, 1.0))
    # 6: generic ÷15
    norm.append(min(raw[6] / _GENERIC_NORM, 1.0))
    # 7: X count ÷3
    norm.append(min(raw[7] / _X_NORM, 1.0))
    # 8: mana value ÷16
    norm.append(min(raw[8] / _MV_NORM, 1.0))
    # 9–14: mana produced ÷3
    for i in range(9, 15):
        norm.append(min(raw[i] / _PRODUCED_NORM, 1.0))
    return norm


# ─── T015: compute_mana_score() ──────────────────────────────────────────────

_ALL_COLORS = ("W", "U", "B", "R", "G", "C")


def compute_mana_score(
    ideal: IdealDistribution,
    actual: ActualSourceCounts,
    n_lands: int,
) -> ManaScore:
    """Compute mana quality score (FR-009/FR-010).

    l1_error = sum(|actual[c] - ideal[c]| for c in all_colors)
    score    = max(0.0, 1.0 - (l1_error + |n_lands - 17|) / 17.0)
    reward   = 2 * score - 1
    """
    l1_error = sum(
        abs(actual.sources.get(c, 0.0) - ideal.ideal.get(c, 0.0))
        for c in _ALL_COLORS
    )
    score = max(0.0, 1.0 - (l1_error + abs(n_lands - 17)) / 17.0)
    reward = 2.0 * score - 1.0

    return ManaScore(score=score, reward=reward, l1_error=l1_error, n_lands=n_lands)
