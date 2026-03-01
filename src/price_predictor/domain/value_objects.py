"""Value objects for the card price predictor domain."""

from __future__ import annotations

from dataclasses import dataclass

COLORS = frozenset({"W", "U", "B", "R", "G"})

# Two-character hybrid pairs (e.g., WU, WB, UB, UR, UG, BR, BG, RG, RW, GW, GU)
_HYBRID_PAIRS = {
    a + b for a in COLORS for b in COLORS if a != b
}

# Phyrexian mana (e.g., WP, UP, BP, RP, GP)
_PHYREXIAN = {c + "P" for c in COLORS}

# Two-or-colored hybrid (e.g., W2, U2, B2, R2, G2)
_TWO_HYBRID = {c + "2" for c in COLORS}


@dataclass(frozen=True)
class ManaCost:
    """Parsed representation of a mana cost in Forge format.

    Forge mana costs are space-separated shards like "2 W W" or "X R".
    """

    total_mana_value: float
    generic_mana: int
    colorless_mana: int
    w: int
    u: int
    b: int
    r: int
    g: int
    color_count: int
    has_x: bool
    has_hybrid: bool
    has_phyrexian: bool

    @staticmethod
    def parse(raw: str) -> ManaCost | None:
        """Parse a Forge mana cost string into a ManaCost value object.

        Returns None if the cost is "no cost" (card cannot be cast).
        """
        stripped = raw.strip()
        if not stripped or stripped == "no cost":
            return None

        shards = stripped.split()

        total_cmc = 0.0
        generic = 0
        colorless = 0
        color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
        has_x = False
        has_hybrid = False
        has_phyrexian = False

        for shard in shards:
            upper = shard.upper()

            if upper == "X":
                has_x = True
                # X contributes 0 to CMC
            elif upper in COLORS:
                color_counts[upper] += 1
                total_cmc += 1
            elif upper == "C":
                # Colorless mana — distinct from generic
                colorless += 1
                total_cmc += 1
            elif upper == "S":
                # Snow mana — no color
                total_cmc += 1
            elif upper in _HYBRID_PAIRS:
                has_hybrid = True
                total_cmc += 1
                # Each color gets 0.5 for counting purposes
                # but we track as present (1) for the multi-hot encoding
                for c in upper:
                    if c in COLORS:
                        color_counts[c] = max(color_counts[c], 1)
            elif upper in _PHYREXIAN:
                has_phyrexian = True
                total_cmc += 1
                color_char = upper[0]
                color_counts[color_char] = max(color_counts[color_char], 1)
            elif upper in _TWO_HYBRID:
                has_hybrid = True
                total_cmc += 2
                color_char = upper[0]
                color_counts[color_char] = max(color_counts[color_char], 1)
            else:
                # Try to parse as a generic mana number
                try:
                    value = int(shard)
                    generic += value
                    total_cmc += value
                except ValueError:
                    # Unknown shard — skip silently
                    pass

        colors_present = sum(1 for v in color_counts.values() if v > 0)

        return ManaCost(
            total_mana_value=total_cmc,
            generic_mana=generic,
            colorless_mana=colorless,
            w=color_counts["W"],
            u=color_counts["U"],
            b=color_counts["B"],
            r=color_counts["R"],
            g=color_counts["G"],
            color_count=colors_present,
            has_x=has_x,
            has_hybrid=has_hybrid,
            has_phyrexian=has_phyrexian,
        )
