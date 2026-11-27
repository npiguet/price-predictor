"""Value objects for the card price predictor domain."""

from __future__ import annotations

from dataclasses import dataclass, field

from price_predictor.domain.card_taxonomy import VALID_RARITIES

WUBRG: tuple[str, ...] = ("W", "U", "B", "R", "G")
COLORS = frozenset(WUBRG)

RECOGNIZED_FORMATS = (
    "standard", "pioneer", "modern", "brawl", "legacy",
    "vintage", "pauper", "commander", "penny", "oathbreaker",
)

# Normalization constants for the cheapest-printing year: (year - YEAR_BASE) / YEAR_RANGE.
# 1993 → ~0.029, 2026 → 1.0. Used by both the sklearn feature engineer and
# the price transformer's printing-data encoder so they stay in lockstep.
YEAR_BASE = 1992
YEAR_RANGE = 34.0


@dataclass(frozen=True)
class PrintingData:
    """Printing data: reserve list, rarity, printings count, cheapest-printing year,
    legalities, ABU-only."""

    is_reserved: bool = False
    rarity: str = "rare"
    printings_count: int = 1
    cheapest_printing_year: int = 1993
    legalities: list[str] = field(default_factory=list)
    is_abu_only: bool = False

    def __post_init__(self) -> None:
        if self.rarity not in VALID_RARITIES:
            raise ValueError(
                f"rarity must be one of {sorted(VALID_RARITIES)}, got '{self.rarity}'"
            )
        self._check_counts()
        for fmt in self.legalities:
            if fmt not in RECOGNIZED_FORMATS:
                raise ValueError(
                    f"legality '{fmt}' not in RECOGNIZED_FORMATS"
                )

    def _check_counts(self) -> None:
        if self.printings_count < 1:
            raise ValueError(f"printings_count must be >= 1, got {self.printings_count}")
        if self.cheapest_printing_year < 1993:
            raise ValueError(
                f"cheapest_printing_year must be >= 1993, got {self.cheapest_printing_year}"
            )

    @classmethod
    def defaults(cls) -> PrintingData:
        """Return a PrintingData with all default values."""
        return cls(
            is_reserved=False,
            rarity="rare",
            printings_count=1,
            cheapest_printing_year=1993,
            legalities=list(RECOGNIZED_FORMATS),
        )

    def normalized_cheapest_printing_year(self) -> float:
        """Return ``cheapest_printing_year`` mapped to ``[0, 1]`` via the shared
        YEAR_BASE/YEAR_RANGE."""
        return min(max((self.cheapest_printing_year - YEAR_BASE) / YEAR_RANGE, 0.0), 1.0)

# Two-character hybrid pairs (e.g., WU, WB, UB, UR, UG, BR, BG, RG, RW, GW, GU)
_HYBRID_PAIRS = {
    a + b for a in COLORS for b in COLORS if a != b
}

# Phyrexian mana (e.g., WP, UP, BP, RP, GP)
_PHYREXIAN = {c + "P" for c in COLORS}

# Two-or-colored hybrid (e.g., W2, U2, R2)
_TWO_HYBRID = {c + "2" for c in COLORS}


class _ShardTally:
    """The running totals of one mana cost while its shards are read in order."""

    def __init__(self) -> None:
        self.total_cmc = 0.0
        self.generic = 0
        self.colorless = 0
        self.color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
        self.has_x = False
        self.has_hybrid = False
        self.has_phyrexian = False

    def add(self, shard: str) -> None:
        upper = shard.upper()
        if upper == "X":
            self.has_x = True  # X contributes 0 to CMC
        elif upper in COLORS:
            self.color_counts[upper] += 1
            self.total_cmc += 1
        elif upper in ("C", "S"):
            # C is colorless mana, distinct from generic; S is snow, no color
            self.colorless += int(upper == "C")
            self.total_cmc += 1
        elif not self._add_mixed(upper):
            self._add_generic(shard)

    def _add_mixed(self, upper: str) -> bool:
        """Count a hybrid or Phyrexian shard; False when ``upper`` is neither."""
        if upper in _HYBRID_PAIRS:
            self.has_hybrid = True
            self.total_cmc += 1
            # Each color is tracked as present (1) for the multi-hot encoding
            for c in upper:
                self._mark_present(c)
        elif upper in _PHYREXIAN:
            self.has_phyrexian = True
            self.total_cmc += 1
            self._mark_present(upper[0])
        elif upper in _TWO_HYBRID:
            self.has_hybrid = True
            self.total_cmc += 2
            self._mark_present(upper[0])
        else:
            return False
        return True

    def _mark_present(self, color: str) -> None:
        if color in COLORS:
            self.color_counts[color] = max(self.color_counts[color], 1)

    def _add_generic(self, shard: str) -> None:
        try:
            value = int(shard)
        except ValueError:
            return  # Unknown shard: skipped silently
        self.generic += value
        self.total_cmc += value


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

        tally = _ShardTally()
        for shard in stripped.split():
            tally.add(shard)
        color_counts = tally.color_counts
        colors_present = sum(1 for v in color_counts.values() if v > 0)

        return ManaCost(
            total_mana_value=tally.total_cmc,
            generic_mana=tally.generic,
            colorless_mana=tally.colorless,
            w=color_counts["W"],
            u=color_counts["U"],
            b=color_counts["B"],
            r=color_counts["R"],
            g=color_counts["G"],
            color_count=colors_present,
            has_x=tally.has_x,
            has_hybrid=tally.has_hybrid,
            has_phyrexian=tally.has_phyrexian,
        )
