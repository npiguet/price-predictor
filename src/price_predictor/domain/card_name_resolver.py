"""Resolves a card name (any case, possibly a split-card front) to its canonical name."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from price_predictor.domain.value_objects import PrintingData


@dataclass(frozen=True)
class ResolvedCard:
    """A card name that has been matched to a known canonical name, price, and printing data."""

    canonical_name: str
    price_eur: float
    printing_data: PrintingData


class CardNameResolver:
    """Canonicalize a card name and look up its price + printing data.

    Handles three concerns that every caller used to reimplement:
      * Case-insensitive matching (converted card names are lowercase).
      * Split cards: a query for "Fire" matches the canonical "Fire // Ice".
      * Defaulting `PrintingData` when the printing-data map lacks an entry.

    Either `price_map` or `printing_data_map` must be provided; their keys define
    the universe of known canonical names. Train/eval call sites pass both
    so they can call `resolve(...)`. Predict call sites typically pass only
    `printing_data_map` and call `lookup_printing_data(...)`.
    """

    def __init__(
        self,
        price_map: dict[str, float] | None = None,
        printing_data_map: dict[str, PrintingData] | None = None,
    ) -> None:
        self._price_map = price_map or {}
        self._printing_data_map = printing_data_map or {}
        # Universe of canonical names: union of whatever we were given.
        names: Iterable[str] = self._price_map.keys() | self._printing_data_map.keys()
        self._lower_to_canonical: dict[str, str] = {k.lower(): k for k in names}

    def resolve(self, name: str) -> ResolvedCard | None:
        """Resolve `name` to a ResolvedCard with canonical name, price, and printing data.

        Returns None if the name is unknown or no price is recorded for it.
        """
        canonical = self.canonicalize(name)
        if canonical is None or canonical not in self._price_map:
            return None
        return ResolvedCard(
            canonical_name=canonical,
            price_eur=self._price_map[canonical],
            printing_data=self._printing_data_map.get(canonical) or PrintingData.defaults(),
        )

    def lookup_printing_data(self, name: str) -> PrintingData:
        """Return the PrintingData for `name`, or defaults if unknown.

        Used by prediction paths that need printing data but no card price.
        """
        canonical = self.canonicalize(name)
        if canonical is None:
            return PrintingData.defaults()
        return self._printing_data_map.get(canonical) or PrintingData.defaults()

    def canonicalize(self, name: str) -> str | None:
        """Canonicalize `name` (case-insensitive, split-card aware). None if unknown."""
        lowered = name.lower()
        canonical = self._lower_to_canonical.get(lowered)
        if canonical is not None:
            return canonical
        prefix = lowered + " // "
        for full_lower, full_canonical in self._lower_to_canonical.items():
            if full_lower.startswith(prefix):
                return full_canonical
        return None
