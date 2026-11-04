"""Deck: an ordered multiset of card names built from a sealed pool."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Basic land identity is domain knowledge (not a file-lookup detail), so it
# lives here. ``converted_card_locator`` re-exports it for its callers.
# Every card with the Basic supertype, the rule the connector's ``BasicLands``
# takes from Forge, so the pools and cards-played rows it writes and the decks
# read here agree: Wastes and the snow-covered basics count too.
BASIC_LAND_NAMES: frozenset[str] = frozenset(
    {
        "plains", "island", "swamp", "mountain", "forest", "wastes",
        "snow-covered plains", "snow-covered island", "snow-covered swamp",
        "snow-covered mountain", "snow-covered forest", "snow-covered wastes",
    }
)


@dataclass(frozen=True)
class Deck:
    """A built deck: card names in pick order, duplicate copies repeated.

    The set the deck was drawn from is context owned by the deck's container (a
    ``SealedPool``, a ``GeneratedDeck`` line, or a match), so it is not stored
    on the deck itself.
    """

    cards: tuple[str, ...]

    @classmethod
    def of(cls, names: Iterable[str]) -> "Deck":
        """Build a ``Deck`` from any sequence of card names."""
        return cls(tuple(names))

    def __len__(self) -> int:
        return len(self.cards)

    def card_multiset(self) -> Counter[str]:
        """Card name → copy count — the deck's identity for mirror comparison."""
        return Counter(self.cards)

    def nonbasic_cards(self) -> list[str]:
        """Cards excluding basic lands, order preserved, duplicates repeated."""
        return [c for c in self.cards if c.lower() not in BASIC_LAND_NAMES]
