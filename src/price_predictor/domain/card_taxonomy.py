"""Single source of truth for MTG card taxonomy registries.

These tuples and frozensets were previously duplicated (and slightly diverged)
across feature_engineering.py, entities.py, value_objects.py, and
converted_card_parser.py. Importing from this module keeps them in lockstep.

The ordered tuples (CARD_TYPES, SUPERTYPES, LAYOUTS, RARITIES) are the
contract for one-hot encoding feature positions — do not reorder existing
entries without updating the corresponding tests and any persisted models.
"""

from __future__ import annotations

CARD_TYPES: tuple[str, ...] = (
    "Creature", "Instant", "Sorcery", "Enchantment",
    "Artifact", "Planeswalker", "Land", "Battle",
    "Scheme", "Plane", "Conspiracy", "Vanguard", "Phenomenon",
    "Tribal", "Kindred",
)

SUPERTYPES: tuple[str, ...] = (
    "Legendary", "Basic", "Snow", "World", "Ongoing", "Host",
)

LAYOUTS: tuple[str, ...] = (
    "normal", "doublefaced", "split", "adventure", "modal", "flip",
)

# The four rarities used for one-hot encoding. Cards reported as "special" or
# "bonus" are mapped to "rare" by the feature engineer's effective_rarity
# fallback — see VALID_RARITIES for the broader validation set.
RARITIES: tuple[str, ...] = (
    "common", "uncommon", "rare", "mythic",
)

KNOWN_CARD_TYPES: frozenset[str] = frozenset(CARD_TYPES)
KNOWN_SUPERTYPES: frozenset[str] = frozenset(SUPERTYPES)
VALID_LAYOUTS: frozenset[str] = frozenset(LAYOUTS)

# Validation set for PrintingData.rarity. Includes the two special-purpose
# rarities MTGJSON occasionally emits ("special", "bonus") on top of the
# four canonical rarities used for encoding.
VALID_RARITIES: frozenset[str] = frozenset(RARITIES) | frozenset({"special", "bonus"})
