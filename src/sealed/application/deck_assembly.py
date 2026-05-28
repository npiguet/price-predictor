"""Shared deck-assembly helpers for the scorer/picker build pipelines.

``build-decks``, ``pick-decks``, and ``evaluate-scorer`` all turn a sealed pool
into one 40-card deck the same way: load the pool's card embeddings, run a
nonland selector (greedy search or picker walk), then complete the chosen
nonlands to 40 cards with the heuristic basic-land manabase. Only the selector
differs; the load and the completion are identical and live here so the three
use cases cannot drift.
"""

from __future__ import annotations

import numpy as np

from sealed.domain.manabase import compute_basic_lands
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator


def load_pool_embeddings(
    pool_names: list[str], locator: ConvertedCardLocator,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load each pool card's ``.npz`` embedding, dropping names with none.

    Returns ``(embeddings_by_name, valid_names)`` where ``valid_names`` is the
    subset of ``pool_names`` (in original order) that had an embedding on disk
    and ``embeddings_by_name`` maps each such name to its vector.
    """
    embeddings: dict[str, np.ndarray] = {}
    valid_names: list[str] = []
    for name in pool_names:
        emb = locator.load_embedding(name)
        if emb is not None:
            embeddings[name] = emb
            valid_names.append(name)
    return embeddings, valid_names


def assemble_full_deck(
    nonland_deck: list[str], locator: ConvertedCardLocator,
) -> list[str]:
    """Complete a chosen set of nonlands to a 40-card deck.

    Adds the basic-land manabase (``compute_basic_lands`` over the nonlands'
    mana costs) to the chosen nonlands. Returns the nonlands followed by the
    basics, with duplicate cards repeated.
    """
    nonland_texts = [
        text
        for name in nonland_deck
        if (text := locator.load_text(name)) is not None
    ]
    lands = compute_basic_lands(nonland_texts)
    full_deck: list[str] = list(nonland_deck)
    for land_name, count in lands.items():
        full_deck.extend([land_name] * count)
    return full_deck
