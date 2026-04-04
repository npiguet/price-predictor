"""ValidateEmbeddingsUseCase: probe card embeddings for feature decodability."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from sealed.domain.embedding_probe import (
    CardData,
    ProbeResult,
    ValidationResult,
    _is_land_text,
    build_default_probes,
    run_probes,
)


class ValidateEmbeddingsUseCase:
    """Load card embeddings + texts and run linear probes to validate quality."""

    def execute(
        self,
        cards_path: Path,
        threshold_accuracy: float = 0.95,
        threshold_r2: float = 0.85,
        on_result: Callable[[ProbeResult], None] | None = None,
        embed_dim: int | None = None,
    ) -> ValidationResult:
        """Discover cards, run probes, return ValidationResult.

        Args:
            embed_dim: If provided, probes use only the first *embed_dim* dimensions
                of each embedding — i.e., the transformer-learned portion, excluding
                any appended explicit features.

        Raises:
            ValueError: if fewer than 50 paired cards are found.
        """
        cards = _load_cards(cards_path)

        if len(cards) < 50:
            raise ValueError(
                f"Insufficient data: only {len(cards)} cards with both .npz and .txt "
                f"found in {cards_path}. At least 50 are required."
            )

        n_lands = sum(1 for c in cards if _is_land_text(c.text))
        probes = build_default_probes(threshold_accuracy, threshold_r2)
        results = run_probes(cards, probes, on_result=on_result, embed_dim=embed_dim)
        all_passed = all(r.passed for r in results)

        return ValidationResult(
            probe_results=results,
            n_cards=len(cards),
            n_lands=n_lands,
            all_passed=all_passed,
        )


def _load_cards(cards_path: Path) -> list[CardData]:
    """Discover paired .npz/.txt files and load them as CardData objects."""
    cards: list[CardData] = []
    for npz_path in sorted(cards_path.rglob("*.npz")):
        txt_path = npz_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        embedding = np.load(npz_path)["embedding"]
        text = txt_path.read_text(encoding="utf-8")
        cards.append(CardData(name=npz_path.stem, embedding=embedding, text=text))
    return cards
