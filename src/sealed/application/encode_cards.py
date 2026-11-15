"""EncodeCardsUseCase: encode card scripts to .npz embeddings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from price_predictor.domain.card_text import ConvertedCardText


@dataclass
class EncodeCardsConfig:
    cards_path: Path
    clean: bool = False
    #: Where the ``.npz`` files go, laid out like ``cards_path``; None puts
    #: each beside its card.
    embeddings_path: Path | None = None

    @property
    def embeddings_root(self) -> Path:
        return self.cards_path if self.embeddings_path is None else self.embeddings_path


@dataclass
class EncodeCardsResult:
    processed: int
    skipped: int
    errors: list[str]


class EncodeCardsUseCase:
    """Scan a directory of .txt card scripts and produce .npz embeddings.

    Each card's ``.npz`` sits at the card's place in the embeddings tree
    (beside the card unless ``embeddings_path`` names another tree).
    Already-encoded cards (where a matching .npz exists) are skipped, and
    ``clean`` deletes every ``.npz`` of the embeddings tree first.
    Errors in individual cards are collected; processing continues.
    """

    def execute(
        self,
        config: EncodeCardsConfig,
        encoder,
        store,
        progress: Callable[[int, int], None] | None = None,
    ) -> EncodeCardsResult:
        root = config.embeddings_root
        if config.clean:
            for f in root.rglob("*.npz"):
                f.unlink()

        txt_files = sorted(config.cards_path.rglob("*.txt"))
        processed = 0
        skipped = 0
        errors: list[str] = []

        for txt_path in txt_files:
            npz_path = (root / txt_path.relative_to(config.cards_path)).with_suffix(".npz")

            if npz_path.exists():
                skipped += 1
                continue

            try:
                converted = ConvertedCardText.from_file(txt_path)
                embedding = encoder.encode(converted)
                store.save(npz_path, embedding)
                processed += 1
            except Exception as exc:
                errors.append(f"{txt_path}: {exc}")
                continue

            if progress and processed % 100 == 0 and processed > 0:
                progress(processed, skipped)

        return EncodeCardsResult(processed=processed, skipped=skipped, errors=errors)
