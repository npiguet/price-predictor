"""EncodeCardsUseCase: encode card scripts to .npz embeddings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EncodeCardsConfig:
    cards_path: Path
    clean: bool = False


@dataclass
class EncodeCardsResult:
    processed: int
    skipped: int
    errors: list[str]


class EncodeCardsUseCase:
    """Scan a directory of .txt card scripts and produce .npz embeddings.

    Already-encoded cards (where a matching .npz exists) are skipped.
    Errors in individual cards are collected; processing continues.
    """

    def execute(
        self,
        config: EncodeCardsConfig,
        encoder,
        store,
        progress: Callable[[int, int], None] | None = None,
    ) -> EncodeCardsResult:
        if config.clean:
            for f in config.cards_path.rglob("*.npz"):
                f.unlink()

        txt_files = sorted(config.cards_path.rglob("*.txt"))
        processed = 0
        skipped = 0
        errors: list[str] = []

        for txt_path in txt_files:
            npz_path = txt_path.with_suffix(".npz")

            if npz_path.exists():
                skipped += 1
                continue

            try:
                text = txt_path.read_text(encoding="utf-8")
                embedding = encoder.encode(text)
                store.save(npz_path, embedding)
                processed += 1
            except Exception as exc:
                errors.append(f"{txt_path}: {exc}")
                continue

            if progress and processed % 100 == 0 and processed > 0:
                progress(processed, skipped)

        return EncodeCardsResult(processed=processed, skipped=skipped, errors=errors)
