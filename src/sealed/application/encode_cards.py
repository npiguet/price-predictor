"""EncodeCardsUseCase: encode card scripts to .npz embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    def execute(self, cards_path: Path, encoder, store) -> EncodeCardsResult:
        txt_files = sorted(cards_path.rglob("*.txt"))
        processed = 0
        skipped = 0
        errors: list[str] = []

        for i, txt_path in enumerate(txt_files):
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

            if processed % 100 == 0 and processed > 0:
                print(f"\rProgress: {processed} encoded ({skipped} skipped)", end="", flush=True)

        if processed > 0 or skipped > 0:
            print()  # newline after \r progress

        return EncodeCardsResult(processed=processed, skipped=skipped, errors=errors)
