"""The two files an ability tokenizer is built from, as one value.

They travel together: a checkpoint records both paths and both hashes, every
inference command re-checks both, and the vocabulary's header records the
encoding surface. Hashing, loading and reading that header are IO, and live
with the infrastructure that reads them
(:func:`effects.infrastructure.effect_model_store.tokenizer_hashes`,
:func:`effects.infrastructure.tokenizer_loading.load_tokenizer`,
:func:`effects.infrastructure.vocabulary_file.vocabulary_surface`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TokenizerFiles:
    """``--vocab-path`` and ``--keyword-definitions``."""

    vocab: Path
    keywords: Path

    @classmethod
    def of(cls, vocab: Path | str, keywords: Path | str) -> TokenizerFiles:
        return cls(Path(vocab), Path(keywords))
