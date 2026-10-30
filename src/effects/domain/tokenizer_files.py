"""The two files an ability tokenizer is built from, as one value.

They travel together: a checkpoint records both paths and both hashes, every
inference command re-checks both, and the vocabulary's name decides the
encoding surface. Hashing and loading them is IO, and lives with the
infrastructure that reads them
(:func:`effects.infrastructure.effect_model_store.tokenizer_hashes`,
:func:`effects.infrastructure.tokenizer_loading.load_tokenizer`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effects.domain.ability_tokenizer import Surface


@dataclass(frozen=True, slots=True)
class TokenizerFiles:
    """``--vocab-path`` and ``--keyword-definitions``."""

    vocab: Path
    keywords: Path

    @classmethod
    def of(cls, vocab: Path | str, keywords: Path | str) -> TokenizerFiles:
        return cls(Path(vocab), Path(keywords))

    @property
    def surface(self) -> Surface:
        """The encoding surface the vocabulary implies (:func:`surface_of`)."""
        # Deferred: the encoder module loads torch, and this value is read by
        # torch-free commands too.
        from effects.domain.ability_encoder import surface_of

        return surface_of(self.vocab)
