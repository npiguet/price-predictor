"""The ability tokenizer a command reads with, built from the files it names.

Torch-free at import, so ``build-vocab`` can use it without the GPU stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from effects.domain.ability_tokenizer import AbilityTokenizer, tokenizer_surface
from price_predictor.infrastructure.tokenizer_store import load_vocabulary


def load_definitions(keyword_path: Path) -> dict:
    """The keyword definitions at ``keyword_path``; none where it is absent,
    since a checkpoint trained before the table existed recorded none."""
    if not Path(keyword_path).exists():
        return {}
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )

    return load_keyword_definitions(Path(keyword_path))


def load_tokenizer(
    vocab_path: Path, keyword_path: Path, *,
    training_settings: Mapping | None = None,
) -> AbilityTokenizer:
    """The tokenizer over ``vocab_path`` and the definitions at ``keyword_path``.

    With a loaded checkpoint's ``training_settings`` it applies the rules that
    checkpoint trained under (:func:`tokenizer_surface`); without, the
    vocabulary's own surface's.
    """
    from effects.domain.ability_encoder import surface_of

    surface = surface_of(vocab_path)
    if training_settings is not None:
        surface = tokenizer_surface(surface, training_settings)
    return AbilityTokenizer(
        load_vocabulary(Path(vocab_path)), load_definitions(keyword_path),
        surface=surface,
    )
