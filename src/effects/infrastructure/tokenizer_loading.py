"""The ability tokenizer a command reads with, built from the files it names.

Torch-free at import, so ``build-vocab`` can use it without the GPU stack.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from effects.domain.ability_tokenizer import AbilityTokenizer, tokenizer_surface
from effects.domain.keyword_definitions import KeywordDefinition
from effects.domain.tokenizer_files import TokenizerFiles
from price_predictor.infrastructure.tokenizer_store import load_vocabulary


def _normalized(template: str | None) -> str | None:
    """Forge's typographic apostrophe as the ASCII one the vocabulary spells
    (FR-017); normalized at load, never in the file."""
    return template.replace("’", "'") if template else template


def load_keyword_definitions(path: Path) -> dict[str, KeywordDefinition]:
    """Read the definition file into ``display name -> KeywordDefinition``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        keyword: KeywordDefinition(
            keyword=keyword,
            reminder_template=_normalized(entry.get("reminder_template")),
            generated_script=entry.get("generated_script"),
            formatter=entry.get("formatter"),
        )
        for keyword, entry in data.items()
    }


def load_definitions(keyword_path: Path) -> dict[str, KeywordDefinition]:
    """The keyword definitions at ``keyword_path``; none where it is absent,
    since a checkpoint trained before the table existed recorded none."""
    if not Path(keyword_path).exists():
        return {}
    return load_keyword_definitions(Path(keyword_path))


def load_tokenizer(
    files: TokenizerFiles, *, training_settings: Mapping | None = None,
) -> AbilityTokenizer:
    """The tokenizer over ``files``' vocabulary and keyword definitions.

    With a loaded checkpoint's ``training_settings`` it applies the rules that
    checkpoint trained under (:func:`tokenizer_surface`); without, the
    vocabulary's own surface's.
    """
    surface = files.surface
    if training_settings is not None:
        surface = tokenizer_surface(surface, training_settings)
    return AbilityTokenizer(
        load_vocabulary(files.vocab), load_definitions(files.keywords),
        surface=surface,
    )


def tokenizer_for(checkpoint, files: TokenizerFiles) -> AbilityTokenizer:
    """The tokenizer ``checkpoint`` trained with, under the rules it recorded
    (a gen-1 checkpoint reads its script vocabulary with the prose grammar)."""
    return load_tokenizer(files, training_settings=checkpoint.training_settings or {})
