"""The shape a training run's recorded settings take, which every checkpoint
and ``resume.pt`` records and every reader requires."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Where the settings record their own shape.
SETTINGS_SHAPE_KEY = "settings_shape"

#: One dict: every config flag its field's metadata records, the corpus,
#: selection set and shard listing by digest, the tokenizer files by hash,
#: the script-API vocabularies, the tokenizer rules and the input layout.
#: Settings recorded before (flags split between the settings, the payload and
#: the resume file alone) record no shape.
SETTINGS_SHAPE = "one-dict"


class SettingsShapeError(ValueError):
    """A checkpoint or resume file records its settings in another shape."""


def require_settings_shape(settings: Mapping[str, Any] | None, *, holder: str) -> None:
    """Refuse settings ``holder`` (``"checkpoint"``, ``"resume file"``)
    records in another shape than :data:`SETTINGS_SHAPE`: the flags it trained
    under would be read where it never wrote them."""
    recorded = (settings or {}).get(SETTINGS_SHAPE_KEY)
    if recorded != SETTINGS_SHAPE:
        raise SettingsShapeError(
            f"the {holder} records its settings in the old shape ({SETTINGS_SHAPE_KEY} "
            f"{recorded!r}, this code reads {SETTINGS_SHAPE!r}): it was written before "
            "a run's settings were recorded once, in one dict, so the flags it trained "
            "under are not where this code reads them. Train it again."
        )
