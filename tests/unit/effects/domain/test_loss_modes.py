"""The run-parameter modes whose ``delta`` reads as a loss are Forge's own
(``event_schema.LOSS_MODES``): a name Forge never writes would sign nothing,
and nothing else would say so."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.event_schema import LOSS_MODES, Event, EventType, signed_delta

#: The sibling Forge checkout the collector compiles against.
_FORGE_GAME = (Path(__file__).resolve().parents[4].parent / "forge" / "forge-game" / "src"
               / "main" / "java" / "forge" / "game")


def _enum_constants(path: Path, enum: str) -> set[str]:
    """The constants an enum declares, read off its source."""
    if not path.exists():
        pytest.skip(f"no Forge checkout at {path}")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    body = source.split(f"enum {enum}", 1)[1].split("{", 1)[1].split(";", 1)[0]
    body = re.sub(r"\([^()]*\)", "", body)
    return {name.strip() for name in body.split(",") if name.strip()}


def test_every_loss_mode_is_a_forge_trigger_or_replacement_mode():
    """The mode a run-parameter event carries is a ``TriggerType`` or a
    ``ReplacementType`` name (``RunParamNormalizer.modeOf``)."""
    modes = (_enum_constants(_FORGE_GAME / "trigger" / "TriggerType.java", "TriggerType")
             | _enum_constants(_FORGE_GAME / "replacement" / "ReplacementType.java",
                               "ReplacementType"))
    assert {mode for names in LOSS_MODES.values() for mode in names} <= modes


@pytest.mark.parametrize(("mode", "delta"), [
    ("LifeLost", -3.0), ("PayLife", -3.0), ("LifeReduced", -3.0),
    ("LifeGained", 3.0), ("GainLife", 3.0), (None, 3.0),
])
def test_a_life_change_reads_its_sign_from_its_mode(mode, delta):
    params = {"delta": 3} if mode is None else {"delta": 3, "mode": mode}
    assert signed_delta(Event(type=EventType.LIFE_CHANGE, params=params)) == delta
