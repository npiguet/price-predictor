"""The run-parameter modes whose ``delta`` reads as a loss are Forge's own
(``event_schema.LOSS_MODES``): a name Forge never writes would sign nothing,
and nothing else would say so."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.event_schema import LOSS_MODES, Event, EventType, signed_delta
from tests.unit.forge_sources import FORGE_GAME, forge_source

#: The sibling Forge checkout the collector compiles against.
_FORGE_GAME = FORGE_GAME


def _enum_constants(path: Path, enum: str) -> set[str]:
    """The constants an enum declares, read off its source."""
    forge_source(path)
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


@pytest.mark.parametrize(("mode", "delta", "read"), [
    ("CounterRemovedOnce", -3, -3.0), ("CounterRemovedOnce", 3, -3.0),
    ("CounterRemoved", -1, -1.0), ("RemoveCounter", 0, 0.0), ("CounterAddedOnce", 2, 2.0),
])
def test_a_counter_change_reads_its_sign_from_its_mode(mode, delta, read):
    """The collector writes each mode's delta signed; the loss modes read
    negative whatever sign the amount carries."""
    event = Event(type=EventType.COUNTER_CHANGE,
                  params={"counter_type": "P1P1", "delta": delta, "mode": mode})
    assert signed_delta(event) == read
