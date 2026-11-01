"""``trigger_mode`` on real gen-2 trigger records (T026)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import TriggerPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.trigger_mode import FAMILY, acting_root_params


def _mode_of(record, sidecars):
    return (acting_root_params(record, sidecars) or {}).get("Mode")


def test_a_fired_trigger_of_the_named_mode_fires(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("trigger")
                  if r.payload.fired and _mode_of(r, gen2_sidecars) == "ChangesZone")
    found = FAMILY.match(record, gen2_sidecars, {"mode": "ChangesZone"})
    assert found.polarity is Polarity.FIRES and found.observed


def test_the_same_trigger_edited_to_unfired_is_the_near_miss(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("trigger")
                  if r.payload.fired and _mode_of(r, gen2_sidecars) == "ChangesZone")
    unfired = replace(record, payload=TriggerPayload(event=record.payload.event, fired=False))
    found = FAMILY.match(unfired, gen2_sidecars, {"mode": "ChangesZone"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_real_unfired_trigger_is_the_near_miss(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("trigger")
                  if not r.payload.fired and _mode_of(r, gen2_sidecars) == "Phase")
    assert FAMILY.match(record, gen2_sidecars, {"mode": "Phase"}).polarity is Polarity.DOES_NOT_FIRE


def test_another_mode_or_zone_shows_nothing(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("trigger") if _mode_of(r, gen2_sidecars) == "ChangesZone")
    assert FAMILY.match(record, gen2_sidecars, {"mode": "Attacks"}) is None
    assert FAMILY.match(record, gen2_sidecars,
                        {"mode": "ChangesZone", "destination": "Nowhere"}) is None


def test_without_the_sidecars_no_mode_can_be_read(by_kind):
    assert FAMILY.match(by_kind("trigger")[0], None, {"mode": "Phase"}) is None


def test_a_declaration_without_a_mode_is_refused(by_kind, gen2_sidecars):
    with pytest.raises(ValueError, match="mode"):
        FAMILY.match(by_kind("trigger")[0], gen2_sidecars, {})


def test_a_selector_naming_a_card_is_refused(by_kind, gen2_sidecars):
    with pytest.raises(ValueError, match="naming a card"):
        FAMILY.match(by_kind("trigger")[0], gen2_sidecars,
                     {"mode": "ChangesZone", "selector": "Card.namedGoblin King"})
