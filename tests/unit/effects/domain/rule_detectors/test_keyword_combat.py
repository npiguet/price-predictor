"""``keyword_combat`` on real gen-2 combat records and probes (T027, story 1 acceptance 2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.keyword_combat import FAMILY


def test_a_deathtouch_probe_is_the_near_miss_naming_its_carrier(by_kind, gen2_sidecars):
    probe = next(r for r in by_kind("combat") if r.payload.probed_keyword == "deathtouch")
    found = FAMILY.match(probe, gen2_sidecars, {"keyword": "deathtouch"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    assert found.entities == (probe.payload.probed_entity,)


def test_a_probe_of_another_keyword_shows_nothing(by_kind, gen2_sidecars):
    probe = next(r for r in by_kind("combat") if r.payload.probed_keyword == "trample")
    assert FAMILY.match(probe, gen2_sidecars, {"keyword": "deathtouch"}) is None


def test_a_real_combat_with_a_qualifying_carrier_fires(by_kind, gen2_sidecars):
    """The probe's mirror is a real combat in which the carrier kept the keyword."""
    probe = next(r for r in by_kind("combat") if r.payload.probed_keyword == "trample")
    real = next((r for r in by_kind("combat") if r.record_id == probe.mirror_of), None)
    if real is None:
        pytest.skip("the probe's real combat is not in the fixture")
    found = FAMILY.match(real, gen2_sidecars, {"keyword": "trample"})
    assert found.polarity is Polarity.FIRES and probe.payload.probed_entity in found.entities


def test_the_carrier_edited_out_of_combat_shows_nothing(by_kind, gen2_sidecars):
    probe = next(r for r in by_kind("combat") if r.payload.probed_keyword == "trample")
    real = next((r for r in by_kind("combat") if r.record_id == probe.mirror_of), None)
    if real is None:
        pytest.skip("the probe's real combat is not in the fixture")
    carrier = probe.payload.probed_entity
    state = replace(real.state, entities=tuple(
        replace(e, combat=None) if e.id == carrier else e for e in real.state.entities))
    found = FAMILY.match(replace(real, state=state), gen2_sidecars, {"keyword": "trample"})
    assert found is None or carrier not in found.entities


def test_a_keyword_outside_gate_two_s_table_is_refused(by_kind):
    with pytest.raises(ValueError, match="not a damage-step keyword"):
        FAMILY.match(by_kind("combat")[0], None, {"keyword": "flying"})
