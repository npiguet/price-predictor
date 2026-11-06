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
    # A trigger record carries no outcome: the triggering event's subjects
    # and the trigger's source are its condition.
    assert found.outcome == ()
    assert set(record.payload.event.subjects) <= set(found.condition)
    if record.state.refs.source and record.state.entity(record.state.refs.source):
        assert record.state.refs.source in found.condition


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


def _keyword_trigger(by_kind, sidecars, fired):
    return next(r for r in by_kind("trigger") if bool(r.payload.fired) == fired
                and r.ability and r.ability[0].trait_kind == "keyword"
                and sidecars.first_line(r.ability).keyword_display_name)


class TestKeyword:
    def test_a_trigger_a_keyword_generates_shows_on_its_side(self, by_kind, gen2_sidecars):
        for fired, polarity in ((True, Polarity.FIRES), (False, Polarity.DOES_NOT_FIRE)):
            record = _keyword_trigger(by_kind, gen2_sidecars, fired)
            name = gen2_sidecars.first_line(record.ability).keyword_display_name
            found = FAMILY.match(record, gen2_sidecars, {"keyword": name})
            assert found.polarity is polarity and found.observed is fired

    def test_the_name_is_compared_in_the_packages_spelling(self, by_kind, gen2_sidecars):
        record = _keyword_trigger(by_kind, gen2_sidecars, True)
        name = gen2_sidecars.first_line(record.ability).keyword_display_name
        spelled = name.lower().replace(" ", "_")
        assert FAMILY.match(record, gen2_sidecars, {"keyword": spelled}) is not None

    def test_another_keyword_or_a_script_trigger_shows_nothing(self, by_kind, gen2_sidecars):
        record = _keyword_trigger(by_kind, gen2_sidecars, True)
        assert FAMILY.match(record, gen2_sidecars, {"keyword": "no such keyword"}) is None
        scripted = next(r for r in by_kind("trigger") if _mode_of(r, gen2_sidecars))
        assert FAMILY.match(scripted, gen2_sidecars, {"keyword": "exalted"}) is None

    def test_a_keyword_takes_no_script_parameter(self, by_kind, gen2_sidecars):
        with pytest.raises(ValueError, match="no script parameters"):
            FAMILY.match(by_kind("trigger")[0], gen2_sidecars,
                         {"keyword": "exalted", "mode": "Attacks"})


def _phase_trigger(by_kind, sidecars, fired=True):
    return next(r for r in by_kind("trigger") if bool(r.payload.fired) == fired
                and (acting_root_params(r, sidecars) or {}).get("Phase"))


class TestT030Parameters:
    """Phase, script parameters, step, source and spawned triggers (T030 follow-up)."""

    def test_the_phase_and_a_script_parameter_read_the_trigger_segment(
            self, by_kind, gen2_sidecars):
        record = _phase_trigger(by_kind, gen2_sidecars)
        phase = acting_root_params(record, gen2_sidecars)["Phase"]
        assert FAMILY.match(record, gen2_sidecars, {"mode": "Phase", "phase": phase})
        assert FAMILY.match(record, gen2_sidecars, {"mode": "Phase", "phase": "Nowhere"}) is None
        assert FAMILY.match(record, gen2_sidecars, {"mode": "Phase", "script_param": "Phase"})
        assert FAMILY.match(record, gen2_sidecars,
                            {"mode": "Phase", "script_param": "NoSuchKey"}) is None

    def test_a_step_alone_speaks_for_every_trigger_evaluated_then(self, by_kind, gen2_sidecars):
        record = by_kind("trigger")[0]
        step = record.state.global_.phase.lower()
        found = FAMILY.match(record, gen2_sidecars, {"step": step})
        assert found.polarity is (Polarity.FIRES if record.payload.fired
                                  else Polarity.DOES_NOT_FIRE)
        other = next(p for p in ("cleanup", "untap") if p != step)
        assert FAMILY.match(record, gen2_sidecars, {"step": other}) is None
        with pytest.raises(ValueError, match="step"):
            FAMILY.match(record, gen2_sidecars, {"step": "EndCombat"})

    @staticmethod
    def _sourced(by_kind, fired=True):
        return next(r for r in by_kind("trigger") if bool(r.payload.fired) == fired
                    and r.state.refs.source and r.state.entity(r.state.refs.source))

    @staticmethod
    def _source_as(record, **changes):
        source = record.state.refs.source
        return replace(record, state=replace(record.state, entities=tuple(
            replace(e, **changes) if e.id == source else e for e in record.state.entities)))

    def test_a_copy_source_is_read_off_the_trigger_source(self, by_kind):
        record = self._sourced(by_kind)
        declared = {"step": record.state.global_.phase.lower(), "source_is_copy": True}
        copied = self._source_as(record, copy_source="E1")
        assert FAMILY.match(copied, None, declared).polarity is Polarity.FIRES
        assert FAMILY.match(self._source_as(record, copy_source=None), None, declared) is None

    def test_a_designation_counts_on_the_side_it_explains(self, by_kind):
        for fired, polarity in ((True, Polarity.FIRES), (False, Polarity.DOES_NOT_FIRE)):
            record = self._sourced(by_kind, fired)
            declared = {"step": record.state.global_.phase.lower(),
                        "source_designation": "harnessed"}
            held = self._source_as(record, lines=(), designations=("harnessed",))
            lacking = self._source_as(record, lines=(), designations=())
            explained, other = (held, lacking) if fired else (lacking, held)
            assert FAMILY.match(explained, None, declared).polarity is polarity
            assert FAMILY.match(other, None, declared) is None
            # A gen-2 source records no designation.
            assert FAMILY.match(record, None, declared) is None
        assert FAMILY.needs({"mode": "Phase", "source_designation": "x"}) == {"designations"}

    def test_a_delayed_trigger_is_read_off_the_segment_that_created_it(self, by_kind):
        from effects.domain.provenance import ProvenanceKey, SidecarLine

        spell = ProvenanceKey("cardsfolder/x/x.txt", 0, "spell", 0)
        line = SidecarLine(
            line_index=0, line_kind="spell", provenance=(spell,), script_api_type="Dig",
            script_text="SP$ Dig | DigNum$ 3 | SubAbility$ SV1 [SEG] SV1: DB$ DelayedTrigger "
                        "| Mode$ Phase | Phase$ End of Turn | Execute$ SV2")

        class Lines:
            def first_line(self, keys):
                return line
        record = replace(by_kind("trigger")[0], ability=(spell,))
        declared = {"spawned_by": "DelayedTrigger", "mode": "Phase"}
        assert FAMILY.match(record, Lines(), declared) is not None
        assert FAMILY.match(record, Lines(), {**declared, "phase": "End of Turn"}) is not None
        assert FAMILY.match(record, Lines(), {"spawned_by": "ImmediateTrigger"}) is None
        # A trigger line's own chain is the printed trigger, not one it created.
        printed = replace(record, ability=(replace(spell, trait_kind="trigger"),))
        assert FAMILY.match(printed, Lines(), declared) is None
        with pytest.raises(ValueError, match="spawned_by"):
            FAMILY.match(record, Lines(), {"spawned_by": "Effect"})
