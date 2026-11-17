"""``rewrite_event`` on real gen-3 rewrite records (T029)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import RewriteResult
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import acting_line
from effects.domain.rule_detectors.rewrite_event import (
    FAMILY,
    caused_with_empty_stack,
    is_prevention_line,
)


def _entering(by_kind):
    """A real zone change a replacement effect replaced with another event."""
    return next(r for r in by_kind("rewrite")
                if r.payload.incoming.type.value == "zone_change"
                and r.payload.result is RewriteResult.REPLACED)


def test_a_replaced_zone_change_fires_a_zone_replacement_rule(by_kind):
    record = _entering(by_kind)
    params = {"event": "zone_change", "result": "replaced",
              "to_zone": record.payload.incoming.params["to_zone"]}
    found = FAMILY.match(record, None, params)
    assert found.polarity is Polarity.FIRES and found.observed
    # The entering object is both what the replacement reads and where its
    # result falls, as proposed and as it went out.
    incoming = record.payload.incoming.subjects
    assert set(incoming) <= set(found.condition)
    outgoing = record.payload.outgoing.subjects if record.payload.outgoing else ()
    assert found.outcome == tuple(dict.fromkeys((*incoming, *outgoing)))


def test_the_same_record_edited_to_not_replaced_is_the_near_miss(by_kind):
    record = _entering(by_kind)
    edited = replace(record, payload=replace(record.payload, result=RewriteResult.NOT_REPLACED,
                                             outgoing=None, replaced_by=()))
    found = FAMILY.match(edited, None, {"event": "zone_change", "result": "replaced"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_the_mode_and_any_incoming_parameter_narrow_the_event(by_kind):
    record = _entering(by_kind)
    incoming = record.payload.incoming
    mode = incoming.params["mode"]
    for declared in ({"mode": mode}, {"mode": mode.lower()},
                     {"to_zone": incoming.params["to_zone"].upper()}):
        found = FAMILY.match(record, None, {"event": "zone_change", "result": "replaced",
                                            **declared})
        assert found.polarity is Polarity.FIRES, declared
    assert FAMILY.match(record, None, {"event": "zone_change", "mode": "BeginPhase"}) is None
    with pytest.raises(ValueError, match="no parameter"):
        FAMILY.validate({"event": "zone_change", "counter_type": "P1P1"})


def test_a_result_list_names_every_fires_result(by_kind):
    record = _entering(by_kind)
    found = FAMILY.match(record, None,
                         {"event": "zone_change", "result": ["replaced", "updated"]})
    assert found.polarity is Polarity.FIRES
    found = FAMILY.match(record, None,
                         {"event": "zone_change", "result": ["updated", "skipped"]})
    assert found.polarity is Polarity.DOES_NOT_FIRE


def _keyword_rewrite(by_kind, sidecars):
    return next(r for r in by_kind("rewrite") if acting_line(r, sidecars) is not None
                and acting_line(r, sidecars).keyword_display_name)


def test_a_keyword_replacement_shows_by_its_keyword(by_kind, gen3_sidecars):
    record = _keyword_rewrite(by_kind, gen3_sidecars)
    name = acting_line(record, gen3_sidecars).keyword_display_name
    event = record.payload.incoming.type.value
    result = record.payload.result.value
    found = FAMILY.match(record, gen3_sidecars,
                         {"event": event, "result": result, "keyword": name.lower()})
    assert found.polarity is Polarity.FIRES
    assert FAMILY.match(record, gen3_sidecars, {"event": event, "keyword": "Equip"}) is None


class TestPrevention:
    @staticmethod
    def _shield(by_kind, sidecars):
        return next(r for r in by_kind("rewrite")
                    if r.payload.incoming.type.value == "damage_dealt"
                    and r.payload.result in (RewriteResult.REPLACED, RewriteResult.UPDATED)
                    and is_prevention_line(acting_line(r, sidecars)))

    def test_a_shield_replacing_damage_fires_whatever_result_it_reports(
            self, by_kind, gen3_sidecars):
        record = self._shield(by_kind, gen3_sidecars)
        found = FAMILY.match(record, gen3_sidecars,
                             {"event": "damage_dealt", "prevention": True})
        assert found.polarity is Polarity.FIRES and found.observed

    def test_a_shield_preventing_nothing_is_the_near_miss(self, by_kind, gen3_sidecars):
        record = self._shield(by_kind, gen3_sidecars)
        edited = replace(record, payload=replace(
            record.payload, result=RewriteResult.NOT_REPLACED, outgoing=None, replaced_by=()))
        found = FAMILY.match(edited, gen3_sidecars, {"event": "damage_dealt", "prevention": True})
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        found = FAMILY.match(edited, gen3_sidecars, {"event": "damage_dealt",
                                                     "prevention": True,
                                                     "result": "not_replaced"})
        assert found.polarity is Polarity.FIRES

    def test_another_replacement_is_no_prevention_but_a_prevented_result_is(
            self, by_kind, gen3_sidecars):
        record = _keyword_rewrite(by_kind, gen3_sidecars)
        assert not is_prevention_line(acting_line(record, gen3_sidecars))
        event = record.payload.incoming.type.value
        declared = {"event": event, "prevention": True}
        assert FAMILY.match(record, gen3_sidecars, declared) is None
        prevented = replace(record, payload=replace(record.payload,
                                                    result=RewriteResult.PREVENTED))
        assert FAMILY.match(prevented, gen3_sidecars, declared).polarity is Polarity.FIRES


def test_another_event_or_destination_shows_nothing(by_kind):
    record = _entering(by_kind)
    assert FAMILY.match(record, None, {"event": "damage_dealt"}) is None
    assert FAMILY.match(record, None, {"event": "zone_change", "to_zone": "exile"}) is None


class TestSubjectQualifiers:
    """``outcome_event``'s subject qualifiers on the incoming event (T030 follow-up)."""

    @staticmethod
    def _rewrite(by_kind):
        return next((r, s) for r in by_kind("rewrite")
                    if r.payload.result is not None
                    for s in r.payload.incoming.subjects if r.state.entity(s) is not None)

    @staticmethod
    def _declared(record, **qualifier):
        return {"event": record.payload.incoming.type.value,
                "result": record.payload.result.value, **qualifier}

    def test_a_melded_subject_keeps_the_record_and_another_drops_it(self, by_kind):
        record, subject = self._rewrite(by_kind)
        declared = self._declared(record, subject_face_state="melded")

        def subject_as(face_state):
            return replace(record, state=replace(record.state, entities=tuple(
                replace(e, face_state=face_state) if e.id == subject else e
                for e in record.state.entities)))
        found = FAMILY.match(subject_as("melded"), None, declared)
        assert found.polarity is Polarity.FIRES and subject in found.outcome
        assert FAMILY.match(subject_as("front"), None, declared) is None

    def test_a_merged_subject_and_its_need(self, by_kind):
        from effects.domain.state_snapshot import Link

        record, subject = self._rewrite(by_kind)
        merged = replace(record, state=replace(record.state, links=(
            Link(from_="E999999", to=subject, type="merged_with"),)))
        declared = self._declared(record, subject_merged=True)
        assert FAMILY.match(merged, None, declared).polarity is Polarity.FIRES
        assert FAMILY.needs(declared) == {"links"}
        assert FAMILY.needs(self._declared(record)) == frozenset()


def test_a_record_whose_incoming_event_is_unknown_shows_no_side(by_kind):
    """WP7: ``incoming`` then holds the event as it left; what came in, and
    whether the replacement edited it, are unknown, so no side is read."""
    record = _entering(by_kind)
    params = {"event": "zone_change", "result": "replaced"}
    assert FAMILY.match(record, None, params) is not None
    unknown = replace(record, payload=replace(record.payload, incoming_unknown=True,
                                              outgoing=None))
    assert FAMILY.match(unknown, None, params) is None


class TestPaidAsCost:
    """118.11 (WP6): ``caused_with_empty_stack`` reads an incoming event naming a cause
    while the stack is empty. Forge pays an activated ability's costs before
    it puts the ability on the stack, so a cost paid then is read; so is any
    other event a cause made off the stack, and a cost paid while the stack
    held something is on neither side."""

    PARAMS = {"event": "zone_change", "from_zone": "battlefield", "to_zone": "graveyard",
              "caused_with_empty_stack": True}

    @staticmethod
    def _sacrificed(by_kind, *, cause="E2", stack_size=0, result=RewriteResult.REPLACED):
        record = next(r for r in by_kind("rewrite")
                      if r.payload.incoming.type.value == "zone_change")
        incoming = replace(record.payload.incoming, params={
            "mode": "Moved", "from_zone": "battlefield", "to_zone": "graveyard",
            **({"cause": cause} if cause else {})})
        state = replace(record.state, global_=replace(record.state.global_,
                                                      stack_size=stack_size))
        return replace(record, state=state, payload=replace(
            record.payload, incoming=incoming, outgoing=None, result=result))

    def test_a_cost_s_move_a_replacement_changed_fires(self, by_kind):
        found = FAMILY.match(self._sacrificed(by_kind), None, self.PARAMS)
        assert found.polarity is Polarity.FIRES and found.observed

    def test_a_cost_s_move_left_alone_is_the_near_miss(self, by_kind):
        record = self._sacrificed(by_kind, result=RewriteResult.NOT_REPLACED)
        assert FAMILY.match(record, None, self.PARAMS).polarity is Polarity.DOES_NOT_FIRE

    def test_a_move_while_the_stack_holds_something_shows_no_side(self, by_kind):
        assert FAMILY.match(self._sacrificed(by_kind, stack_size=1), None, self.PARAMS) is None

    def test_a_move_nothing_caused_shows_no_side(self, by_kind):
        """A state-based death names no cause."""
        assert FAMILY.match(self._sacrificed(by_kind, cause=None), None, self.PARAMS) is None
        declared = {key: value for key, value in self.PARAMS.items()
                    if key != "caused_with_empty_stack"}
        assert FAMILY.match(self._sacrificed(by_kind, cause=None), None, declared) is not None

    def test_any_event_a_cause_made_off_the_stack_is_read_not_a_cost_s_alone(self, by_kind):
        """A land played names its cause with the stack empty, as a cost's
        move does: the parameter reads it, and only the zones keep it out."""
        record = self._sacrificed(by_kind)
        land = replace(record, payload=replace(record.payload, incoming=replace(
            record.payload.incoming, params={"mode": "Moved", "from_zone": "hand",
                                             "to_zone": "battlefield", "cause": "E2"})))
        assert caused_with_empty_stack(land, land.payload.incoming)
        assert FAMILY.match(land, None, self.PARAMS) is None
        assert FAMILY.match(land, None, {"event": "zone_change",
                                         "caused_with_empty_stack": True}) is not None

    def test_it_is_declared_true_or_left_out(self):
        FAMILY.validate(self.PARAMS)
        with pytest.raises(ValueError, match="caused_with_empty_stack"):
            FAMILY.validate({**self.PARAMS, "caused_with_empty_stack": False})
        with pytest.raises(ValueError, match="boolean"):
            FAMILY.validate({**self.PARAMS, "caused_with_empty_stack": "yes"})
