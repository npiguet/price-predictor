"""``rewrite_event`` on real gen-2 rewrite records (T029)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import RewriteResult
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import acting_line
from effects.domain.rule_detectors.rewrite_event import FAMILY, is_prevention_line


def _entering(by_kind):
    return next(r for r in by_kind("rewrite")
                if r.payload.incoming.type.value == "zone_change"
                and r.payload.result is RewriteResult.UPDATED)


def test_an_updated_zone_change_fires_an_enters_replacement_rule(by_kind):
    params = {"event": "zone_change", "result": "updated", "to_zone": "battlefield"}
    record = _entering(by_kind)
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
    found = FAMILY.match(edited, None, {"event": "zone_change", "result": "updated"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_the_mode_and_any_incoming_parameter_narrow_the_event(by_kind):
    record = _entering(by_kind)
    incoming = record.payload.incoming
    mode = incoming.params["mode"]
    for declared in ({"mode": mode}, {"mode": mode.lower()},
                     {"to_zone": incoming.params["to_zone"].upper()}):
        found = FAMILY.match(record, None, {"event": "zone_change", "result": "updated",
                                            **declared})
        assert found.polarity is Polarity.FIRES, declared
    assert FAMILY.match(record, None, {"event": "zone_change", "mode": "BeginPhase"}) is None
    with pytest.raises(ValueError, match="no parameter"):
        FAMILY.match(record, None, {"event": "zone_change", "counter_type": "P1P1"})


def test_a_result_list_names_every_fires_result(by_kind):
    record = _entering(by_kind)
    found = FAMILY.match(record, None,
                         {"event": "zone_change", "result": ["replaced", "updated"]})
    assert found.polarity is Polarity.FIRES
    found = FAMILY.match(record, None,
                         {"event": "zone_change", "result": ["replaced", "skipped"]})
    assert found.polarity is Polarity.DOES_NOT_FIRE


def _keyword_rewrite(by_kind, sidecars):
    return next(r for r in by_kind("rewrite") if acting_line(r, sidecars) is not None
                and acting_line(r, sidecars).keyword_display_name)


def test_a_keyword_replacement_shows_by_its_keyword(by_kind, gen2_sidecars):
    record = _keyword_rewrite(by_kind, gen2_sidecars)
    name = acting_line(record, gen2_sidecars).keyword_display_name
    event = record.payload.incoming.type.value
    result = record.payload.result.value
    found = FAMILY.match(record, gen2_sidecars,
                         {"event": event, "result": result, "keyword": name.lower()})
    assert found.polarity is Polarity.FIRES
    assert FAMILY.match(record, gen2_sidecars, {"event": event, "keyword": "Madness"}) is None


class TestPrevention:
    @staticmethod
    def _shield(by_kind, sidecars):
        return next(r for r in by_kind("rewrite")
                    if r.payload.incoming.type.value == "damage_dealt"
                    and r.payload.result in (RewriteResult.REPLACED, RewriteResult.UPDATED)
                    and is_prevention_line(acting_line(r, sidecars)))

    def test_a_shield_replacing_damage_fires_whatever_result_it_reports(
            self, by_kind, gen2_sidecars):
        record = self._shield(by_kind, gen2_sidecars)
        found = FAMILY.match(record, gen2_sidecars,
                             {"event": "damage_dealt", "prevention": True})
        assert found.polarity is Polarity.FIRES and found.observed

    def test_a_shield_preventing_nothing_is_the_near_miss(self, by_kind, gen2_sidecars):
        record = self._shield(by_kind, gen2_sidecars)
        edited = replace(record, payload=replace(
            record.payload, result=RewriteResult.NOT_REPLACED, outgoing=None, replaced_by=()))
        found = FAMILY.match(edited, gen2_sidecars, {"event": "damage_dealt", "prevention": True})
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        found = FAMILY.match(edited, gen2_sidecars, {"event": "damage_dealt",
                                                     "prevention": True,
                                                     "result": "not_replaced"})
        assert found.polarity is Polarity.FIRES

    def test_another_replacement_is_no_prevention_but_a_prevented_result_is(
            self, by_kind, gen2_sidecars):
        record = _keyword_rewrite(by_kind, gen2_sidecars)
        assert not is_prevention_line(acting_line(record, gen2_sidecars))
        event = record.payload.incoming.type.value
        declared = {"event": event, "prevention": True}
        assert FAMILY.match(record, gen2_sidecars, declared) is None
        prevented = replace(record, payload=replace(record.payload,
                                                    result=RewriteResult.PREVENTED))
        assert FAMILY.match(prevented, gen2_sidecars, declared).polarity is Polarity.FIRES


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
                replace(e, lines=(), face_state=face_state) if e.id == subject else e
                for e in record.state.entities)))
        found = FAMILY.match(subject_as("melded"), None, declared)
        assert found.polarity is Polarity.FIRES and subject in found.outcome
        assert FAMILY.match(subject_as("front"), None, declared) is None
        # A gen-2 subject has no face state: the record speaks for nothing.
        assert FAMILY.match(record, None, declared) is None

    def test_a_merged_subject_and_its_need(self, by_kind):
        from effects.domain.state_snapshot import Link

        record, subject = self._rewrite(by_kind)
        merged = replace(record, state=replace(record.state, links=(
            Link(from_="E999999", to=subject, type="merged_with"),)))
        declared = self._declared(record, subject_merged=True)
        assert FAMILY.match(merged, None, declared).polarity is Polarity.FIRES
        assert FAMILY.needs(declared) == {"links"}
        assert FAMILY.needs(self._declared(record)) == frozenset()
