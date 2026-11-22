"""``outcome_event``'s qualifiers on real gen-3 effect halves: subject, source and event
qualifiers, the shapes a declaration reads another way, and the gen-3 subject fields."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.outcome_event import FAMILY, acting_api
from tests.unit.effects.domain.rule_detectors.builders import effect_halves


def _on_its_api(record, lines, declared):
    """``declared`` scoped to ``record``'s own acting API, the scope a
    qualifier's near miss needs (finding 7.1)."""
    api = acting_api(record, lines)
    assert api is not None
    return {**declared, "api": api}


def _in_combat(record, subject):
    entity = record.state.entity(subject)
    return entity is not None and entity.combat is not None and bool(
        entity.combat.attacking or entity.combat.blocking)


def _combat_departure(by_kind):
    """An effect half moving a creature in combat off the battlefield."""
    return next(r for r in effect_halves(by_kind) for e in r.payload.events
                if e.type is EventType.ZONE_CHANGE
                and e.params.get("from_zone") == "battlefield"
                and any(_in_combat(r, s) for s in e.subjects))


class TestQualifiers:
    def test_a_creature_leaving_combat_fires_and_one_out_of_combat_is_the_near_miss(
            self, by_kind, gen3_sidecars):
        record = _combat_departure(by_kind)
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_in_combat": True}
        found = FAMILY.match(record, None, declared)
        assert found.polarity is Polarity.FIRES and found.observed
        assert any(_in_combat(record, s) for s in found.entities)
        out_of_combat = replace(record, state=replace(record.state, entities=tuple(
            replace(e, combat=None) for e in record.state.entities)))
        assert FAMILY.match(out_of_combat, None, declared) is None
        found = FAMILY.match(out_of_combat, gen3_sidecars,
                             _on_its_api(record, gen3_sidecars, declared))
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed

    def test_a_token_subject_reads_off_the_snapshot(self, by_kind, gen3_sidecars):
        """No fixture effect half acts on a token, so the subject of a real
        move is edited into one."""
        record, subject = _leaving(by_kind)
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_is_token": True}
        token = replace(record, state=replace(record.state, entities=tuple(
            replace(e, token_script_id="g_1_1_wolf") if e.id == subject else e
            for e in record.state.entities)))
        assert FAMILY.match(token, None, declared).polarity is Polarity.FIRES
        carded = replace(record, state=replace(record.state, entities=tuple(
            replace(e, token_script_id=None) for e in record.state.entities)))
        assert FAMILY.match(carded, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE

    def test_another_event_in_the_record_is_with_or_without(self, by_kind, gen3_sidecars):
        record = next(r for r in effect_halves(by_kind)
                      if len({e.type for e in r.payload.events}) >= 2
                      and acting_api(r, gen3_sidecars))
        first, other = list(dict.fromkeys(e.type.value for e in record.payload.events))[:2]
        missing = next(t.value for t in EventType
                       if t not in {e.type for e in record.payload.events})
        sides = {
            ("with_event", other): Polarity.FIRES,
            ("without_event", other): Polarity.DOES_NOT_FIRE,
            ("with_event", missing): Polarity.DOES_NOT_FIRE,
            ("without_event", missing): Polarity.FIRES,
        }
        for (name, value), polarity in sides.items():
            found = FAMILY.match(record, gen3_sidecars, _on_its_api(
                record, gen3_sidecars, {"event": first, name: value}))
            assert found.polarity is polarity, (name, value)

    def test_controller_is_owner_needs_the_gen3_owner(self):
        assert FAMILY.needs({"event": "zone_change", "controller_is_owner": False}) == {"owner"}
        assert FAMILY.needs({"event": "zone_change"}) == frozenset()

    def test_a_qualifier_and_absent_are_refused_together(self, gen3_records):
        with pytest.raises(ValueError, match="qualifier"):
            FAMILY.validate({"event": "zone_change", "api": "Draw",
                                                 "absent": True, "subject_is_token": True})


def _leaves_battlefield(event):
    return event.type is EventType.ZONE_CHANGE and event.params.get("from_zone") == "battlefield"


class TestT036Shapes:
    """The shapes where a declaration named an event the record writes another way."""

    def test_a_counter_type_compares_in_one_spelling(self, gen3_records):
        """A record may write ``+1/+1`` or ``P1P1``. The fixture's counter
        changes are loyalty and level, so a real one is edited to each spelling."""
        record = next(r for r in gen3_records if r.kind.value == "resolution"
                      and r.moment.value == "resolution" and any(
                          e.type is EventType.COUNTER_CHANGE for e in r.payload.events))
        for written in ("+1/+1", "P1P1"):
            events = tuple(
                replace(e, params={**e.params, "counter_type": written})
                if e.type is EventType.COUNTER_CHANGE else e for e in record.payload.events)
            edited = replace(record, payload=replace(record.payload, events=events))
            for spelled in ("+1/+1", "P1P1", "p1p1"):
                found = FAMILY.match(edited, None, {"event": "counter_change",
                                                    "counter_type": spelled})
                assert found.polarity is Polarity.FIRES, (written, spelled)

    def _self_move(self, by_kind):
        """A real effect half moving its own source off the battlefield (a
        sacrifice), edited to hold another event beside it (a Saga's last
        chapter, then its sacrifice)."""
        for record in effect_halves(by_kind):
            source = record.state.refs.source
            for event in filter(_leaves_battlefield, record.payload.events):
                if source in event.subjects:
                    beside = Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                                   params={"delta": 1})
                    edited = replace(record, payload=replace(
                        record.payload, events=(*record.payload.events, beside)))
                    return edited, event
        pytest.fail("no effect half in the fixture moves its own source")

    def test_a_move_of_the_source_itself_is_what_subject_is_source_reads(
            self, by_kind, gen3_sidecars):
        record, event = self._self_move(by_kind)
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_is_source": True}
        found = FAMILY.match(record, None, declared)
        assert found.polarity is Polarity.FIRES and record.state.refs.source in found.entities
        elsewhere = replace(record, state=replace(record.state, refs=replace(
            record.state.refs, source="E0")))
        assert FAMILY.match(elsewhere, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE

    def test_with_event_takes_the_event_s_filters(self, by_kind, gen3_sidecars):
        record, event = self._self_move(by_kind)
        other = next(e for e in record.payload.events if e is not event)
        to_zone = event.params.get("to_zone")
        declared = {"event": other.type.value,
                    "with_event": {"event": "zone_change", "from_zone": "battlefield",
                                   "to_zone": to_zone}}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES
        declared["with_event"] = {**declared["with_event"], "to_zone": "nowhere"}
        assert FAMILY.match(record, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": other.type.value,
                                        "with_event": {"event": "zone_change", "amount": 1}})


def _leaving(by_kind):
    """An effect half moving a snapshot permanent off the battlefield, and that permanent."""
    return next((r, s) for r in effect_halves(by_kind) for e in r.payload.events
                if e.type is EventType.ZONE_CHANGE
                and e.params.get("from_zone") == "battlefield"
                for s in e.subjects if r.state.entity(s) is not None)


def _with_subject(record, subject, **changes):
    """``record`` with its entity ``subject`` changed."""
    return replace(record, state=replace(record.state, entities=tuple(
        replace(e, **changes) if e.id == subject else e
        for e in record.state.entities)))


class TestGen3SubjectQualifiers:
    """A melded, merged, copied or designated subject (T030 follow-up)."""

    _MOVE = {"event": "zone_change", "from_zone": "battlefield"}

    def test_a_melded_subject_fires_and_a_front_one_is_the_near_miss(
            self, by_kind, gen3_sidecars):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_face_state": "melded"}
        found = FAMILY.match(_with_subject(record, subject, face_state="melded"), None, declared)
        assert found.polarity is Polarity.FIRES and subject in found.outcome
        found = FAMILY.match(_with_subject(record, subject, face_state="front"), gen3_sidecars,
                             _on_its_api(record, gen3_sidecars, declared))
        assert found.polarity is Polarity.DOES_NOT_FIRE

    def test_a_designation_is_read_off_the_gen3_subject(self, by_kind, gen3_sidecars):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_designation": "unearthed"}
        held = _with_subject(record, subject, designations=("unearthed",))
        assert FAMILY.match(held, None, declared).polarity is Polarity.FIRES
        lacking = _with_subject(record, subject, designations=())
        assert FAMILY.match(lacking, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE

    def test_a_merged_subject_is_one_a_merged_with_link_reaches(self, by_kind, gen3_sidecars):
        from effects.domain.state_snapshot import Link

        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_merged": True}
        merged = replace(record, state=replace(record.state, links=(
            Link(from_="E999999", to=subject, type="merged_with"),)))
        assert FAMILY.match(merged, None, declared).polarity is Polarity.FIRES
        alone = replace(record, state=replace(record.state, links=()))
        assert FAMILY.match(alone, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE

    def test_a_copy_names_its_copy_source(self, by_kind, gen3_sidecars):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_is_copy": True}
        copied = replace(record, state=replace(record.state, entities=tuple(
            replace(e, copy_source="E1") if e.id == subject else replace(e, copy_source=None)
            for e in record.state.entities)))
        assert FAMILY.match(copied, None, declared).polarity is Polarity.FIRES
        uncopied = replace(copied, state=replace(copied.state, entities=tuple(
            replace(e, copy_source=None) for e in copied.state.entities)))
        assert FAMILY.match(uncopied, gen3_sidecars, _on_its_api(
            record, gen3_sidecars, declared)).polarity is Polarity.DOES_NOT_FIRE

    def test_each_needs_its_gen3_field(self):
        assert FAMILY.needs({"event": "zone_change", "subject_face_state": "melded"}) == {
            "face_state"}
        assert FAMILY.needs({"event": "zone_change", "subject_designation": "x"}) == {
            "designations"}
        assert FAMILY.needs({"event": "zone_change", "subject_merged": True}) == {"links"}
        assert FAMILY.needs({"event": "zone_change", "subject_is_copy": True}) == frozenset()
