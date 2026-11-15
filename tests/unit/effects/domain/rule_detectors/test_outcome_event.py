"""``outcome_event`` on real gen-2 effect halves."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.records import CollectionMode, ResolutionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import acting_line, acting_root_params
from effects.domain.rule_detectors.outcome_event import (
    FAMILY,
    acting_api,
    source_is_token_of_kind,
    token_kind_of,
)

PREDEFINED = {"treasure", "food", "clue", "blood", "map", "gold", "powerstone", "shard",
              "incubator", "junk"}


def _token_maker(by_kind, sidecars):
    return next(r for r in by_kind("resolution")
                if r.moment.value == "resolution" and acting_api(r, sidecars)
                and any(e.type is EventType.TOKEN_CREATED for e in r.payload.events))


def test_a_resolution_creating_a_token_fires(by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    found = FAMILY.match(record, gen2_sidecars, {"event": "token_created"})
    assert found.polarity is Polarity.FIRES and found.observed


def test_the_same_resolution_edited_to_create_nothing_is_the_near_miss_of_its_api(
        by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    api = acting_api(record, gen2_sidecars)
    edited = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    found = FAMILY.match(edited, gen2_sidecars, {"event": "token_created", "api": api})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    assert FAMILY.match(edited, gen2_sidecars, {"event": "token_created"}) is None


@pytest.mark.parametrize("hookless", [{"mode": CollectionMode.DEGRADED}])
def test_a_hookless_record_without_a_hook_event_shows_no_side(by_kind, gen2_sidecars, hookless):
    """V5 #15: a degraded worker hears no hook, so a missing ``token_created``
    (an outcome-hook event) may have happened: neither the near miss nor an
    ``absent`` declaration's ``fires`` side is read; the event present still fires."""
    record = _token_maker(by_kind, gen2_sidecars)
    api = acting_api(record, gen2_sidecars)
    emptied = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    assert emptied.mode is CollectionMode.PATCHED
    for params in ({"event": "token_created", "api": api},
                   {"event": "token_created", "api": api, "absent": True}):
        assert FAMILY.match(emptied, gen2_sidecars, params) is not None
        assert FAMILY.match(replace(emptied, **hookless), gen2_sidecars, params) is None
    found = FAMILY.match(replace(record, **hookless), gen2_sidecars,
                         {"event": "token_created", "api": api})
    assert found.polarity is Polarity.FIRES
    # A bus event's absence stays readable on a hookless record.
    bus = FAMILY.match(replace(emptied, **hookless), gen2_sidecars,
                       {"event": "zone_change", "api": api, "to_zone": "exile"})
    assert bus is not None


def test_a_forced_resolution_without_a_hook_event_is_a_near_miss(by_kind, gen2_sidecars):
    """A patched worker's forced resolution hears the clause and outcome
    hooks (``ForkHooks``), so its missing ``token_created`` did not happen,
    as on a live record."""
    record = _token_maker(by_kind, gen2_sidecars)
    api = acting_api(record, gen2_sidecars)
    kept = tuple(e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)
    forced = replace(record, interventional=True, fork=True, link_id=None,
                     payload=ResolutionPayload(events=kept))
    found = FAMILY.match(forced, gen2_sidecars, {"event": "token_created", "api": api})
    assert found is not None and found.polarity is Polarity.DOES_NOT_FIRE


def test_another_api_shows_nothing(by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    assert FAMILY.match(record, gen2_sidecars,
                        {"event": "token_created", "api": "NoSuchApi"}) is None


def test_zones_and_count_narrow_the_event(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("resolution") if r.moment.value == "resolution"
                  and any(e.type is EventType.ZONE_CHANGE for e in r.payload.events))
    move = next(e for e in record.payload.events if e.type is EventType.ZONE_CHANGE)
    zones = {k: move.params[k] for k in ("from_zone", "to_zone") if k in move.params}
    assert FAMILY.match(record, None, {"event": "zone_change", **zones}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"event": "zone_change", "to_zone": "nowhere"}) is None
    assert FAMILY.match(record, None, {"event": "zone_change", "min_count": 999}) is None


def _loyalty_change(gen2_records):
    return next(r for r in gen2_records if r.kind.value == "resolution"
                and r.moment.value == "resolution" and any(
                    e.type is EventType.COUNTER_CHANGE
                    and str(e.params.get("counter_type")).upper() == "LOYALTY"
                    for e in r.payload.listed_events()))


class TestEventParameters:
    def test_a_parameter_of_the_event_narrows_it_without_case(self, gen2_records):
        record = _loyalty_change(gen2_records)
        declared = {"event": "counter_change", "counter_type": "loyalty"}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES

    def test_another_value_shows_nothing(self, gen2_records):
        record = _loyalty_change(gen2_records)
        assert FAMILY.match(record, None, {"event": "counter_change",
                                           "counter_type": "P1P1"}) is None

    def test_a_parameter_the_event_does_not_carry_is_refused(self, gen2_records):
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": "zone_change", "counter_type": "x"})
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": "zone_change", "cause": "E1"})


class TestAbsent:
    def test_the_half_without_the_event_is_the_fires_side(self, by_kind, gen2_sidecars):
        record = _token_maker(by_kind, gen2_sidecars)
        api = acting_api(record, gen2_sidecars)
        declared = {"event": "token_created", "api": api, "absent": True}
        found = FAMILY.match(record, gen2_sidecars, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
        found = FAMILY.match(edited, gen2_sidecars, declared)
        assert found.polarity is Polarity.FIRES and found.observed
        # The outcome is an absence: the tag names its condition alone.
        assert found.outcome == () and record.actor_player in found.condition

    def test_absent_needs_an_api(self, gen2_records):
        with pytest.raises(ValueError, match="needs api"):
            FAMILY.validate({"event": "token_created", "absent": True})


class TestTokenKinds:
    def test_a_predefined_token_made_shows_by_its_kind(self, gen2_records):
        record, kind = next(
            (r, sorted(token_kind_of(e) & PREDEFINED)[0])
            for r in gen2_records if r.kind.value == "resolution"
            for e in r.payload.listed_events()
            if e.type is EventType.TOKEN_CREATED and token_kind_of(e) & PREDEFINED)
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": kind.title()}).observed
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": "Nonesuch"}) is None

    def test_token_kind_reads_token_created_only(self, gen2_records):
        with pytest.raises(ValueError, match="token_kind"):
            FAMILY.validate({"event": "zone_change", "token_kind": "Food"})

    def test_an_ability_of_a_predefined_token_shows_by_its_source(self, gen2_records):
        record, kind = next(
            (r, sorted({s.lower() for s in source.subtypes} & PREDEFINED)[0])
            for r in gen2_records if r.kind.value == "resolution"
            and r.moment.value == "resolution"
            for source in r.state.entities
            if source.id == r.state.refs.source and source.token_script_id
            and {s.lower() for s in source.subtypes} & PREDEFINED)
        event = next(e for e in record.payload.listed_events()).type.value
        assert source_is_token_of_kind(record.state, kind)
        assert FAMILY.match(record, None, {"event": event, "source_token_kind": kind}) is not None
        assert FAMILY.match(record, None, {"event": event,
                                           "source_token_kind": "nonesuch"}) is None


def _effect_halves(by_kind):
    return [r for r in by_kind("resolution") if r.moment.value == "resolution"]


class TestKeyword:
    @staticmethod
    def _keyword_resolution(by_kind, sidecars):
        return next(r for r in _effect_halves(by_kind) if r.payload.events
                    and acting_line(r, sidecars) is not None
                    and acting_line(r, sidecars).keyword_display_name)

    def test_an_ability_a_keyword_generates_shows_by_its_keyword(self, by_kind, gen2_sidecars):
        record = self._keyword_resolution(by_kind, gen2_sidecars)
        name = acting_line(record, gen2_sidecars).keyword_display_name
        event = record.payload.events[0].type.value
        found = FAMILY.match(record, gen2_sidecars, {"event": event, "keyword": name})
        assert found.polarity is Polarity.FIRES and found.observed
        spelled = name.lower().replace(" ", "_")
        assert FAMILY.match(record, gen2_sidecars, {"event": event, "keyword": spelled})
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": event, "keyword": "no such keyword"}) is None

    def test_the_keyword_scopes_the_near_miss_as_api_does(self, by_kind, gen2_sidecars):
        record = self._keyword_resolution(by_kind, gen2_sidecars)
        name = acting_line(record, gen2_sidecars).keyword_display_name
        event = record.payload.events[0].type
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not event)))
        found = FAMILY.match(edited, gen2_sidecars, {"event": event.value, "keyword": name})
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        absent = FAMILY.match(edited, gen2_sidecars,
                              {"event": event.value, "keyword": name, "absent": True})
        assert absent.polarity is Polarity.FIRES

    def test_a_script_line_is_no_keyword_line(self, by_kind, gen2_sidecars):
        record = _token_maker(by_kind, gen2_sidecars)
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": "token_created", "keyword": "Equip"}) is None

    def test_beside_an_api_a_keyword_admits_its_own_line_too(self, by_kind, gen2_sidecars):
        keyworded = self._keyword_resolution(by_kind, gen2_sidecars)
        name = acting_line(keyworded, gen2_sidecars).keyword_display_name
        scripted = _token_maker(by_kind, gen2_sidecars)
        api = acting_api(scripted, gen2_sidecars)
        for record in (keyworded, scripted):
            event = record.payload.events[0].type.value
            assert FAMILY.match(record, gen2_sidecars,
                                {"event": event, "api": api, "keyword": name}) is not None
        event = scripted.payload.events[0].type.value
        assert FAMILY.match(scripted, gen2_sidecars, {"event": event, "api": "NoSuchApi",
                                                      "keyword": name}) is None


class TestScriptParam:
    @staticmethod
    def _with_param(by_kind, sidecars):
        for record in _effect_halves(by_kind):
            found = acting_root_params(record, sidecars) or {}
            name = next((k for k in found if k not in ("SP", "AB", "DB") and found[k]), None)
            if record.payload.events and name:
                return record, name, found[name], record.payload.events[0].type.value
        raise AssertionError("no effect half acting through a script with a parameter")

    def test_a_parameter_present_or_holding_a_value_narrows_the_line(
            self, by_kind, gen2_sidecars):
        record, name, value, event = self._with_param(by_kind, gen2_sidecars)
        for spec in (name, {name: "*"}, {name: value.upper()}):
            assert FAMILY.match(record, gen2_sidecars,
                                {"event": event, "script_param": spec}).observed
        for spec in ("NoSuchParam", {name: value + "-not"}):
            assert FAMILY.match(record, gen2_sidecars,
                                {"event": event, "script_param": spec}) is None

    def test_a_malformed_parameter_is_refused(self, by_kind, gen2_sidecars):
        record, _, _, event = self._with_param(by_kind, gen2_sidecars)
        with pytest.raises(ValueError, match="script_param"):
            FAMILY.validate({"event": event, "script_param": ["a"]})


def _in_combat(record, subject):
    entity = record.state.entity(subject)
    return entity is not None and entity.combat is not None and bool(
        entity.combat.attacking or entity.combat.blocking)


class TestQualifiers:
    def test_a_creature_leaving_combat_fires_and_one_out_of_combat_is_the_near_miss(
            self, by_kind):
        record = next(r for r in _effect_halves(by_kind) for e in r.payload.events
                      if e.type is EventType.ZONE_CHANGE
                      and e.params.get("from_zone") == "battlefield"
                      and any(_in_combat(r, s) for s in e.subjects))
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_in_combat": True}
        found = FAMILY.match(record, None, declared)
        assert found.polarity is Polarity.FIRES and found.observed
        assert any(_in_combat(record, s) for s in found.entities)
        out_of_combat = replace(record, state=replace(record.state, entities=tuple(
            replace(e, combat=None) for e in record.state.entities)))
        found = FAMILY.match(out_of_combat, None, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed

    def test_a_token_subject_reads_off_the_snapshot(self, by_kind):
        no_token = SimpleNamespace(token_script_id=None)
        record, event = next(
            (r, e) for r in _effect_halves(by_kind) for e in r.payload.events
            if any((r.state.entity(s) or no_token).token_script_id for s in e.subjects))
        declared = {"event": event.type.value, "subject_is_token": True}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES
        carded = replace(record, state=replace(record.state, entities=tuple(
            replace(e, token_script_id=None) for e in record.state.entities)))
        assert FAMILY.match(carded, None, declared).polarity is Polarity.DOES_NOT_FIRE

    def test_another_event_in_the_record_is_with_or_without(self, by_kind):
        record = next(r for r in _effect_halves(by_kind)
                      if len({e.type for e in r.payload.events}) >= 2)
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
            found = FAMILY.match(record, None, {"event": first, name: value})
            assert found.polarity is polarity, (name, value)

    def test_controller_is_owner_needs_the_gen3_owner(self):
        assert FAMILY.needs({"event": "zone_change", "controller_is_owner": False}) == {"owner"}
        assert FAMILY.needs({"event": "zone_change"}) == frozenset()

    def test_a_qualifier_and_absent_are_refused_together(self, gen2_records):
        with pytest.raises(ValueError, match="qualifier"):
            FAMILY.validate({"event": "zone_change", "api": "Draw",
                                                 "absent": True, "subject_is_token": True})


class TestSourceDies:
    """``source_dies`` reads whether a damage event's source dies in the same
    record (510.2: a creature dying in a combat damage step still dealt its
    damage there), on a real gen-2 combat record, which names the source as
    the event's ``cause``; a gen-3 record names it ``source`` too."""

    DECLARED = {"event": "damage_dealt", "combat": True, "source_dies": True}

    @staticmethod
    def _combat(by_kind):
        """A real combat record holding combat damage, and its damage events."""
        for record in by_kind("combat"):
            dealt = [e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
                     and e.params.get("combat") and e.params.get("cause")]
            if dealt:
                return record, dealt
        pytest.fail("no combat record in the fixture deals combat damage")

    @staticmethod
    def _events(record, events):
        return replace(record, payload=replace(record.payload, events=tuple(events)))

    @staticmethod
    def _dies(subject):
        return Event(type=EventType.ZONE_CHANGE, subjects=(subject,),
                     params={"from_zone": "battlefield", "to_zone": "graveyard"})

    def test_a_source_that_dies_in_the_step_fires(self, by_kind):
        record, dealt = self._combat(by_kind)
        source = dealt[0].params["cause"]
        traded = self._events(record, (*dealt, self._dies(source)))
        found = FAMILY.match(traded, None, self.DECLARED)
        assert found.polarity is Polarity.FIRES and found.observed

    def test_a_victim_dying_to_a_source_that_survives_is_the_near_miss(self, by_kind):
        """The first-strike step: the striker deals its damage and lives, the
        creature it kills dealt none."""
        record, dealt = self._combat(by_kind)
        striker = dealt[0]
        victim = striker.subjects[0]
        struck_first = self._events(record, (striker, self._dies(victim)))
        assert victim != striker.params["cause"]
        found = FAMILY.match(struck_first, None, self.DECLARED)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed

    def test_false_flips_the_sides(self, by_kind):
        record, dealt = self._combat(by_kind)
        striker = dealt[0]
        declared = {**self.DECLARED, "source_dies": False}
        died = self._events(record, (striker, self._dies(striker.params["cause"])))
        assert FAMILY.match(died, None, declared).polarity is Polarity.DOES_NOT_FIRE
        assert FAMILY.match(self._events(record, (striker,)), None,
                            declared).polarity is Polarity.FIRES

    def test_a_gen3_source_is_read_before_the_cause(self, by_kind):
        record, dealt = self._combat(by_kind)
        named = [replace(e, params={**e.params, "source": e.params["cause"], "cause": "E0"})
                 for e in dealt]
        traded = self._events(record, (*named, self._dies(dealt[0].params["cause"])))
        assert FAMILY.match(traded, None, self.DECLARED).polarity is Polarity.FIRES

    def test_it_reads_an_event_that_names_a_source_only(self):
        FAMILY.validate({"event": "damage_prevented", "source_dies": True})
        with pytest.raises(ValueError, match="source_dies"):
            FAMILY.validate({"event": "zone_change", "source_dies": True})
        with pytest.raises(ValueError, match="qualifier"):
            FAMILY.validate({"event": "damage_dealt", "api": "DealDamage", "absent": True,
                             "source_dies": True})


def test_a_combat_records_events_show_too(by_kind):
    record = next(r for r in by_kind("combat") if not r.fork and any(
        e.type is EventType.DAMAGE_DEALT for e in r.payload.events))
    assert FAMILY.match(record, None, {"event": "damage_dealt"}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"event": "damage_dealt", "api": "DealDamage"}) is None
    # The same combat played in a damage-step probe: its events show alike.
    probe = replace(record, fork=True, mirror_of=record.record_id,
                    payload=replace(record.payload, probed_keyword="trample",
                                    probed_entity=record.payload.attackers[0]))
    assert probe.is_probe
    assert FAMILY.match(probe, None, {"event": "damage_dealt"}).polarity is Polarity.FIRES


class TestT036Shapes:
    """The shapes gen-2 writes where a declaration named an event it never writes."""

    def test_a_counter_type_compares_in_one_spelling(self, gen2_records):
        """Effect halves write ``+1/+1`` and ``P1P1`` both."""
        record = next(r for r in gen2_records if r.kind.value == "resolution"
                      and r.moment.value == "resolution" and any(
                          e.type is EventType.COUNTER_CHANGE
                          and str(e.params.get("counter_type")) in ("+1/+1", "P1P1")
                          for e in r.payload.listed_events()))
        for spelled in ("+1/+1", "P1P1", "p1p1"):
            found = FAMILY.match(record, None, {"event": "counter_change",
                                                "counter_type": spelled})
            assert found.polarity is Polarity.FIRES, spelled

    def _self_move(self, by_kind):
        """A real effect half moving its own source off the battlefield, with
        another event beside it (a Saga's last chapter, then its sacrifice)."""
        for record in _effect_halves(by_kind):
            source = record.state.refs.source
            for event in record.payload.events:
                if (event.type is EventType.ZONE_CHANGE and source in event.subjects
                        and event.params.get("from_zone") == "battlefield"
                        and len(record.payload.events) >= 2):
                    return record, event
        pytest.fail("no effect half in the fixture moves its own source")

    def test_a_move_of_the_source_itself_is_what_subject_is_source_reads(self, by_kind):
        record, event = self._self_move(by_kind)
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_is_source": True}
        found = FAMILY.match(record, None, declared)
        assert found.polarity is Polarity.FIRES and record.state.refs.source in found.entities
        elsewhere = replace(record, state=replace(record.state, refs=replace(
            record.state.refs, source="E0")))
        assert FAMILY.match(elsewhere, None, declared).polarity is Polarity.DOES_NOT_FIRE

    def test_with_event_takes_the_event_s_filters(self, by_kind):
        record, event = self._self_move(by_kind)
        other = next(e for e in record.payload.events if e is not event)
        to_zone = event.params.get("to_zone")
        declared = {"event": other.type.value,
                    "with_event": {"event": "zone_change", "from_zone": "battlefield",
                                   "to_zone": to_zone}}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES
        declared["with_event"] = {**declared["with_event"], "to_zone": "nowhere"}
        assert FAMILY.match(record, None, declared).polarity is Polarity.DOES_NOT_FIRE
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": other.type.value,
                                        "with_event": {"event": "zone_change", "amount": 1}})


class TestApiInChain:
    @staticmethod
    def _chained(by_kind, sidecars):
        """An effect half whose acting line chains a sub-ability of another API."""
        from effects.domain.script_text import api_of, params, segments

        for record in _effect_halves(by_kind):
            line = acting_line(record, sidecars)
            if line is None or not line.script_text or not record.payload.events:
                continue
            apis = [api_of(params(s)) for s in segments(line.script_text)]
            sub = next((a for a in apis[1:] if a and a != apis[0]), None)
            if apis[0] and sub:
                return record, apis[0], sub, record.payload.events[0].type.value
        raise AssertionError("no effect half acting through a chained script")

    def test_a_sub_ability_s_api_is_read_only_when_asked_for(self, by_kind, gen2_sidecars):
        record, _, sub, event = self._chained(by_kind, gen2_sidecars)
        assert FAMILY.match(record, gen2_sidecars, {"event": event, "api": sub}) is None
        found = FAMILY.match(record, gen2_sidecars,
                             {"event": event, "api": sub, "api_in_chain": True})
        assert found.polarity is Polarity.FIRES

    def test_the_root_api_still_matches_in_the_chain(self, by_kind, gen2_sidecars):
        record, root, _, event = self._chained(by_kind, gen2_sidecars)
        for declared in ({"event": event, "api": root},
                         {"event": event, "api": root, "api_in_chain": True}):
            assert FAMILY.match(record, gen2_sidecars, declared).polarity is Polarity.FIRES
        assert FAMILY.match(record, gen2_sidecars, {"event": event, "api": "NoSuchApi",
                                                    "api_in_chain": True}) is None


class TestEntityRoles:
    """Condition: the acting source, its targets and controller; outcome:
    where the events are placed, and a hand's or a library's owner."""

    @staticmethod
    def _resolution(*events, entities=(), targets=()):
        from effects.domain.state_snapshot import PlayerState, Refs
        from tests.unit.effects.builders import make_entity, make_record, make_snapshot

        players = tuple(PlayerState(id=p, life=20, hand=3, library=30, graveyard=0)
                        for p in ("P0", "P1"))
        board = (make_entity("E1", "source"), *entities)
        state = make_snapshot(entities=board, players=players,
                              refs=Refs(source="E1", targets=tuple(targets)))
        return make_record(state=state, payload=ResolutionPayload(events=events))

    def test_a_discard_names_the_card_and_the_player_who_discarded_it(self):
        from tests.unit.effects.builders import make_entity

        card = make_entity("E5", "discarded", zone="hand", controller="P1", owner="P1")
        record = self._resolution(
            Event(EventType.ZONE_CHANGE, ("E5",), {"from_zone": "hand", "to_zone": "graveyard"}),
            entities=(card,), targets=("P1",))
        found = FAMILY.match(record, None, {"event": "zone_change", "from_zone": "hand",
                                            "to_zone": "graveyard"})
        assert found.polarity is Polarity.FIRES
        assert found.condition == ("E1", "P1", "P0")
        assert found.outcome == ("E5", "P1")

    def test_a_spell_copy_is_placed_on_the_targeted_stack_object(self):
        from tests.unit.effects.builders import make_entity

        spell = make_entity("E7", "spell", zone="stack")
        record = self._resolution(Event(EventType.SPELL_COPIED, (), {"count": 1}),
                                  entities=(spell,), targets=("E7",))
        found = FAMILY.match(record, None, {"event": "spell_copied"})
        assert found.polarity is Polarity.FIRES and found.outcome == ("E7",)


def _leaving(by_kind):
    """An effect half moving a snapshot permanent off the battlefield, and that permanent."""
    return next((r, s) for r in _effect_halves(by_kind) for e in r.payload.events
                if e.type is EventType.ZONE_CHANGE
                and e.params.get("from_zone") == "battlefield"
                for s in e.subjects if r.state.entity(s) is not None)


def _with_subject(record, subject, **changes):
    """``record`` with its entity ``subject`` changed, as a gen-3 entity (lines listed)."""
    return replace(record, state=replace(record.state, entities=tuple(
        replace(e, lines=(), **changes) if e.id == subject else e
        for e in record.state.entities)))


class TestGen3SubjectQualifiers:
    """A melded, merged, copied or designated subject (T030 follow-up)."""

    _MOVE = {"event": "zone_change", "from_zone": "battlefield"}

    def test_a_melded_subject_fires_and_a_front_one_is_the_near_miss(self, by_kind):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_face_state": "melded"}
        found = FAMILY.match(_with_subject(record, subject, face_state="melded"), None, declared)
        assert found.polarity is Polarity.FIRES and subject in found.outcome
        found = FAMILY.match(_with_subject(record, subject, face_state="front"), None, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE

    def test_a_gen2_subject_has_no_face_state_to_read(self, by_kind):
        record, _ = _leaving(by_kind)
        assert FAMILY.match(record, None, {**self._MOVE, "subject_face_state": "melded"}) is None

    def test_a_designation_is_read_off_the_gen3_subject(self, by_kind):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_designation": "unearthed"}
        held = _with_subject(record, subject, designations=("unearthed",))
        assert FAMILY.match(held, None, declared).polarity is Polarity.FIRES
        lacking = _with_subject(record, subject, designations=())
        assert FAMILY.match(lacking, None, declared).polarity is Polarity.DOES_NOT_FIRE

    def test_a_merged_subject_is_one_a_merged_with_link_reaches(self, by_kind):
        from effects.domain.state_snapshot import Link

        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_merged": True}
        merged = replace(record, state=replace(record.state, links=(
            Link(from_="E999999", to=subject, type="merged_with"),)))
        assert FAMILY.match(merged, None, declared).polarity is Polarity.FIRES
        alone = replace(record, state=replace(record.state, links=()))
        assert FAMILY.match(alone, None, declared).polarity is Polarity.DOES_NOT_FIRE
        # A snapshot with no links at all (gen-2) cannot say.
        assert FAMILY.match(record, None, declared) is None

    def test_a_copy_names_its_copy_source(self, by_kind):
        record, subject = _leaving(by_kind)
        declared = {**self._MOVE, "subject_is_copy": True}
        copied = replace(record, state=replace(record.state, entities=tuple(
            replace(e, copy_source="E1") if e.id == subject else replace(e, copy_source=None)
            for e in record.state.entities)))
        assert FAMILY.match(copied, None, declared).polarity is Polarity.FIRES
        uncopied = replace(copied, state=replace(copied.state, entities=tuple(
            replace(e, copy_source=None) for e in copied.state.entities)))
        assert FAMILY.match(uncopied, None, declared).polarity is Polarity.DOES_NOT_FIRE

    def test_each_needs_its_gen3_field(self):
        assert FAMILY.needs({"event": "zone_change", "subject_face_state": "melded"}) == {
            "face_state"}
        assert FAMILY.needs({"event": "zone_change", "subject_designation": "x"}) == {
            "designations"}
        assert FAMILY.needs({"event": "zone_change", "subject_merged": True}) == {"links"}
        assert FAMILY.needs({"event": "zone_change", "subject_is_copy": True}) == frozenset()


class TestChainParam:
    """A parameter on any segment of the acting line's chain (T030 follow-up)."""

    def test_a_sub_ability_parameter_narrows_the_line(self, by_kind, gen2_sidecars):
        from effects.domain.script_text import params, segments

        record, below = next(
            (r, name) for r in _effect_halves(by_kind) if r.payload.events
            for line in [acting_line(r, gen2_sidecars)]
            if line is not None and line.script_text and len(segments(line.script_text)) > 1
            for name in params(segments(line.script_text)[1])
            if name not in (acting_root_params(r, gen2_sidecars) or {}))
        event = record.payload.events[0].type.value
        assert FAMILY.match(record, gen2_sidecars, {"event": event, "script_param": below}) is None
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": event, "chain_param": below}).polarity is Polarity.FIRES
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": event, "chain_param": "NoSuchParam"}) is None
