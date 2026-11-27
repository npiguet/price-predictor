"""``outcome_event`` on real gen-3 effect halves: parameters, tokens, keywords and roles."""

from __future__ import annotations

from dataclasses import replace

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
from tests.unit.effects.domain.rule_detectors.builders import effect_halves

PREDEFINED = {"treasure", "food", "clue", "blood", "map", "gold", "powerstone", "shard",
              "incubator", "junk"}


def _token_maker(by_kind, sidecars):
    return next(r for r in by_kind("resolution")
                if r.moment.value == "resolution" and acting_api(r, sidecars)
                and any(e.type is EventType.TOKEN_CREATED for e in r.payload.events))


def test_a_resolution_creating_a_token_fires(by_kind, gen3_sidecars):
    record = _token_maker(by_kind, gen3_sidecars)
    found = FAMILY.match(record, gen3_sidecars, {"event": "token_created"})
    assert found.polarity is Polarity.FIRES and found.observed


def test_the_same_resolution_edited_to_create_nothing_is_the_near_miss_of_its_api(
        by_kind, gen3_sidecars):
    record = _token_maker(by_kind, gen3_sidecars)
    api = acting_api(record, gen3_sidecars)
    edited = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    found = FAMILY.match(edited, gen3_sidecars, {"event": "token_created", "api": api})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    assert FAMILY.match(edited, gen3_sidecars, {"event": "token_created"}) is None


@pytest.mark.parametrize("hookless", [{"mode": CollectionMode.DEGRADED}])
def test_a_hookless_record_without_a_hook_event_shows_no_side(by_kind, gen3_sidecars, hookless):
    """A degraded worker hears no hook, so a missing ``token_created``
    (an outcome-hook event) may have happened: neither the near miss nor an
    ``absent`` declaration's ``fires`` side is read; the event present still fires."""
    record = _token_maker(by_kind, gen3_sidecars)
    api = acting_api(record, gen3_sidecars)
    emptied = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    assert emptied.mode is CollectionMode.PATCHED
    for params in ({"event": "token_created", "api": api},
                   {"event": "token_created", "api": api, "absent": True}):
        assert FAMILY.match(emptied, gen3_sidecars, params) is not None
        assert FAMILY.match(replace(emptied, **hookless), gen3_sidecars, params) is None
    found = FAMILY.match(replace(record, **hookless), gen3_sidecars,
                         {"event": "token_created", "api": api})
    assert found.polarity is Polarity.FIRES
    # A bus event's absence stays readable on a hookless record.
    bus = FAMILY.match(replace(emptied, **hookless), gen3_sidecars,
                       {"event": "zone_change", "api": api, "to_zone": "exile"})
    assert bus is not None


def test_a_forced_resolution_without_a_hook_event_is_a_near_miss(by_kind, gen3_sidecars):
    """A patched worker's forced resolution hears the clause and outcome
    hooks (``ForkHooks``), so its missing ``token_created`` did not happen,
    as on a live record."""
    record = _token_maker(by_kind, gen3_sidecars)
    api = acting_api(record, gen3_sidecars)
    kept = tuple(e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)
    forced = replace(record, interventional=True, fork=True, link_id=None,
                     payload=ResolutionPayload(events=kept))
    found = FAMILY.match(forced, gen3_sidecars, {"event": "token_created", "api": api})
    assert found is not None and found.polarity is Polarity.DOES_NOT_FIRE


def test_a_list_of_apis_counts_a_line_of_any_of_them(by_kind, gen3_sidecars):
    """One mechanic two APIs carry (702.1c's shared keywords, through Pump and
    PumpAll): either side is read for a line of any API listed, and only those."""
    record = _token_maker(by_kind, gen3_sidecars)
    api = acting_api(record, gen3_sidecars)
    edited = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    listed = {"event": "token_created", "api": ["Pump", api]}
    FAMILY.validate(listed)

    assert FAMILY.match(record, gen3_sidecars, listed).polarity is Polarity.FIRES
    assert FAMILY.match(edited, gen3_sidecars, listed).polarity is Polarity.DOES_NOT_FIRE
    assert FAMILY.match(record, gen3_sidecars,
                        {"event": "token_created", "api": ["Pump", "PumpAll"]}) is None
    with pytest.raises(ValueError, match="NoSuchApi"):
        FAMILY.validate({"event": "token_created", "api": ["Pump", "NoSuchApi"]})


def test_another_api_shows_nothing(by_kind, gen3_sidecars):
    record = _token_maker(by_kind, gen3_sidecars)
    assert FAMILY.match(record, gen3_sidecars,
                        {"event": "token_created", "api": "NoSuchApi"}) is None


def test_zones_and_count_narrow_the_event(by_kind, gen3_sidecars):
    record = next(r for r in by_kind("resolution") if r.moment.value == "resolution"
                  and any(e.type is EventType.ZONE_CHANGE for e in r.payload.events))
    move = next(e for e in record.payload.events if e.type is EventType.ZONE_CHANGE)
    zones = {k: move.params[k] for k in ("from_zone", "to_zone") if k in move.params}
    assert FAMILY.match(record, None, {"event": "zone_change", **zones}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"event": "zone_change", "to_zone": "nowhere"}) is None
    assert FAMILY.match(record, None, {"event": "zone_change", "min_count": 999}) is None


def _loyalty_change(gen3_records):
    return next(r for r in gen3_records if r.kind.value == "resolution"
                and r.moment.value == "resolution" and any(
                    e.type is EventType.COUNTER_CHANGE
                    and str(e.params.get("counter_type")).upper() == "LOYALTY"
                    for e in r.payload.listed_events()))


class TestEventParameters:
    def test_a_parameter_of_the_event_narrows_it_without_case(self, gen3_records):
        record = _loyalty_change(gen3_records)
        declared = {"event": "counter_change", "counter_type": "loyalty"}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES

    def test_another_value_shows_nothing(self, gen3_records):
        record = _loyalty_change(gen3_records)
        assert FAMILY.match(record, None, {"event": "counter_change",
                                           "counter_type": "P1P1"}) is None

    def test_a_parameter_the_event_does_not_carry_is_refused(self, gen3_records):
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": "zone_change", "counter_type": "x"})
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.validate({"event": "zone_change", "cause": "E1"})


class TestAbsent:
    def test_the_half_without_the_event_is_the_fires_side(self, by_kind, gen3_sidecars):
        record = _token_maker(by_kind, gen3_sidecars)
        api = acting_api(record, gen3_sidecars)
        declared = {"event": "token_created", "api": api, "absent": True}
        found = FAMILY.match(record, gen3_sidecars, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
        found = FAMILY.match(edited, gen3_sidecars, declared)
        assert found.polarity is Polarity.FIRES and found.observed
        # The outcome is an absence: the tag names where it would have fallen,
        # the token maker's controller where its clause targets nothing.
        assert record.actor_player in found.condition
        assert found.outcome == (record.state.refs.targets or (record.actor_player,))

    def test_absent_needs_an_api(self, gen3_records):
        with pytest.raises(ValueError, match="needs api"):
            FAMILY.validate({"event": "token_created", "absent": True})


def _predefined_kinds(source):
    """The predefined token kinds among the entity's subtypes, when it is a token."""
    if not source.token_script_id:
        return set()
    return {s.lower() for s in source.subtypes} & PREDEFINED


def _predefined_token_source(gen3_records):
    """A resolution whose source is a predefined token, and that token's kind."""
    return next(
        (r, sorted(_predefined_kinds(source))[0])
        for r in gen3_records if r.kind.value == "resolution" and r.moment.value == "resolution"
        for source in r.state.entities
        if source.id == r.state.refs.source and _predefined_kinds(source))


class TestTokenKinds:
    def test_a_predefined_token_made_shows_by_its_kind(self, gen3_records):
        record, kind = next(
            (r, sorted(token_kind_of(e) & PREDEFINED)[0])
            for r in gen3_records if r.kind.value == "resolution"
            for e in r.payload.listed_events()
            if e.type is EventType.TOKEN_CREATED and token_kind_of(e) & PREDEFINED)
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": kind.title()}).observed
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": "Nonesuch"}) is None

    def test_token_kind_reads_token_created_only(self, gen3_records):
        with pytest.raises(ValueError, match="token_kind"):
            FAMILY.validate({"event": "zone_change", "token_kind": "Food"})

    def test_an_ability_of_a_predefined_token_shows_by_its_source(self, gen3_records):
        record, kind = _predefined_token_source(gen3_records)
        event = next(e for e in record.payload.listed_events()).type.value
        assert source_is_token_of_kind(record.state, kind)
        assert FAMILY.match(record, None, {"event": event, "source_token_kind": kind}) is not None
        assert FAMILY.match(record, None, {"event": event,
                                           "source_token_kind": "nonesuch"}) is None


class TestKeyword:
    @staticmethod
    def _keyword_resolution(by_kind, sidecars):
        return next(r for r in effect_halves(by_kind) if r.payload.events
                    and acting_line(r, sidecars) is not None
                    and acting_line(r, sidecars).keyword_display_name)

    def test_an_ability_a_keyword_generates_shows_by_its_keyword(self, by_kind, gen3_sidecars):
        record = self._keyword_resolution(by_kind, gen3_sidecars)
        name = acting_line(record, gen3_sidecars).keyword_display_name
        event = record.payload.events[0].type.value
        found = FAMILY.match(record, gen3_sidecars, {"event": event, "keyword": name})
        assert found.polarity is Polarity.FIRES and found.observed
        spelled = name.lower().replace(" ", "_")
        assert FAMILY.match(record, gen3_sidecars, {"event": event, "keyword": spelled})
        assert FAMILY.match(record, gen3_sidecars,
                            {"event": event, "keyword": "no such keyword"}) is None

    def test_the_keyword_scopes_the_near_miss_as_api_does(self, by_kind, gen3_sidecars):
        record = self._keyword_resolution(by_kind, gen3_sidecars)
        name = acting_line(record, gen3_sidecars).keyword_display_name
        event = record.payload.events[0].type
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not event)))
        found = FAMILY.match(edited, gen3_sidecars, {"event": event.value, "keyword": name})
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        absent = FAMILY.match(edited, gen3_sidecars,
                              {"event": event.value, "keyword": name, "absent": True})
        assert absent.polarity is Polarity.FIRES

    def test_a_script_line_is_no_keyword_line(self, by_kind, gen3_sidecars):
        record = _token_maker(by_kind, gen3_sidecars)
        assert FAMILY.match(record, gen3_sidecars,
                            {"event": "token_created", "keyword": "Equip"}) is None

    def test_beside_an_api_a_keyword_admits_its_own_line_too(self, by_kind, gen3_sidecars):
        keyworded = self._keyword_resolution(by_kind, gen3_sidecars)
        name = acting_line(keyworded, gen3_sidecars).keyword_display_name
        scripted = _token_maker(by_kind, gen3_sidecars)
        api = acting_api(scripted, gen3_sidecars)
        for record in (keyworded, scripted):
            event = record.payload.events[0].type.value
            assert FAMILY.match(record, gen3_sidecars,
                                {"event": event, "api": api, "keyword": name}) is not None
        event = scripted.payload.events[0].type.value
        assert FAMILY.match(scripted, gen3_sidecars, {"event": event, "api": "NoSuchApi",
                                                      "keyword": name}) is None


class TestScriptParam:
    @staticmethod
    def _with_param(by_kind, sidecars):
        for record in effect_halves(by_kind):
            found = acting_root_params(record, sidecars) or {}
            name = next((k for k in found if k not in ("SP", "AB", "DB") and found[k]), None)
            if record.payload.events and name:
                return record, name, found[name], record.payload.events[0].type.value
        raise AssertionError("no effect half acting through a script with a parameter")

    def test_a_parameter_present_or_holding_a_value_narrows_the_line(
            self, by_kind, gen3_sidecars):
        record, name, value, event = self._with_param(by_kind, gen3_sidecars)
        for spec in (name, {name: "*"}, {name: value.upper()}):
            assert FAMILY.match(record, gen3_sidecars,
                                {"event": event, "script_param": spec}).observed
        for spec in ("NoSuchParam", {name: value + "-not"}):
            assert FAMILY.match(record, gen3_sidecars,
                                {"event": event, "script_param": spec}) is None

    def test_a_malformed_parameter_is_refused(self, by_kind, gen3_sidecars):
        record, _, _, event = self._with_param(by_kind, gen3_sidecars)
        with pytest.raises(ValueError, match="script_param"):
            FAMILY.validate({"event": event, "script_param": ["a"]})


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


def _chained_apis(record, sidecars):
    """The root API and the first other API of the acting line's chain, or None."""
    from effects.domain.script_text import api_of, params, segments

    line = acting_line(record, sidecars)
    if not record.payload.events or line is None:
        return None
    if not line.script_text:
        return None
    apis = [api_of(params(s)) for s in segments(line.script_text)]
    sub = next((a for a in apis[1:] if a and a != apis[0]), None)
    return (apis[0], sub) if apis[0] and sub else None


class TestApiInChain:
    @staticmethod
    def _chained(by_kind, sidecars):
        """An effect half whose acting line chains a sub-ability of another API."""
        for record in effect_halves(by_kind):
            chain = _chained_apis(record, sidecars)
            if chain is not None:
                return record, *chain, record.payload.events[0].type.value
        raise AssertionError("no effect half acting through a chained script")

    def test_a_sub_ability_s_api_is_read_only_when_asked_for(self, by_kind, gen3_sidecars):
        record, _, sub, event = self._chained(by_kind, gen3_sidecars)
        assert FAMILY.match(record, gen3_sidecars, {"event": event, "api": sub}) is None
        found = FAMILY.match(record, gen3_sidecars,
                             {"event": event, "api": sub, "api_in_chain": True})
        assert found.polarity is Polarity.FIRES

    def test_the_root_api_still_matches_in_the_chain(self, by_kind, gen3_sidecars):
        record, root, _, event = self._chained(by_kind, gen3_sidecars)
        for declared in ({"event": event, "api": root},
                         {"event": event, "api": root, "api_in_chain": True}):
            assert FAMILY.match(record, gen3_sidecars, declared).polarity is Polarity.FIRES
        assert FAMILY.match(record, gen3_sidecars, {"event": event, "api": "NoSuchApi",
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


class TestChainParam:
    """A parameter on any segment of the acting line's chain."""

    def test_a_sub_ability_parameter_narrows_the_line(self, by_kind, gen3_sidecars):
        from effects.domain.script_text import params, segments

        record, below = next(
            (r, name) for r in effect_halves(by_kind) if r.payload.events
            for line in [acting_line(r, gen3_sidecars)]
            if line is not None and line.script_text and len(segments(line.script_text)) > 1
            for name in params(segments(line.script_text)[1])
            if name not in (acting_root_params(r, gen3_sidecars) or {}))
        event = record.payload.events[0].type.value
        assert FAMILY.match(record, gen3_sidecars, {"event": event, "script_param": below}) is None
        assert FAMILY.match(record, gen3_sidecars,
                            {"event": event, "chain_param": below}).polarity is Polarity.FIRES
        assert FAMILY.match(record, gen3_sidecars,
                            {"event": event, "chain_param": "NoSuchParam"}) is None
