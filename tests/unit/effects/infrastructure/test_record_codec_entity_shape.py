"""Record codec: the gen-3 entity shape and payload members as the Java
collector writes them, read off the goldens and set one by one."""

from __future__ import annotations

import json
from dataclasses import replace
from enum import Enum

from effects.domain.event_schema import Event, EventType
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    ConditionMet,
    FailedCheck,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    TurnActionSubkind,
)
from effects.domain.state_snapshot import (
    CardChoices,
    CardHistory,
    CardLine,
    CommandZoneObject,
    CostsPaid,
    Dungeon,
    EntityState,
    LineRef,
    Link,
    PlayerState,
    PowerToughness,
    RandomResults,
    RealFace,
    Refs,
    TurnCounts,
    normalize_counter_type,
)
from effects.infrastructure.record_codec import (
    format_record_line,
    record_from_dict,
    record_to_dict,
)
from tests.unit.effects.builders import (
    _ENVELOPE_ONLY_GOLDENS,
    JAVA_GOLDENS,
    gen3_fixture_records,
    java_golden_records,
)
from tests.unit.effects.infrastructure.record_codec_support import (
    _record,
)


def _reader_spellings(value):
    """A collector's JSON with exactly the spellings the reader normalizes:
    the overlay keywords of an entity and every counter type."""
    if isinstance(value, dict):
        return {k: _member_reader_spelling(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_reader_spellings(v) for v in value]
    return value


def _member_reader_spelling(k, v):
    if k == "granted_temporary" and isinstance(v, dict):
        return {**v, "keywords": [normalize_keyword(x) for x in v["keywords"]]}
    if k == "counter_type" and isinstance(v, str):
        return normalize_counter_type(v)
    if k == "counters" and isinstance(v, dict):
        return {normalize_counter_type(name): count for name, count in v.items()}
    return _reader_spellings(v)


class TestTheGen3EntityShape:
    """``lines``, ``base_pt`` and ``pt_boosts`` as the Java collector writes them (T074)."""

    def test_every_golden_record_reads_as_gen3(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        assert len(records) > 20
        assert {record.generation for record in records} == {3}
        assert all(entity.lines is not None
                   for record in records for entity in record.state.entities)

    def test_a_golden_record_writes_back_as_the_collector_wrote_it(self):
        """Every member, in the collector's order, with the collector's values:
        nothing gen-3 writes is dropped or changed on a read and a write
        (T074). The written line is the Java line byte for byte, but for the
        two spellings the reader normalizes by design: an overlay keyword
        (``protection from green``) and a counter type (``+1/+1``)."""
        lines = [line for path in sorted(JAVA_GOLDENS.glob("*.jsonl"))
                 if path.name not in _ENVELOPE_ONLY_GOLDENS
                 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) > 40
        renormalized = 0
        for line in lines:
            raw = json.loads(line)
            written = format_record_line(record_from_dict(raw))
            expected = _reader_spellings(raw)
            if expected == raw:
                assert written == line, raw["record_id"]
            else:
                renormalized += 1
                assert json.loads(written) == expected, raw["record_id"]
                assert list(json.loads(written)) == list(raw), raw["record_id"]
        assert 0 < renormalized < len(lines)

    def test_the_gen3_blocks_read_into_their_fields(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        states = [record.state for record in records]
        _assert_the_links_read(states)
        _assert_the_players_and_global_blocks_read(states)
        _assert_the_entity_blocks_read(states)

    def test_the_gen3_payload_members_read_into_their_fields(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        _assert_every_resolution_half_meets_no_condition(records)
        _assert_every_candidate_is_timely_and_no_land_play(records)
        # The live goldens' attackers; a fork played on to the turn's end
        # writes its own (a summoning-sick creature that may not attack).
        _assert_the_live_attackers_read(records)
        _assert_every_forked_attackers_record_has_none_legal(records)
        _assert_every_blockers_record_must_block_none(records)

    def test_every_golden_payload_member_reads_into_the_field_of_its_name(self):
        """Per kind, each payload member the collector wrote is the value of
        the field of that name, read off the decoded record rather than written
        back: a member decoded into the wrong field and written back from it
        again would pass the write-back test. Events compare by their parts;
        members holding records of their own (costs, candidates, forbidden
        entries, contributions) are left to their own tests."""
        checked: set[tuple[str, str]] = set()
        for raw in java_golden_records():
            checked |= _checked_payload_members(raw)
        assert {("resolution", "outcome"), ("resolution", "conditions_met"),
                ("resolution", "events"), ("trigger", "fired"), ("trigger", "event"),
                ("combat", "blocks"), ("combat", "assignment_choices"),
                ("rewrite", "result"), ("rewrite", "incoming"), ("rewrite", "replaced_by"),
                ("playability", "legal_blockers"), ("playability", "min_blockers"),
                ("continuous", "board_hash"), ("turn_action", "records_withheld")} <= checked

    def test_every_gen3_member_round_trips_where_set(self):
        """The members the goldens leave at their empty value, each set."""
        key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
        records = _records_with_every_gen3_member_set(key)
        for record in records:
            assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record
        assert list(record_to_dict(records[2])["payload"]) == [
            "anchor_attacker", "legal_defenders", "must_attack", "tapped_by_attack",
            "legal_attackers", "forbidden"]
        assert list(record_to_dict(records[3])["payload"]) == [
            "anchor_attacker", "legal_blockers", "forbidden", "must_block", "min_blockers"]

    def test_a_player_s_land_plays_read_and_write_right_after_this_turn(self):
        """1 without a raise in every golden; null is unlimited, which an
        absent member (a record collected before it) is not (T074a)."""
        records = [record_from_dict(raw) for raw in java_golden_records()]
        players = [p for r in records for p in r.state.players]
        assert players and all(p.max_land_plays == 1 and not p.land_plays_unlimited
                               for p in players)
        raw = java_golden_records()[0]
        members = list(raw["state"]["players"][0])
        assert members[members.index("this_turn") + 1] == "max_land_plays"

        unlimited = json.loads(json.dumps(raw))
        unlimited["state"]["players"][0]["max_land_plays"] = None
        read = record_from_dict(unlimited).state.players[0]
        assert (read.max_land_plays, read.land_plays_unlimited) == (None, True)
        written = record_to_dict(record_from_dict(unlimited))["state"]["players"][0]
        assert "max_land_plays" in written and written["max_land_plays"] is None

    def test_an_attackers_record_carries_no_what_if(self):
        attackers = [raw for raw in java_golden_records()
                     if raw["kind"] == "playability" and raw["subkind"] == "attackers"]
        assert attackers
        assert not any("what_if" in raw for raw in attackers)

    def test_a_non_creature_writes_null_power_and_toughness(self):
        data = _entity_dict(EntityState(id="E1", name="Bolt", zone="stack",
                                        controller="P0", lines=()))
        assert (data["base_pt"], data["pt_boosts"]) == (None, None)
        assert not {"pt", "printed", "granted_attached"} & data.keys()

    def test_every_optional_line_member_round_trips_and_is_written_only_when_set(self):
        key = ProvenanceKey("cardsfolder/k/kraken.txt", 0, "keyword", 0)
        lines = (
            CardLine(keys=(key,), kind="keyword", granted=False, cast_with=2,
                     fragment="Multikicker:1 U"),
            CardLine(keys=(), kind="static", granted=True, active=False),
            CardLine(keys=(key,), kind="spell", granted=False, chosen=(0, 2)),
        )
        entity = EntityState(
            id="E1", name="A", zone="battlefield", controller="P0", lines=lines,
            pt=PowerToughness(base=(5, 5), boosts=(1, 0), counters=(2, 2)),
            counters={"P1P1": 2, "LOYALTY": 3},
        )
        data = _entity_dict(entity)
        assert data["base_pt"] == [5, 5]
        assert data["pt_boosts"] == {"temporary": [1, 0], "counters": [2, 2]}
        assert [sorted(line) for line in data["lines"]] == [
            ["cast_with", "fragment", "granted", "keys", "kind"],
            ["active", "granted", "keys", "kind"],
            ["chosen", "granted", "keys", "kind"],
        ]
        record = _record(state=replace(_record().state, entities=(entity,)))
        assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record

    def test_a_turn_action_record_reads_from_the_collector_s_shape(self):
        """No golden holds one yet: a golden envelope carrying a draw."""
        raw = dict(java_golden_records()[0])
        raw.update(kind="turn_action", moment=None, subkind="draw", ability=None,
                   link_id=None, mirror_of=None, fork=False, payload={"events": [{
                       "type": "zone_change", "subjects": ["E9"],
                       "params": {"from_zone": "library", "to_zone": "hand"},
                       "duration": "instant", "attributed_to": "unresolved"}]})
        record = record_from_dict(raw)
        assert (record.kind, record.subkind) == (RecordKind.TURN_ACTION, TurnActionSubkind.DRAW)
        assert [event.type for event in record.payload.events] == [EventType.ZONE_CHANGE]
        assert record_from_dict(record_to_dict(record)) == record


class TestALinesGrant:
    """A granted line's ``grant``, why it is granted (finding 1.7)."""

    @staticmethod
    def _with_a_granted_trigger(grant: str | None) -> tuple[dict, dict]:
        """A real fixture record with its first granted trigger line's
        ``grant`` set (left out on None), every other line's left out, and
        that line."""
        raw, line = next(
            (raw, line) for raw in gen3_fixture_records()
            for entity in raw["state"]["entities"] for line in entity.get("lines") or ()
            if line["granted"] and line["kind"] == "trigger")
        _drop_grants(raw)
        if grant is not None:
            line["grant"] = grant
        return raw, line

    def test_it_reads_and_writes_back_right_after_granted(self):
        raw, line = self._with_a_granted_trigger("static")
        written = record_to_dict(record_from_dict(raw))
        lines = [written_line for entity in written["state"]["entities"]
                 for written_line in entity.get("lines") or ()
                 if written_line.get("grant") is not None]
        assert lines == [line]
        assert list(lines[0])[:4] == ["keys", "kind", "granted", "grant"]

    def test_a_line_written_before_it_existed_reads_none(self):
        raw, line = self._with_a_granted_trigger(None)
        entities = record_from_dict(raw).state.entities
        assert [card_line.grant for entity in entities for card_line in entity.lines
                if card_line.granted and card_line.kind == "trigger"][:1] == [None]
        assert "grant" not in line


def _drop_grants(raw: dict) -> None:
    """Leave ``grant`` out of every line of ``raw``'s board."""
    for entity in raw["state"]["entities"]:
        for line in entity.get("lines") or ():
            line.pop("grant", None)


def _entity_dict(entity: EntityState) -> dict:
    record = _record(state=replace(_record().state, entities=(entity,)))
    return record_to_dict(record)["state"]["entities"][0]


def _assert_the_links_read(states) -> None:
    assert all(state.links is not None for state in states)
    assert any(_is_a_first_target_link(link) for state in states for link in state.links)


def _is_a_first_target_link(link: Link) -> bool:
    return link.type == "targets" and isinstance(link.from_, LineRef) and link.index == 0


def _assert_the_players_and_global_blocks_read(states) -> None:
    assert all(player.turn_counts is not None
               for state in states for player in state.players)
    assert any(made.source_lines for state in states
               for made in state.global_.floating_effects)
    assert any(state.refs.random == RandomResults() for state in states)


def _assert_the_entity_blocks_read(states) -> None:
    entities = [entity for state in states for entity in state.entities]
    assert all(_entity_blocks_read_empty(entity) for entity in entities)
    assert any(entity.mana_cost == {"generic": 1, "G": 1} for entity in entities)


def _entity_blocks_read_empty(entity: EntityState) -> bool:
    """Owned by a seat, with a history and no choice made."""
    return (entity.owner in ("P0", "P1") and entity.history is not None
            and entity.chosen == CardChoices())


def _payloads_of(records, payload_type) -> list:
    return [r.payload for r in records if isinstance(r.payload, payload_type)]


def _attackers_payloads(records, *, fork: bool) -> list:
    """The attackers payloads of forks, or of live records."""
    return [r.payload for r in records
            if isinstance(r.payload, PlayabilityAttackersPayload) and bool(r.fork) is fork]


def _assert_every_resolution_half_meets_no_condition(records) -> None:
    halves = _payloads_of(records, ResolutionPayload)
    assert halves and all(payload.conditions_met == () for payload in halves)


def _assert_every_candidate_is_timely_and_no_land_play(records) -> None:
    candidates = [c for payload in _payloads_of(records, PlayabilityDecisionPayload)
                  for c in payload.candidates]
    assert candidates and all(_is_timely_and_no_land_play(c) for c in candidates)


def _is_timely_and_no_land_play(candidate) -> bool:
    return (candidate.timing is True and candidate.failed_checks == ()
            and candidate.land_play is False)


def _assert_the_live_attackers_read(records) -> None:
    attackers = _attackers_payloads(records, fork=False)
    assert {(a.anchor_attacker, a.legal_defenders, a.must_attack, a.tapped_by_attack)
            for a in attackers} == {("E2", ("P1",), False, True), ("E3", (), False, None)}


def _assert_every_forked_attackers_record_has_none_legal(records) -> None:
    forked = _attackers_payloads(records, fork=True)
    assert forked and all(a.legal_attackers == () for a in forked)


def _assert_every_blockers_record_must_block_none(records) -> None:
    blockers = _payloads_of(records, PlayabilityBlockersPayload)
    assert blockers and all(b.must_block == () for b in blockers)


_NESTED_MEMBERS = frozenset({"costs", "candidates", "forbidden", "contributions"})


def _checked_payload_members(raw: dict) -> set[tuple[str, str]]:
    """Checks each payload member of one golden record against the decoded
    field of its name; the (kind, member) pairs checked."""
    payload = record_from_dict(raw).payload
    checked: set[tuple[str, str]] = set()
    for member, value in _reader_spellings(raw["payload"]).items():
        if member in _NESTED_MEMBERS:
            continue
        _check_payload_member(raw, payload, member, value)
        checked.add((raw["kind"], member))
    return checked


def _check_payload_member(raw: dict, payload, member: str, value) -> None:
    if member in ("replaced_by", "buffered"):
        value = [{k: v for k, v in key.items()} for key in value]
        assert [k.as_dict() for k in getattr(payload, member)] == value, (
            raw["record_id"], member)
        return
    value = _member_event_parts(member, value)
    assert _plain(getattr(payload, member)) == value, (raw["record_id"], member)


def _member_event_parts(member: str, value):
    """A member holding events, as the parts an event compares by."""
    if member in ("events", "buffered_events"):
        return [_event_parts(e) for e in value]
    if member in ("incoming", "outgoing", "event") and value is not None:
        return _event_parts(value)
    return value


def _plain(value):
    if isinstance(value, Event):
        return {"type": value.type.value, "subjects": list(value.subjects),
                "params": _plain(value.params), "duration": value.duration,
                "attributed_to": value.attributed_to}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _event_parts(raw: dict) -> dict:
    return {"type": raw["type"], "subjects": raw.get("subjects", []),
            "params": raw.get("params", {}), "duration": raw.get("duration"),
            "attributed_to": raw.get("attributed_to")}


def _an_entity_with_every_gen3_member_set(key: ProvenanceKey) -> EntityState:
    return EntityState(
        id="E3", name="Morph", zone="battlefield", controller="P1", lines=(
            CardLine(keys=(key,), kind="static", granted=False, active=True,
                     affected=("E1", "E2", "P0")),),
        owner="P0", face_state="face_down", mana_cost={"generic": 2, "W/U": 1},
        printed_loyalty=3, printed_defense=5,
        real_face=RealFace(name="Akroma", types=("creature",), subtypes=("angel",),
                           colors=("W",), mana_value=8, base_pt=(6, 6)),
        designations=("goaded", "monstrous"), x_paid=4,
        chosen=CardChoices(colors=("R",), type="Goblin", number=3, player="P1",
                           cards=("E9",), names=("Shock",), direction="left",
                           stored_rolls=(4, 6)),
        costs_paid=CostsPaid(mana={"W": 1, "U": 1}, cast_from="hand", mana_rider=True),
        history=CardHistory(entered_this_turn=True, loyalty_activated=True),
        order_rank=2, role="exiled_with",
    )


def _a_state_with_every_gen3_member_set(key: ProvenanceKey):
    entity = _an_entity_with_every_gen3_member_set(key)
    return replace(_record().state, entities=(entity,), links=(
        Link(from_=LineRef("E3", key), to="P1", type="targets", index=1),
        Link(from_="P0", to="E3", type="owns"),
        Link(from_=LineRef("E3", key), to="P0", type="affects"),
    ), global_=replace(_record().state.global_, day_night="night",
                       floating_effects=(CommandZoneObject(
                           id="F17", source_lines=(key,), affected=("E3",),
                           duration="end_of_turn", shield_left=2), CommandZoneObject(
                           id="E41", source_lines=(key,), affected=("E3", "P0"),
                           duration="end_of_turn")),
                       delayed_triggers=(CommandZoneObject(
                           id="D4", source_lines=(key,), remembered=("E3", "P1")),),
                       dungeons=(Dungeon(player="P0", dungeon="E40",
                                         room="Yawning Portal"),)),
        players=(PlayerState(id="P0", life=20, hand=7, library=40, graveyard=0,
                             max_land_plays=3,
                             speed=2, city_blessing=True, rad=3, monarch=True,
                             initiative=True, turn_counts=TurnCounts(cards_drawn=2,
                                                                     descended=1)),
                 PlayerState(id="P1", life=20, hand=7, library=40, graveyard=0,
                             land_plays_unlimited=True,
                             turn_counts=TurnCounts())),
        refs=Refs(targets_on_stack=(), random=RandomResults(dice=(4, 17), coin_flips_won=1,
                                                            coin_flips_lost=2)))


def _records_with_every_gen3_member_set(key: ProvenanceKey) -> list:
    """A resolution half, a decision, an attackers and a blockers record
    over one state with every gen-3 member set."""
    state = _a_state_with_every_gen3_member_set(key)
    return [
        _record(state=state, payload=ResolutionPayload(conditions_met=(
            ConditionMet("root", True), ConditionMet("1", False)))),
        _record(kind=RecordKind.PLAYABILITY, moment=None, ability=None, link_id=None,
                subkind=PlayabilitySubkind.DECISION, state=state,
                payload=PlayabilityDecisionPayload(candidates=(Candidate(
                    ability=(key,), can_play=False, affordable=True,
                    has_legal_target=True, responsible_static=(key,), timing=False,
                    failed_checks=(FailedCheck("casting_speed"),
                                   FailedCheck("static_prohibition", (key,))),
                    land_play=False), Candidate(
                    ability=(key,), can_play=False, affordable=True,
                    has_legal_target=True, timing=True,
                    failed_checks=(FailedCheck("land_play_limit"),),
                    land_play=True)))),
        _record(kind=RecordKind.PLAYABILITY, moment=None, ability=None, link_id=None,
                subkind=PlayabilitySubkind.ATTACKERS, state=state,
                payload=PlayabilityAttackersPayload(
                    legal_attackers=("E3",), anchor_attacker="E3",
                    legal_defenders=("P1", "E40"), must_attack=True,
                    tapped_by_attack=False)),
        _record(kind=RecordKind.PLAYABILITY, moment=None, ability=None, link_id=None,
                subkind=PlayabilitySubkind.BLOCKERS, state=state, what_if=False,
                payload=PlayabilityBlockersPayload(
                    anchor_attacker="E3", legal_blockers=("E5",), must_block=("E5",),
                    min_blockers=1)),
    ]
