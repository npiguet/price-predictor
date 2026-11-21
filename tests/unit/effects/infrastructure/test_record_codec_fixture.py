"""Record codec: every record of the gen-3 fixture reads, writes back as
the collector wrote it and holds what the stories ask of it."""

from __future__ import annotations

import json

from effects.domain.event_schema import EventType
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import (
    ConditionMet,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityCheck,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
    TurnActionSubkind,
)
from effects.domain.state_snapshot import (
    normalize_counter_type,
)
from effects.infrastructure.record_codec import (
    record_from_dict,
    record_to_dict,
)
from tests.unit.effects.builders import (
    GEN3_RECORDS,
    gen3_fixture_records,
)


def _spelling_free(value):
    """A record's JSON less the spellings the reader normalizes by design: an
    entity's overlay keywords, an event's counter type and a contribution's
    keywords (gen-3 writes a keyword's instance, ``ward:1``)."""
    if isinstance(value, dict):
        return {k: (None if k == "granted_temporary"
                    else normalize_counter_type(v) if k == "counter_type"
                    else [normalize_keyword(x) for x in v]
                    if k == "keywords" and "entity" in value
                    else _spelling_free(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_spelling_free(v) for v in value]
    return value


class TestTheGen3Fixture:
    """The gen-3 fixture cut from the T079 pilot: every record reads, writes back
    as the collector wrote it and carries the payload members story 2 asks of it
    (T074 on real records, T079a)."""

    @staticmethod
    def _records() -> list[EffectRecord]:
        return [record_from_dict(raw) for raw in gen3_fixture_records()]

    def test_every_record_reads_as_gen3(self):
        records = self._records()
        assert len(records) > 100
        assert {record.generation for record in records} == {3}
        assert all(entity.lines is not None
                   for record in records for entity in record.state.entities)
        assert all(record.state.links is not None for record in records)

    def test_it_is_the_fixture_its_constants_describe(self):
        """A regeneration moves the digest, the counts and every count a test
        pins on the fixture together (``builders.GEN3_FIXTURE_*``)."""
        import gzip
        import hashlib

        from tests.unit.effects.builders import (
            GEN3_FIXTURE_GAMES,
            GEN3_FIXTURE_RECORDS,
            GEN3_FIXTURE_SHA256,
            GEN3_RECORDS,
        )

        assert hashlib.sha256(gzip.decompress(GEN3_RECORDS.read_bytes())).hexdigest() == (
            GEN3_FIXTURE_SHA256)
        raws = gen3_fixture_records()
        assert len(raws) == GEN3_FIXTURE_RECORDS
        assert len({raw["game_id"] for raw in raws}) == GEN3_FIXTURE_GAMES

    def test_every_game_ends_in_its_marker(self):
        """Cut from games that ended: each game's last record is its one
        game-end marker, as every reader needs to keep the game."""
        last: dict[str, EffectRecord] = {}
        markers: dict[str, int] = {}
        for record in self._records():
            last[record.game_id] = record
            if record.subkind is TurnActionSubkind.GAME_END:
                markers[record.game_id] = markers.get(record.game_id, 0) + 1
        assert markers == dict.fromkeys(last, 1)
        assert all(r.subkind is TurnActionSubkind.GAME_END for r in last.values())

    def test_the_shard_passes_the_generation_check(self):
        from effects.infrastructure.shard_io import require_generation

        require_generation([GEN3_RECORDS])

    def test_every_record_writes_back_as_the_collector_wrote_it(self):
        for raw in gen3_fixture_records():
            written = record_to_dict(record_from_dict(raw))
            assert json.dumps(_spelling_free(written)) == json.dumps(
                _spelling_free(raw)), (
                raw["record_id"])

    def test_it_holds_a_random_seat_game_and_a_forge_ai_game(self):
        records = self._records()
        seat_games = {r.game_id for r in records if r.random_seat}
        assert seat_games
        assert {r.game_id for r in records} - seat_games

    def test_every_turn_action_subkind_reads_with_its_events(self):
        actions = [r for r in self._records() if r.kind is RecordKind.TURN_ACTION]
        assert {r.subkind for r in actions} == set(TurnActionSubkind)
        assert all(r.ability is None for r in actions)
        # The game-end marker alone holds no event (WP7): its payload is
        # ``{events: [], records_withheld}``.
        _assert_every_played_action_holds_events(actions)
        _assert_every_game_end_marker_holds_none(actions)
        _assert_every_draw_draws_and_moves(actions)

    def test_a_sorcery_off_turn_fails_its_casting_speed(self):
        """Story 2 acceptance 8: timing false, can_play false, casting_speed failed."""
        found = [c for r in self._records()
                 if isinstance(r.payload, PlayabilityDecisionPayload)
                 and r.state.global_.active != r.actor_player
                 for c in r.payload.candidates
                 if c.timing is False and c.can_play is False
                 and any(f.check == PlayabilityCheck.CASTING_SPEED for f in c.failed_checks)]
        assert found

    def test_both_seats_have_candidates(self):
        """Story 2 acceptance 8: the player without priority is evaluated too."""
        decisions = [r for r in self._records()
                     if isinstance(r.payload, PlayabilityDecisionPayload)]
        assert any(r.actor_player != r.state.global_.priority for r in decisions)
        assert any(r.actor_player == r.state.global_.priority for r in decisions)

    def test_every_failed_check_reads(self):
        checks = {f.check for r in self._records()
                  if isinstance(r.payload, PlayabilityDecisionPayload)
                  for c in r.payload.candidates for f in c.failed_checks}
        assert checks == set(PlayabilityCheck)

    def test_an_unmet_condition_names_its_segment(self):
        """Story 2 acceptance 9."""
        unmet = [m for r in self._records() if isinstance(r.payload, ResolutionPayload)
                 for m in r.payload.conditions_met if m.met is False]
        assert unmet and all(isinstance(m, ConditionMet) and m.segment for m in unmet)

    def test_attackers_are_anchored_and_vigilance_keeps_its_anchor_untapped(self):
        """Story 2 acceptance 11."""
        records = [r for r in self._records()
                   if isinstance(r.payload, PlayabilityAttackersPayload)]
        assert records and all(r.payload.anchor_attacker for r in records)
        declared = _vigilance_and_tapping_of_declared_anchors(records)
        assert set(declared.values()) == {(False, True), (True, False)}
        assert any(r.payload.legal_defenders == () for r in records)
        assert any(r.payload.must_attack for r in records)

    def test_a_trigger_s_record_holds_no_death_of_the_check_it_arrived_in(self):
        """SC-004 on the burn kills the fixture holds: the dies trigger's
        resolution, when present, holds no death of a creature its snapshot
        no longer shows on the battlefield."""
        for r in filter(_is_a_live_trigger_resolution, self._records()):
            zones = {e.id: e.zone for e in r.state.entities}
            for event in filter(_is_an_unresolved_death, r.payload.events):
                assert all(zones.get(s, "battlefield") == "battlefield"
                           for s in event.subjects), r.record_id


def _assert_every_played_action_holds_events(actions: list[EffectRecord]) -> None:
    played = [r for r in actions if r.subkind is not TurnActionSubkind.GAME_END]
    assert all(r.payload.events and r.payload.records_withheld is None for r in played)


def _assert_every_game_end_marker_holds_none(actions: list[EffectRecord]) -> None:
    markers = [r for r in actions if r.subkind is TurnActionSubkind.GAME_END]
    assert all(r.payload.events == () and r.payload.records_withheld is not None
               for r in markers)


def _assert_every_draw_draws_and_moves(actions: list[EffectRecord]) -> None:
    draws = [r for r in actions if r.subkind is TurnActionSubkind.DRAW]
    assert all({EventType.CARD_DRAWN, EventType.ZONE_CHANGE}
               <= {e.type for e in r.payload.events} for r in draws)


def _vigilance_and_tapping_of_declared_anchors(
        records: list[EffectRecord]) -> dict[str, tuple[bool, bool]]:
    """Per attackers record that declared its anchor (``tapped_by_attack``
    set): whether the anchor has vigilance, and whether the attack tapped it."""
    declared = {}
    for r in records:
        if r.payload.tapped_by_attack is None:
            continue
        declared[r.record_id] = (_anchor_is_vigilant(r), r.payload.tapped_by_attack)
    return declared


def _anchor_is_vigilant(record: EffectRecord) -> bool:
    anchor = next(e for e in record.state.entities if e.id == record.payload.anchor_attacker)
    return any(line.keyword == "vigilance" for line in anchor.keyword_lines)


def _is_a_live_trigger_resolution(record: EffectRecord) -> bool:
    """A trigger's resolution half, not a fork's."""
    if not isinstance(record.payload, ResolutionPayload):
        return False
    if record.moment is not Moment.RESOLUTION:
        return False
    return _resolves_a_trigger(record) and not record.fork


def _resolves_a_trigger(record: EffectRecord) -> bool:
    return bool(record.ability) and record.ability[0].trait_kind == "trigger"


def _is_an_unresolved_death(event) -> bool:
    """A move from the battlefield to the graveyard no resolution caused."""
    if event.type is not EventType.ZONE_CHANGE:
        return False
    moved = (event.params.get("from_zone"), event.params.get("to_zone"))
    return moved == ("battlefield", "graveyard") and event.attributed_to == "unresolved"
