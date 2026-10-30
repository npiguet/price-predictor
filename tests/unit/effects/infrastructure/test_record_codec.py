"""Record codec round-trips (T027).

The corpus cannot be rebuilt cheaply, so a serializer bug is unrecoverable
rather than inconvenient. Every record kind round-trips here, including the
stage-two-to-four kinds no collector writes yet — the schema is frozen before
collection precisely so those can be checked now.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    Costs,
    EffectRecord,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    PlayerState,
    PowerToughness,
    Refs,
    StackExtras,
    StateSnapshot,
)
from effects.infrastructure.record_codec import (
    format_record_line,
    parse_record_line,
    record_from_dict,
    record_to_dict,
)
from effects.infrastructure.record_io import (
    iter_shard_lines,
    read_records,
    read_shard,
    write_shard,
)

_KEY = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "trigger", 0)
_DONOR = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 1)


def _rich_snapshot() -> StateSnapshot:
    """A snapshot exercising every optional block, so nothing round-trips by luck."""
    return StateSnapshot(
        global_=GlobalState(
            turn=7, phase="combat_damage", active="P0", priority="P1",
            stack_size=2, combat_substep="first_strike", emblems=(_DONOR,),
        ),
        players=(
            PlayerState(
                id="P0", life=14, hand=3, library=21, graveyard=9, poison=1,
                energy=2, this_turn={"creatures_died": 1, "spells_cast": 2},
                floating_mana={"R": 1}, untapped_production={"G": 3},
            ),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
        entities=(
            EntityState(
                id="E12", name="Serra Angel", zone="battlefield", controller="P0",
                types=("Creature",), subtypes=("Angel",), colors=("W",),
                mana_value=5,
                pt=PowerToughness(base=(4, 4), boosts=(1, 0), counters=(0, 1)),
                tapped=True, damage=2, counters={"P1P1": 1},
                combat=CombatStatus(attacking="P1", blocked_by=("E20",),
                                    became_blocked=True),
                granted_attached=(_DONOR,),
                granted_temporary=GrantedTemporary(
                    keywords=("flying",), abilities=(_DONOR,),
                ),
                printed=(_KEY,),
            ),
            EntityState(
                id="S1", name="Fireball", zone="stack", controller="P1",
                stack_extras=StackExtras(
                    targets=("E12",), per_target_amounts={"E12": 3},
                    up_to_counts={"E12": 1},
                ),
            ),
        ),
        refs=Refs(
            targets=("E12",), source="S1", modes=("option-0",), x=3,
            choices={"color": "red"},
        ),
        tiers=frozenset({InclusionTier.REFERENCED, InclusionTier.CORE}),
    )


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run-uuid.4.10237",
        "run_id": "run-uuid",
        "timestamp": "2026-09-06T15:31:12.152018Z",
        "game_id": "run-uuid.4.812",
        "kind": RecordKind.RESOLUTION,
        "actor_player": "P0",
        "state": _rich_snapshot(),
        "moment": Moment.RESOLUTION,
        "ability": (_KEY,),
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("E12",),
                    params={"amount": 3, "combat": False}, attributed_to="0",
                ),
            )
        ),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


ALL_KINDS = {
    "resolution-activation": dict(
        moment=Moment.ACTIVATION,
        payload=ActivationPayload(
            costs=Costs(
                mana_by_color={"R": 1}, tapped=("E12",), life=2,
                sacrificed=("E13",), discarded=("E14",), exiled=("E15",),
            ),
            outcome=ResolutionOutcome.RESOLVED,
        ),
        link_id="pair-7",
    ),
    "resolution-effect": dict(link_id="pair-7"),
    "rewrite": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 5}),
            outgoing=Event(type=EventType.DAMAGE_DEALT, params={"amount": 1}),
            result=RewriteResult.UPDATED,
        ),
    ),
    # The shape most of the channel actually takes: Forge substitutes by
    # running another ability, so nothing in the event's map is edited.
    "rewrite-substituted": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.ZONE_CHANGE, subjects=("E12",),
                           params={"to_zone": "graveyard"}),
            result=RewriteResult.REPLACED,
            replaced_by=(
                ProvenanceKey(
                    script_file="cardsfolder/l/leyline_of_the_void.txt",
                    face=0, trait_kind="replacement", index_within_kind=0,
                ),
            ),
        ),
    ),
    "rewrite-declined": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 5}),
            result=RewriteResult.NOT_REPLACED,
        ),
    ),
    "continuous": dict(
        kind=RecordKind.CONTINUOUS, moment=None,
        payload=ContinuousPayload(
            contributions=(
                Contribution(
                    entity="E12", pt_boost=(1, 1), keywords=("vigilance",),
                    types=("Creature",), colors=("W",), name=None,
                ),
            ),
            board_hash="ab12cd",
        ),
    ),
    "combat": dict(
        kind=RecordKind.COMBAT, moment=None, ability=None,
        payload=CombatPayload(
            attackers=("E12",), blocks={"E12": ("E20",)},
            assignment_choices={"E12": {"E20": 4}},
            events=(Event(type=EventType.DAMAGE_DEALT,
                          params={"amount": 4, "combat": True}),),
        ),
    ),
    "trigger": dict(
        kind=RecordKind.TRIGGER, moment=None,
        payload=TriggerPayload(
            event=Event(type=EventType.LIFE_CHANGE, params={"delta": 3}),
            fired=True,
        ),
    ),
    "playability-decision": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.DECISION,
        payload=PlayabilityDecisionPayload(
            candidates=(
                Candidate(
                    ability=(_KEY,), can_play=True, affordable=False,
                    has_legal_target=True, legal_targets=("E20",),
                    cost_after_adjustment={"R": 2},
                    responsible_static=(_DONOR,),
                ),
            )
        ),
    ),
    "playability-attackers": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.ATTACKERS,
        payload=PlayabilityAttackersPayload(
            legal_attackers=("E12",),
            forbidden=(ForbiddenEntity(entity="E13", responsible_static=(_DONOR,)),),
        ),
    ),
    "playability-blockers": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.BLOCKERS,
        payload=PlayabilityBlockersPayload(
            anchor_attacker="E12", legal_blockers=("E20",),
            forbidden=(ForbiddenEntity(entity="E21"),), min_blockers=2,
        ),
    ),
}


class TestRoundTrip:
    @pytest.mark.parametrize("name", sorted(ALL_KINDS))
    def test_every_record_kind_round_trips(self, name: str):
        record = _record(**ALL_KINDS[name])
        assert record_from_dict(record_to_dict(record)) == record

    @pytest.mark.parametrize("name", sorted(ALL_KINDS))
    def test_every_record_kind_survives_a_line(self, name: str):
        record = _record(**ALL_KINDS[name])
        assert record_from_dict(json.loads(format_record_line(record))) == record

    def test_a_line_holds_no_newline_of_its_own(self):
        assert "\n" not in format_record_line(_record())

    def test_a_non_ascii_name_is_written_as_utf8_and_reads_back(self, tmp_path):
        """The writer is ``orjson``, which does not escape; the reader decodes
        a ``\\u`` escape and the raw character alike."""
        record = _record(state=replace(_rich_snapshot(), entities=(
            EntityState(id="E1", name="Lim-Dûl's Vault", zone="hand", controller="P0"),
        )))
        line = format_record_line(record)
        assert "Lim-Dûl's Vault" in line
        assert record_from_dict(json.loads(line)) == record
        shard = tmp_path / "run.0-a.jsonl.gz"
        write_shard(shard, [record])
        assert list(read_shard(shard)) == [record]

    def test_the_snapshot_survives_in_full(self):
        restored = record_from_dict(record_to_dict(_record()))
        assert restored.state == _rich_snapshot()

    def test_the_two_grant_channels_stay_distinct_across_the_wire(self):
        restored = record_from_dict(record_to_dict(_record()))
        entity = restored.state.entity("E12")
        assert entity.granted_attached == (_DONOR,)
        assert entity.granted_temporary.keywords == ("flying",)

    def test_collected_tiers_are_written_explicitly(self):
        """Without them, an empty entity list and an uncollected tier look alike."""
        data = record_to_dict(_record())
        assert data["state"]["tiers"] == [1, 2]

    def test_the_wire_field_for_the_global_block_is_named_global(self):
        assert "global" in record_to_dict(_record())["state"]


class TestForwardCompatibility:
    def test_an_unknown_envelope_field_is_preserved_on_read(self):
        data = record_to_dict(_record())
        data["collector_build"] = "2027.1"
        restored = record_from_dict(data)
        assert restored.extra_fields == {"collector_build": "2027.1"}

    def test_an_unknown_field_survives_a_rewrite(self):
        data = record_to_dict(_record())
        data["collector_build"] = "2027.1"
        assert record_to_dict(record_from_dict(data))["collector_build"] == "2027.1"

    def test_an_unknown_field_never_shadows_a_known_one(self):
        record = _record()
        restored = record_from_dict(record_to_dict(record))
        assert restored.extra_fields == {}

    def test_ability_unresolved_round_trips(self):
        record = _record(ability=(), ability_unresolved="engine_effect")
        assert record_from_dict(record_to_dict(record)) == record

    def test_a_shard_written_before_the_field_reads_as_no_reason(self):
        """The corpus is append-only, so a line without the key still parses."""
        data = record_to_dict(_record())
        del data["ability_unresolved"]
        restored = record_from_dict(data)
        assert restored.ability_unresolved is None
        assert restored.extra_fields == {}


class TestTheProbedPerturbation:
    """A damage-step probe records what it stripped, and from whom.

    The keyword alone is ambiguous on a board with two tramplers, and an
    evaluator reproducing the perturbation model-side would strip the wrong
    creature — turning the counterfactual the probe exists for into a
    comparison against a different one.
    """

    def _probe(self, **payload_kwargs) -> EffectRecord:
        return _record(
            kind=RecordKind.COMBAT, moment=None, ability=None,
            fork=True, mirror_of="run.0.7",
            payload=CombatPayload(
                attackers=("E12",),
                events=(Event(type=EventType.DAMAGE_DEALT,
                              params={"amount": 2, "combat": True}),),
                **payload_kwargs,
            ),
        )

    def test_both_halves_round_trip(self):
        probe = self._probe(probed_keyword="trample", probed_entity="E12")
        loaded = record_from_dict(record_to_dict(probe))
        assert loaded.payload.probed_keyword == "trample"
        assert loaded.payload.probed_entity == "E12"

    def test_an_observed_combat_record_names_neither(self):
        """Nothing was perturbed, so there is nothing to name."""
        observed = _record(
            kind=RecordKind.COMBAT, moment=None, ability=None,
            payload=CombatPayload(attackers=("E12",)),
        )
        loaded = record_from_dict(record_to_dict(observed))
        assert loaded.payload.probed_keyword is None
        assert loaded.payload.probed_entity is None

    def test_a_record_written_before_the_fields_existed_still_reads(self):
        """Compatibility rule 1: the corpus is append-only, and a shard
        collected before the probe existed cannot be regenerated."""
        wire = record_to_dict(
            _record(kind=RecordKind.COMBAT, moment=None, ability=None,
                    payload=CombatPayload(attackers=("E12",)))
        )
        del wire["payload"]["probed_keyword"]
        del wire["payload"]["probed_entity"]
        loaded = record_from_dict(wire)
        assert loaded.payload.probed_keyword is None
        assert loaded.payload.attackers == ("E12",)


class TestTheRewriteResult:
    """What the replacement did, and which ability did it.

    ``{incoming, outgoing}`` alone modelled an edit-the-event mechanism Forge
    does not have: it substitutes by running a *different* ability, so the two
    maps came back byte-identical and 87% of the channel was dropped as
    identity rewrites. These four fields are the whole of what those records
    were trying to say, so a serializer bug here is the channel going quiet
    again — and the corpus cannot be recollected.
    """

    def _rewrite(self, **payload_kwargs) -> EffectRecord:
        return _record(
            kind=RecordKind.REWRITE, moment=None,
            payload=RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT,
                               subjects=("P1",), params={"amount": 5}),
                **payload_kwargs,
            ),
        )

    def test_an_in_place_edit_round_trips_both_halves_and_its_result(self):
        """Orim's Cure taking 5 combat damage down to 1: a real rewrite."""
        record = self._rewrite(
            outgoing=Event(type=EventType.DAMAGE_DEALT, subjects=("P1",),
                           params={"amount": 1}),
            result=RewriteResult.UPDATED,
        )
        loaded = record_from_dict(record_to_dict(record))
        assert loaded == record
        assert loaded.payload.outgoing.params["amount"] == 1
        assert loaded.payload.result is RewriteResult.UPDATED

    def test_a_substitution_writes_null_rather_than_a_copy(self):
        """The rule the channel's readability rests on: a record whose two
        halves are identical says nothing, and null says "not rewritten"."""
        key = ProvenanceKey(
            script_file="cardsfolder/l/leyline_of_the_void.txt",
            face=0, trait_kind="replacement", index_within_kind=0,
        )
        record = self._rewrite(result=RewriteResult.REPLACED, replaced_by=(key,))
        wire = record_to_dict(record)
        assert wire["payload"]["outgoing"] is None
        assert wire["payload"]["result"] == "replaced"
        loaded = record_from_dict(wire)
        assert loaded.payload.outgoing is None
        assert loaded.payload.replaced_by == (key,)

    def test_a_declined_replacement_round_trips_as_a_negative(self):
        """Every executeReplacement call writes a record, this one included:
        the negatives are what teach when a replacement applies."""
        record = self._rewrite(result=RewriteResult.NOT_REPLACED)
        loaded = record_from_dict(record_to_dict(record))
        assert loaded == record
        assert loaded.payload.result is RewriteResult.NOT_REPLACED
        assert loaded.payload.replaced_by == ()

    @pytest.mark.parametrize("result", list(RewriteResult))
    def test_every_result_survives_the_wire_in_its_snake_case_spelling(
        self, result: RewriteResult,
    ):
        record = self._rewrite(result=result)
        assert record_to_dict(record)["payload"]["result"] == result.value
        assert result.value == result.value.lower()
        assert record_from_dict(record_to_dict(record)).payload.result is result


class TestEveryEventReadsOneCounterSpelling:
    """``+1/+1`` becomes ``P1P1`` on read wherever an event sits, not only in a
    resolution's event list: a rewrite's outgoing event is a target too, and a
    raw spelling there falls through to ``counters_delta_other``."""

    @staticmethod
    def _counter_event() -> Event:
        return Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                     params={"counter_type": "+1/+1", "delta": 1})

    def test_a_rewrites_events(self):
        record = _record(
            kind=RecordKind.REWRITE, moment=None,
            payload=RewritePayload(
                incoming=self._counter_event(), outgoing=self._counter_event(),
                result=RewriteResult.UPDATED,
            ),
        )
        loaded = record_from_dict(record_to_dict(record))
        assert loaded.payload.incoming.params["counter_type"] == "P1P1"
        assert loaded.payload.outgoing.params["counter_type"] == "P1P1"

    def test_a_triggers_event(self):
        record = _record(
            kind=RecordKind.TRIGGER, moment=None,
            payload=TriggerPayload(event=self._counter_event(), fired=True),
        )
        loaded = record_from_dict(record_to_dict(record))
        assert loaded.payload.event.params["counter_type"] == "P1P1"


class TestARewriteShardCollectedBeforeTheResult:
    """Both shapes coexist: the corpus is append-only.

    A shard written before the hook passed a result carries
    ``{incoming, outgoing}`` and nothing else, and its ``outgoing`` is usually
    a copy of its ``incoming``. It has to keep reading, and it has to keep
    reading as a *third* state -- not as ``not_replaced``, which would
    reinterpret every one of those records as a replacement the collector
    watched decline.
    """

    def _old_wire(self) -> dict:
        wire = record_to_dict(
            _record(
                kind=RecordKind.REWRITE, moment=None,
                payload=RewritePayload(
                    incoming=Event(type=EventType.DAMAGE_DEALT,
                                   params={"amount": 3}),
                    outgoing=Event(type=EventType.DAMAGE_DEALT,
                                   params={"amount": 3}),
                ),
            )
        )
        del wire["payload"]["result"]
        del wire["payload"]["replaced_by"]
        return wire

    def test_it_still_parses(self):
        loaded = record_from_dict(self._old_wire())
        assert loaded.payload.incoming.params["amount"] == 3
        assert loaded.payload.outgoing.params["amount"] == 3

    def test_its_missing_result_is_none_and_not_not_replaced(self):
        payload = record_from_dict(self._old_wire()).payload
        assert payload.result is None
        assert payload.result is not RewriteResult.NOT_REPLACED

    def test_it_names_no_substituted_ability_rather_than_failing(self):
        assert record_from_dict(self._old_wire()).payload.replaced_by == ()

    def test_rewriting_it_writes_the_new_keys_explicitly(self):
        """Reading an old line and writing it back yields the current shape
        with nulls, rather than silently reproducing the old one."""
        rewritten = record_to_dict(record_from_dict(self._old_wire()))
        assert rewritten["payload"]["result"] is None
        assert rewritten["payload"]["replaced_by"] == []
        assert rewritten["payload"]["outgoing"] == {
            **rewritten["payload"]["incoming"],
        }


_GEN1_FIXTURE = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-records.jsonl.gz"
)


class TestGen2EnvelopeFields:
    """``random_seat`` and ``what_if`` (FR-030, FR-031, FR-033)."""

    def test_every_gen1_record_reads_with_the_defaults(self):
        records = list(read_shard(_GEN1_FIXTURE))
        assert len(records) > 40
        for record in records:
            assert record.random_seat is None
            assert not record.random_seat
            assert record.what_if is None
            assert record.extra_fields == {}

    def test_a_rewritten_gen1_shard_stays_gen1(self, tmp_path):
        """A curated shard is gen-1 or gen-2 as its source was (FR-033).

        Written with ``random_seat`` on every record whatever the source held,
        a dataset built from gen-1 shards read as gen-2, and the mixed-
        generation guard on ``--corpus`` could never fire.
        """
        from effects.infrastructure.record_io import ShardGeneration, shard_generation, write_shard

        rewritten = tmp_path / "curated.jsonl.gz"
        write_shard(rewritten, read_shard(_GEN1_FIXTURE))
        assert shard_generation(rewritten) is ShardGeneration.GEN_1
        for line in iter_shard_lines(rewritten):
            assert "random_seat" not in json.loads(line)

    def test_a_gen1_line_carries_neither_field(self):
        for line in iter_shard_lines(_GEN1_FIXTURE):
            data = json.loads(line)
            assert "random_seat" not in data and "what_if" not in data

    def test_random_seat_round_trips(self):
        record = _record(random_seat=True)
        assert record_from_dict(record_to_dict(record)) == record

    def test_what_if_round_trips_on_a_legality_record(self):
        for value in (True, False):
            record = _record(**ALL_KINDS["playability-blockers"], what_if=value)
            data = record_to_dict(record)
            assert data["what_if"] is value
            assert record_from_dict(data) == record

    def test_what_if_is_written_only_when_set(self):
        data = record_to_dict(_record(**ALL_KINDS["playability-blockers"]))
        assert "what_if" not in data
        assert data["random_seat"] is False

    def test_random_seat_follows_synthetic_in_the_envelope(self):
        keys = list(record_to_dict(_record(**ALL_KINDS["playability-blockers"],
                                           what_if=False)))
        assert keys.index("random_seat") == keys.index("synthetic") + 1
        assert keys.index("what_if") == keys.index("random_seat") + 1


class TestInternedProvenanceKeys:
    """A shard repeats the same few hundred keys tens of thousands of times."""

    def test_an_equal_key_parses_to_the_same_object(self, tmp_path):
        directory = tmp_path / "records"
        directory.mkdir()
        (directory / _GEN1_FIXTURE.name).write_bytes(_GEN1_FIXTURE.read_bytes())
        seen: dict[ProvenanceKey, ProvenanceKey] = {}
        repeats = 0
        for record in read_records(directory):
            for entity in record.state.entities:
                for key in entity.printed:
                    first = seen.setdefault(key, key)
                    assert first is key
                    repeats += 1
        assert repeats > len(seen) > 1

    def test_an_option_keeps_the_mode_key_apart_from_its_root(self):
        from effects.infrastructure.record_codec import _keys_from_json

        root = {"script_file": "cardsfolder/c/charm.txt", "face": 0,
                "trait_kind": "spell", "index_within_kind": 0}
        (a, b, c) = _keys_from_json([root, {**root, "option": 1}, dict(root)])
        assert a is c
        assert a != b and b.option == 1 and b.root == a

    def test_interned_records_equal_freshly_built_ones(self):
        record = _record(**ALL_KINDS["resolution-activation"])
        data = record_to_dict(record)
        assert record_from_dict(data) == record
        assert record_from_dict(data) == record_from_dict(json.loads(json.dumps(data)))


class TestTheWireFormIsPinned:
    """The bytes a record serializes to, pinned against a digest.

    A round trip alone cannot catch a codec change that alters encoder and
    decoder together — a renamed key, a reordered field — and the corpus is
    append-only, so a shard written under the changed form would no longer
    read the way the shards beside it do. The digests were taken from the
    serializer before its codecs were regrouped.
    """

    @staticmethod
    def _digest(records) -> str:
        import hashlib

        digest = hashlib.sha256()
        for record in records:
            digest.update(format_record_line(record).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def test_the_gen1_fixture_serializes_to_the_pinned_bytes(self):
        """Without ``random_seat``, which a gen-1 line never carried."""
        records = list(read_shard(_GEN1_FIXTURE))
        assert len(records) == 53
        assert self._digest(records) == (
            "588245c598027912271025c937270892f62d31cfb9e8346c4d142ea00a5ea266"
        )

    def test_every_kind_serializes_to_the_pinned_bytes(self):
        records = [_record(**ALL_KINDS[name]) for name in sorted(ALL_KINDS)]
        assert self._digest(records) == (
            "c8736a4d14319f0512573a25c3b307f355d52b16062e6c2c8768c667c1161a83"
        )

    def test_the_fixture_round_trips_through_its_own_wire_form(self):
        for record in read_shard(_GEN1_FIXTURE):
            line = format_record_line(record)
            again = record_from_dict(parse_record_line(line))
            assert again == record
            assert format_record_line(again) == line


class TestThePayloadCodecs:
    """One codec per payload type, chosen by the record's ``(kind, discriminator)``."""

    def test_every_payload_type_has_exactly_one_codec(self):
        from effects.domain.records import PAYLOAD_TYPES
        from effects.infrastructure.record_codec import _PAYLOAD_CODECS

        assert set(_PAYLOAD_CODECS) == set(PAYLOAD_TYPES.values())

    @pytest.mark.parametrize("kind, discriminator", [
        (RecordKind.RESOLUTION, None),
        (RecordKind.PLAYABILITY, None),
        (RecordKind.REWRITE, Moment.RESOLUTION),
        (RecordKind.COMBAT, PlayabilitySubkind.BLOCKERS),
    ])
    def test_a_pair_no_payload_is_defined_for_is_refused(self, kind, discriminator):
        # The decoder used to read anything it did not recognise as a
        # blockers payload; an unknown pair is now refused by name.
        from effects.infrastructure.record_codec import _payload_from_json

        with pytest.raises(ValueError, match="no payload is defined"):
            _payload_from_json({"anchor_attacker": "E1"}, kind, discriminator)
