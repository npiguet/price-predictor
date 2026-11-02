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
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    CombatPayload,
    ConditionMet,
    ContinuousPayload,
    Contribution,
    Costs,
    EffectRecord,
    FailedCheck,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityCheck,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
    TurnActionSubkind,
)
from effects.domain.state_snapshot import (
    CardChoices,
    CardHistory,
    CardLine,
    CombatStatus,
    CommandZoneObject,
    CostsPaid,
    Dungeon,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    LineRef,
    Link,
    PlayerState,
    PowerToughness,
    RandomResults,
    RealFace,
    Refs,
    StackExtras,
    StateSnapshot,
    TurnCounts,
    normalize_counter_type,
)
from effects.infrastructure.record_codec import (
    format_record_line,
    parse_record_line,
    record_from_dict,
    record_to_dict,
)
from effects.infrastructure.shard_io import (
    iter_shard_lines,
    read_records,
    read_shard,
    write_shard,
)
from tests.unit.effects.builders import (
    gen3_fixture_records,
    java_golden_records,
    make_record,
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

    def test_a_gen3_observed_combat_record_is_written_without_them(self):
        """As the gen-3 collector writes it; an earlier generation keeps the
        nulls its pinned bytes hold."""
        observed = dict(kind=RecordKind.COMBAT, moment=None, ability=None,
                        payload=CombatPayload(attackers=("E12",)))
        gen3 = record_to_dict(_record(generation=3, **observed))
        assert not {"probed_keyword", "probed_entity"} & gen3["payload"].keys()
        gen2 = record_to_dict(_record(**observed))
        assert (gen2["payload"]["probed_keyword"], gen2["payload"]["probed_entity"]) == (
            None, None)

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
        from effects.domain.shard_generation import ShardGeneration
        from effects.infrastructure.shard_io import shard_generation, write_shard

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


class TestGen3EnvelopeFields:
    """``generation``, ``aimed_at``, ``directed`` and the ``turn_action`` kind (FR-018, T014)."""

    def test_generation_is_first_after_the_record_id(self):
        keys = list(record_to_dict(_record(generation=3)))
        assert keys[:2] == ["record_id", "generation"]

    def test_the_gen3_members_are_written_only_when_set(self):
        data = record_to_dict(_record())
        assert not {"generation", "aimed_at", "directed"} & data.keys()

    def test_a_gen3_envelope_round_trips(self):
        from effects.domain.fork_operation import ForkOperation
        from effects.domain.records import DirectedTag

        record = make_record(
            RecordKind.COMBAT, generation=3, fork=True, aimed_at=("704.5f",),
            directed=DirectedTag("tougher", ForkOperation.PT))
        data = record_to_dict(record)
        assert data["aimed_at"] == ["704.5f"]
        assert data["directed"] == {"variant": "tougher", "operation": "pt"}
        assert record_from_dict(json.loads(json.dumps(data))) == record

    def test_an_empty_aimed_at_is_kept_apart_from_none(self):
        record = _record(generation=3, aimed_at=())
        data = record_to_dict(record)
        assert data["aimed_at"] == []
        assert record_from_dict(data).aimed_at == ()

    @pytest.mark.parametrize("subkind", list(TurnActionSubkind))
    def test_a_turn_action_record_round_trips(self, subkind):
        record = make_record(RecordKind.TURN_ACTION, subkind=subkind, generation=3)
        data = record_to_dict(record)
        assert (data["kind"], data["subkind"]) == ("turn_action", subkind.value)
        assert record_from_dict(data) == record

    def test_a_gen3_shard_reads_as_gen3(self, tmp_path):
        from effects.domain.shard_generation import ShardGeneration
        from effects.infrastructure.shard_io import shard_generation, write_shard

        path = tmp_path / "g3.jsonl.gz"
        write_shard(path, [replace(r, generation=3) for r in read_shard(_GEN1_FIXTURE)])
        assert shard_generation(path) is ShardGeneration.GEN_3


_GEN2_FIXTURE = _GEN1_FIXTURE.with_name("gen2-records.jsonl.gz")


class TestTheGen2Fixture:
    """The gen-2 fixture cut from real shards reads whole and stays gen-2 (T002, T014)."""

    def test_every_record_reads_and_carries_random_seat(self):
        records = list(read_shard(_GEN2_FIXTURE))
        assert len(records) > 40
        assert all(record.random_seat is not None for record in records)
        assert any(record.random_seat for record in records)
        assert all(record.generation is None for record in records)

    def test_the_shard_is_gen2(self):
        from effects.domain.shard_generation import ShardGeneration
        from effects.infrastructure.shard_io import shard_generation

        assert shard_generation(_GEN2_FIXTURE) is ShardGeneration.GEN_2

    def test_a_parsed_record_survives_a_write_and_a_read(self):
        """Not the raw line: parsing normalizes keyword spellings by design."""
        for record in read_shard(_GEN2_FIXTURE):
            assert record_from_dict(record_to_dict(record)) == record

    def test_every_entity_reads_and_writes_in_the_gen2_shape(self):
        """Gen-2 entities keep reading exactly as before (T074): no lines, and
        written back with the members they were read with, in their order."""
        for line in iter_shard_lines(_GEN2_FIXTURE):
            raw = parse_record_line(line)
            written = record_to_dict(record_from_dict(raw))
            for before, after in zip(raw["state"]["entities"], written["state"]["entities"],
                                     strict=True):
                assert list(after) == list(before)
                assert {k: v for k, v in after.items() if k != "granted_temporary"} == {
                    k: v for k, v in before.items() if k != "granted_temporary"}
        assert all(entity.lines is None for record in read_shard(_GEN2_FIXTURE)
                   for entity in record.state.entities)

    def test_every_gen2_record_writes_back_as_it_was_read(self):
        """Every member in its order, no gen-3 member added: gen-2 shards stay
        gen-2 through a read and a write (T074). An observed combat gains the
        null probe members this codec has always written for it."""
        for line in iter_shard_lines(_GEN2_FIXTURE):
            raw = parse_record_line(line)
            written = record_to_dict(record_from_dict(raw))
            if raw["kind"] == "combat" and not raw["fork"]:
                assert written["payload"].pop("probed_keyword") is None
                assert written["payload"].pop("probed_entity") is None
            assert json.dumps(_spelling_free(written)) == json.dumps(_spelling_free(raw)), (
                raw["record_id"])


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


class TestTheGen3EntityShape:
    """``lines``, ``base_pt`` and ``pt_boosts`` as the Java collector writes them (T074)."""

    def test_every_golden_record_reads_as_gen3(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        assert len(records) > 20
        assert {record.generation for record in records} == {3}
        assert all(entity.lines is not None
                   for record in records for entity in record.state.entities)

    def test_a_golden_record_writes_back_as_the_collector_wrote_it(self):
        """Every member, in the collector's order: nothing gen-3 writes is
        dropped on a read and a write (T074)."""
        for raw in java_golden_records():
            written = record_to_dict(record_from_dict(raw))
            assert json.dumps(_spelling_free(written)) == json.dumps(_spelling_free(raw)), (
                raw["record_id"])

    def test_the_gen3_blocks_read_into_their_fields(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        states = [record.state for record in records]
        assert all(state.links is not None for state in states)
        assert any(link.type == "targets" and isinstance(link.from_, LineRef)
                   and link.index == 0 for state in states for link in state.links)
        assert all(player.turn_counts is not None
                   for state in states for player in state.players)
        assert any(made.source_lines for state in states
                   for made in state.global_.floating_effects)
        assert any(state.refs.random == RandomResults() for state in states)
        entities = [entity for state in states for entity in state.entities]
        assert all(entity.owner in ("P0", "P1") and entity.history is not None
                   and entity.chosen == CardChoices() for entity in entities)
        assert any(entity.mana_cost == {"generic": 1, "G": 1} for entity in entities)

    def test_the_gen3_payload_members_read_into_their_fields(self):
        records = [record_from_dict(raw) for raw in java_golden_records()]
        halves = [r.payload for r in records if isinstance(r.payload, ResolutionPayload)]
        assert halves and all(payload.conditions_met == () for payload in halves)
        candidates = [c for r in records if isinstance(r.payload, PlayabilityDecisionPayload)
                      for c in r.payload.candidates]
        assert candidates and all(c.timing is True and c.failed_checks == ()
                                  for c in candidates)
        attackers = [r.payload for r in records
                     if isinstance(r.payload, PlayabilityAttackersPayload)]
        assert {(a.anchor_attacker, a.legal_defenders, a.must_attack, a.tapped_by_attack)
                for a in attackers} == {("E2", ("P1",), False, True), ("E3", (), False, None)}
        blockers = [r.payload for r in records
                    if isinstance(r.payload, PlayabilityBlockersPayload)]
        assert blockers and all(b.must_block == () for b in blockers)

    def test_every_gen3_member_round_trips_where_set(self):
        """The members the goldens leave at their empty value, each set."""
        key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
        entity = EntityState(
            id="E3", name="Morph", zone="battlefield", controller="P1", lines=(
                CardLine(keys=(key,), kind="static", granted=False, active=True,
                         affected=("E1", "E2")),),
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
        state = replace(_record().state, entities=(entity,), links=(
            Link(from_=LineRef("E3", key), to="P1", type="targets", index=1),
            Link(from_="P0", to="E3", type="owns"),
        ), global_=replace(_record().state.global_, day_night="night",
                           floating_effects=(CommandZoneObject(
                               id="F17", source_lines=(key,), affected=("E3",),
                               duration="end_of_turn", shield_left=2),),
                           delayed_triggers=(CommandZoneObject(
                               id="D4", source_lines=(key,), remembered=("E3", "P1")),),
                           dungeons=(Dungeon(player="P0", dungeon="E40",
                                             room="Yawning Portal"),)),
            players=(PlayerState(id="P0", life=20, hand=7, library=40, graveyard=0,
                                 speed=2, city_blessing=True, rad=3, monarch=True,
                                 initiative=True, turn_counts=TurnCounts(cards_drawn=2,
                                                                         descended=1)),),
            refs=Refs(random=RandomResults(dice=(4, 17), coin_flips_won=1,
                                           coin_flips_lost=2)))
        records = [
            _record(state=state, payload=ResolutionPayload(conditions_met=(
                ConditionMet("root", True), ConditionMet("1", False)))),
            _record(kind=RecordKind.PLAYABILITY, moment=None, ability=None, link_id=None,
                    subkind=PlayabilitySubkind.DECISION, state=state,
                    payload=PlayabilityDecisionPayload(candidates=(Candidate(
                        ability=(key,), can_play=False, affordable=True,
                        has_legal_target=True, responsible_static=(key,), timing=False,
                        failed_checks=(FailedCheck("casting_speed"),
                                       FailedCheck("static_prohibition", (key,)))),))),
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
        for record in records:
            assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record
        assert list(record_to_dict(records[2])["payload"]) == [
            "anchor_attacker", "legal_defenders", "must_attack", "tapped_by_attack",
            "legal_attackers", "forbidden"]
        assert list(record_to_dict(records[3])["payload"]) == [
            "anchor_attacker", "legal_blockers", "forbidden", "must_block", "min_blockers"]

    def test_an_attackers_record_carries_no_what_if(self):
        attackers = [raw for raw in java_golden_records()
                     if raw["kind"] == "playability" and raw["subkind"] == "attackers"]
        assert attackers
        assert not any("what_if" in raw for raw in attackers)

    def test_a_golden_record_survives_a_write_and_a_read(self):
        for raw in java_golden_records():
            record = record_from_dict(raw)
            assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record

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
            counters={"+1/+1": 2, "LOYALTY": 3},
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


def _entity_dict(entity: EntityState) -> dict:
    record = _record(state=replace(_record().state, entities=(entity,)))
    return record_to_dict(record)["state"]["entities"][0]


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

    def test_the_shard_is_gen3(self):
        from effects.domain.shard_generation import ShardGeneration
        from effects.infrastructure.shard_io import shard_generation
        from tests.unit.effects.builders import GEN3_RECORDS

        assert shard_generation(GEN3_RECORDS) is ShardGeneration.GEN_3

    def test_every_record_writes_back_as_the_collector_wrote_it(self):
        for raw in gen3_fixture_records():
            written = record_to_dict(record_from_dict(raw))
            assert json.dumps(_spelling_free(written)) == json.dumps(_spelling_free(raw)), (
                raw["record_id"])

    def test_a_parsed_record_survives_a_write_and_a_read(self):
        for record in self._records():
            assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record

    def test_it_holds_a_random_seat_game_and_a_forge_ai_game(self):
        records = self._records()
        seat_games = {r.game_id for r in records if r.random_seat}
        assert seat_games
        assert {r.game_id for r in records} - seat_games

    def test_every_turn_action_subkind_reads_with_its_events(self):
        actions = [r for r in self._records() if r.kind is RecordKind.TURN_ACTION]
        assert {r.subkind for r in actions} == set(TurnActionSubkind)
        assert all(r.ability is None and r.payload.events for r in actions)
        draws = [r for r in actions if r.subkind is TurnActionSubkind.DRAW]
        assert all({EventType.CARD_DRAWN, EventType.ZONE_CHANGE}
                   <= {e.type for e in r.payload.events} for r in draws)

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
        declared = {}
        for r in records:
            if r.payload.tapped_by_attack is None:
                continue
            anchor = next(e for e in r.state.entities if e.id == r.payload.anchor_attacker)
            vigilant = any(line.keyword == "vigilance" for line in anchor.keyword_lines)
            declared[r.record_id] = (vigilant, r.payload.tapped_by_attack)
        assert set(declared.values()) == {(False, True), (True, False)}
        assert any(r.payload.legal_defenders == () for r in records)
        assert any(r.payload.must_attack for r in records)

    def test_a_trigger_s_record_holds_no_death_of_the_check_it_arrived_in(self):
        """SC-004 on the burn kills the fixture holds: the dies trigger's
        resolution, when present, holds no death of a creature its snapshot
        no longer shows on the battlefield."""
        for r in self._records():
            if not (isinstance(r.payload, ResolutionPayload) and r.moment is Moment.RESOLUTION
                    and r.ability and r.ability[0].trait_kind == "trigger" and not r.fork):
                continue
            zones = {e.id: e.zone for e in r.state.entities}
            for event in r.payload.events:
                if (event.type is EventType.ZONE_CHANGE
                        and event.params.get("from_zone") == "battlefield"
                        and event.params.get("to_zone") == "graveyard"
                        and event.attributed_to == "unresolved"):
                    assert all(zones.get(s, "battlefield") == "battlefield"
                               for s in event.subjects), r.record_id
