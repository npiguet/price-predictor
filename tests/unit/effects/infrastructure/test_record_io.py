"""Shard IO round-trips and recovery (T027).

The corpus cannot be rebuilt cheaply, so a serializer bug is unrecoverable
rather than inconvenient. Every record kind round-trips here, including the
stage-two-to-four kinds no collector writes yet — the schema is frozen before
collection precisely so those can be checked now.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    CollectionMode,
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
from effects.infrastructure.record_io import (
    ShardWriter,
    count_records,
    format_record_line,
    iter_shards,
    read_records,
    read_shard,
    record_from_dict,
    record_to_dict,
    shard_path,
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
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 3}),
            outgoing=Event(type=EventType.DAMAGE_DEALT, params={"amount": 0}),
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

    def test_collection_mode_round_trips(self):
        record = _record(mode=CollectionMode.PATCHED)
        assert record_from_dict(record_to_dict(record)).mode is CollectionMode.PATCHED


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

    def test_extra_fields_are_not_model_inputs(self):
        data = record_to_dict(_record())
        data["collector_build"] = "2027.1"
        assert "extra_fields" not in record_from_dict(data).model_input_fields()


class TestShardDirectory:
    def test_a_shard_is_named_for_its_run_and_worker(self, tmp_path):
        assert shard_path(tmp_path, "run-uuid", 4).name == "run-uuid.4.jsonl"

    def test_a_writer_creates_its_directory(self, tmp_path):
        target = tmp_path / "records"
        with ShardWriter(target, "run-uuid", 0) as writer:
            writer.write(_record())
        assert (target / "run-uuid.0.jsonl").exists()

    def test_written_records_read_back(self, tmp_path):
        records = [_record(record_id=f"run.0.{i}") for i in range(3)]
        with ShardWriter(tmp_path, "run", 0) as writer:
            for record in records:
                writer.write(record)
        assert list(read_records(tmp_path)) == records

    def test_a_writer_appends_rather_than_truncating(self, tmp_path):
        with ShardWriter(tmp_path, "run", 0) as writer:
            writer.write(_record(record_id="run.0.1"))
        with ShardWriter(tmp_path, "run", 0) as writer:
            writer.write(_record(record_id="run.0.2"))
        assert count_records(tmp_path) == 2

    def test_every_shard_in_the_directory_is_loaded(self, tmp_path):
        for worker in (0, 1, 2):
            with ShardWriter(tmp_path, "run", worker) as writer:
                writer.write(_record(record_id=f"run.{worker}.1"))
        assert {r.worker for r in read_records(tmp_path)} == {"0", "1", "2"}


class TestCompressedShards:
    """The Java writer emits `.jsonl.gz` as concatenated gzip members, one per
    block of records. The corpus is the run's real cost on disk — a few
    megabytes a game — and it compresses roughly tenfold.

    Two properties matter and neither is free: a corpus collected before
    compression must keep reading, because it cannot be regenerated; and a
    worker killed mid-block truncates the final member, which must stop the
    read rather than fail it.
    """

    def _write_members(self, path, blocks: list[list[str]]) -> None:
        """One gzip member per block, concatenated, as the Java writer does."""
        import gzip

        with path.open("wb") as handle:
            for block in blocks:
                handle.write(gzip.compress(
                    "".join(line + "\n" for line in block).encode("utf-8")
                ))

    def _line(self, record_id: str) -> str:
        return json.dumps(record_to_dict(_record(record_id=record_id)))

    def test_a_compressed_shard_reads_back(self, tmp_path):
        path = tmp_path / "run.0.jsonl.gz"
        self._write_members(path, [[self._line("run.0.1"), self._line("run.0.2")]])
        assert [r.record_id for r in read_shard(path)] == ["run.0.1", "run.0.2"]

    def test_records_span_members(self, tmp_path):
        """Every member stands alone, so the reader must not stop at the first."""
        path = tmp_path / "run.0.jsonl.gz"
        self._write_members(
            path, [[self._line("run.0.1")], [self._line("run.0.2")]],
        )
        assert [r.record_id for r in read_shard(path)] == ["run.0.1", "run.0.2"]

    def test_a_truncated_final_member_stops_the_read(self, tmp_path):
        """A killed worker leaves one; everything before it still loads."""
        path = tmp_path / "run.0.jsonl.gz"
        self._write_members(
            path, [[self._line("run.0.1")], [self._line("run.0.2")]],
        )
        data = path.read_bytes()
        path.write_bytes(data[: len(data) - 12])
        assert [r.record_id for r in read_shard(path)] == ["run.0.1"]

    def test_a_half_written_line_inside_a_member_is_skipped(self, tmp_path):
        import gzip

        path = tmp_path / "run.0.jsonl.gz"
        body = self._line("run.0.1") + "\n" + self._line("run.0.2")[:20]
        path.write_bytes(gzip.compress(body.encode("utf-8")))
        assert [r.record_id for r in read_shard(path)] == ["run.0.1"]

    def test_an_uncompressed_shard_still_reads(self, tmp_path):
        """The corpus is append-only, so shards predating compression stay."""
        with ShardWriter(tmp_path, "old", 0) as writer:
            writer.write(_record(record_id="old.0.1"))
        assert [r.record_id for r in read_records(tmp_path)] == ["old.0.1"]

    def test_both_spellings_load_together(self, tmp_path):
        with ShardWriter(tmp_path, "old", 0) as writer:
            writer.write(_record(record_id="old.0.1"))
        self._write_members(
            tmp_path / "new.1.jsonl.gz", [[self._line("new.1.1")]],
        )
        assert {r.record_id for r in read_records(tmp_path)} == {
            "old.0.1", "new.1.1",
        }

    def test_counting_covers_both_spellings(self, tmp_path):
        with ShardWriter(tmp_path, "old", 0) as writer:
            writer.write(_record(record_id="old.0.1"))
        self._write_members(
            tmp_path / "new.1.jsonl.gz",
            [[self._line("new.1.1"), self._line("new.1.2")]],
        )
        assert count_records(tmp_path) == 3

    def test_a_shard_named_gz_is_not_also_read_as_plain(self, tmp_path):
        """`*.jsonl` must not claim `x.jsonl.gz` and try to parse it as text."""
        self._write_members(
            tmp_path / "run.0.jsonl.gz", [[self._line("run.0.1")]],
        )
        assert len(iter_shards(tmp_path)) == 1
        assert count_records(tmp_path) == 1

    def test_shards_are_read_in_name_order(self, tmp_path):
        for worker in (2, 0, 1):
            with ShardWriter(tmp_path, "run", worker) as writer:
                writer.write(_record(record_id=f"run.{worker}.1"))
        assert [r.worker for r in read_records(tmp_path)] == ["0", "1", "2"]

    def test_a_trailing_partial_line_is_skipped(self, tmp_path):
        """A JVM crash mid-write is expected, not exceptional."""
        path = shard_path(tmp_path, "run", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        good = format_record_line(_record(record_id="run.0.1"))
        truncated = format_record_line(_record(record_id="run.0.2"))[:40]
        path.write_text(f"{good}\n{truncated}", encoding="utf-8")
        loaded = list(read_shard(path))
        assert [r.record_id for r in loaded] == ["run.0.1"]

    def test_a_partial_line_is_left_on_disk_rather_than_repaired(self, tmp_path):
        path = shard_path(tmp_path, "run", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = format_record_line(_record())[:40]
        path.write_text(content, encoding="utf-8")
        list(read_shard(path))
        assert path.read_text(encoding="utf-8") == content

    def test_a_missing_directory_is_an_empty_corpus_not_an_error(self, tmp_path):
        assert list(read_records(tmp_path / "never-collected")) == []

    def test_non_jsonl_files_in_the_directory_are_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not a shard", encoding="utf-8")
        with ShardWriter(tmp_path, "run", 0) as writer:
            writer.write(_record())
        assert count_records(tmp_path) == 1

    def test_blank_lines_are_ignored(self, tmp_path):
        path = shard_path(tmp_path, "run", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = format_record_line(_record())
        path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
        assert len(list(read_shard(path))) == 2
