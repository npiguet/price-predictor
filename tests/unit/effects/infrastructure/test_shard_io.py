"""Shard IO: reading, recovery, discovery and the generation check (T027).

A JVM killed mid-write is the normal case, so a truncated shard must stop a
read rather than fail it, and every shard under a corpus directory must be
found whatever its spelling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.domain.records import RecordKind
from effects.infrastructure.record_codec import format_record_line, record_to_dict
from effects.infrastructure.shard_io import (
    count_records,
    iter_shard_lines,
    iter_shards,
    read_records,
    read_shard,
    read_window,
)
from tests.unit.effects.builders import RawShardWriter
from tests.unit.effects.infrastructure.test_record_codec import (
    _GEN1_FIXTURE,
    ALL_KINDS,
    _record,
)


class TestShardDirectory:
    def test_every_shard_in_the_directory_is_loaded(self, tmp_path):
        for worker in (0, 1, 2):
            with RawShardWriter(tmp_path, "run", worker, f"L{worker}") as writer:
                writer.write(_record(record_id=f"run.{worker}.1"))
        assert {r.record_id for r in read_records(tmp_path)} == {
            "run.0.1", "run.1.1", "run.2.1",
        }


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
        with RawShardWriter(tmp_path, "old", 0, "L1", compressed=False) as writer:
            writer.write(_record(record_id="old.0.1"))
        assert [r.record_id for r in read_records(tmp_path)] == ["old.0.1"]

    def test_both_spellings_load_together(self, tmp_path):
        with RawShardWriter(tmp_path, "old", 0, "L1", compressed=False) as writer:
            writer.write(_record(record_id="old.0.1"))
        self._write_members(
            tmp_path / "new.1.jsonl.gz", [[self._line("new.1.1")]],
        )
        assert {r.record_id for r in read_records(tmp_path)} == {
            "old.0.1", "new.1.1",
        }

    def test_counting_covers_both_spellings(self, tmp_path):
        with RawShardWriter(tmp_path, "old", 0, "L1", compressed=False) as writer:
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
            with RawShardWriter(tmp_path, "run", worker, f"L{worker}") as writer:
                writer.write(_record(record_id=f"run.{worker}.1"))
        assert [r.record_id for r in read_records(tmp_path)] == [
            "run.0.1", "run.1.1", "run.2.1",
        ]

    def test_a_trailing_partial_line_is_skipped(self, tmp_path):
        """A JVM crash mid-write is expected, not exceptional."""
        path = tmp_path / "run.0-L1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        good = format_record_line(_record(record_id="run.0.1"))
        truncated = format_record_line(_record(record_id="run.0.2"))[:40]
        path.write_text(f"{good}\n{truncated}", encoding="utf-8")
        loaded = list(read_shard(path))
        assert [r.record_id for r in loaded] == ["run.0.1"]

    def test_a_partial_line_is_left_on_disk_rather_than_repaired(self, tmp_path):
        path = tmp_path / "run.0-L1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = format_record_line(_record())[:40]
        path.write_text(content, encoding="utf-8")
        list(read_shard(path))
        assert path.read_text(encoding="utf-8") == content

    def test_a_missing_directory_is_an_empty_corpus_not_an_error(self, tmp_path):
        assert list(read_records(tmp_path / "never-collected")) == []

    def test_non_jsonl_files_in_the_directory_are_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not a shard", encoding="utf-8")
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            writer.write(_record())
        assert count_records(tmp_path) == 1

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "run.0-L1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = format_record_line(_record())
        path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
        assert len(list(read_shard(path))) == 2


class TestShardsInSubdirectories:
    """Shards nested under the corpus directory are part of the corpus (T173).

    A flat directory is what the quickstart writes, but separating a depleted
    collection run from a full-strength one into two subdirectories is the
    obvious way to keep them straight — and a non-recursive glob reads none of
    them while reporting nothing. Every consumer goes through this function:
    the trainer would see an empty corpus, and `collect-coverage` would recount
    every card as having zero records and chase a target already met.
    """

    def _shard(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    def test_it_finds_shards_nested_one_level_down(self, tmp_path) -> None:
        self._shard(tmp_path / "depleted" / "run.0-a.jsonl.gz")
        self._shard(tmp_path / "full-strength" / "run.0-b.jsonl.gz")
        self._shard(tmp_path / "top.0-c.jsonl.gz")

        found = iter_shards(tmp_path)

        assert {p.name for p in found} == {
            "run.0-a.jsonl.gz", "run.0-b.jsonl.gz", "top.0-c.jsonl.gz",
        }

    def test_it_finds_both_spellings_when_nested(self, tmp_path) -> None:
        self._shard(tmp_path / "old" / "run.0-a.jsonl")
        self._shard(tmp_path / "new" / "run.0-b.jsonl.gz")

        assert len(iter_shards(tmp_path)) == 2

    def test_a_nested_gz_is_not_claimed_twice(self, tmp_path) -> None:
        """Each spelling's glob claims its own shards, and only those."""
        self._shard(tmp_path / "sub" / "run.0-a.jsonl.gz")

        found = iter_shards(tmp_path)

        assert len(found) == 1

    def test_the_order_is_stable_across_calls(self, tmp_path) -> None:
        """Shard order fixes the epoch walk and the split's accumulation."""
        for name in ("b", "a", "c"):
            self._shard(tmp_path / f"dir{name}" / f"run.0-{name}.jsonl.gz")

        assert iter_shards(tmp_path) == iter_shards(tmp_path)
        assert iter_shards(tmp_path) == sorted(iter_shards(tmp_path))


def test_write_shard_round_trips_through_the_reader(tmp_path):
    from effects.infrastructure.shard_io import read_shard, write_shard

    records = [_record(record_id=f"run.0-a.{i}") for i in range(5)]
    path = tmp_path / "curated.0-a.jsonl.gz"

    assert write_shard(path, records) == 5

    assert [r.record_id for r in read_shard(path)] == [r.record_id for r in records]


def test_write_shard_creates_the_directory_it_writes_into(tmp_path):
    from effects.infrastructure.shard_io import write_shard

    path = tmp_path / "training" / "curated.0-a.jsonl.gz"
    write_shard(path, [_record(record_id="run.0-a.0")])
    assert path.exists()


def test_write_shard_writes_a_readable_empty_shard_for_no_records(tmp_path):
    from effects.infrastructure.shard_io import read_shard, write_shard

    path = tmp_path / "curated.0-a.jsonl.gz"
    assert write_shard(path, []) == 0
    assert list(read_shard(path)) == []


class TestShardGeneration:
    """FR-033: a shard's generation by its first record; mixed sets refused (T069)."""

    def _gen2_shard(self, path: Path) -> Path:
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as out:
            for line in iter_shard_lines(_GEN1_FIXTURE):
                data = json.loads(line)
                data["random_seat"] = False
                if data.get("subkind") in ("attackers", "blockers"):
                    data["what_if"] = True
                out.write(json.dumps(data) + "\n")
        return path

    def test_the_real_gen1_fixture_is_gen1(self):
        from effects.infrastructure.shard_io import ShardGeneration, shard_generation

        assert shard_generation(_GEN1_FIXTURE) is ShardGeneration.GEN_1

    def test_a_shard_carrying_random_seat_is_gen2(self, tmp_path):
        from effects.infrastructure.shard_io import ShardGeneration, shard_generation

        assert shard_generation(self._gen2_shard(tmp_path / "g2.jsonl.gz")) is ShardGeneration.GEN_2

    def test_an_empty_shard_has_no_generation(self, tmp_path):
        from effects.infrastructure.shard_io import shard_generation

        empty = tmp_path / "e.jsonl"
        empty.write_text("", encoding="utf-8")
        assert shard_generation(empty) is None

    def test_a_set_holding_both_generations_is_refused(self, tmp_path):
        from effects.infrastructure.shard_io import MixedGenerationsError, refuse_mixed_generations

        gen2 = self._gen2_shard(tmp_path / "g2.jsonl.gz")
        with pytest.raises(MixedGenerationsError, match="gen-1"):
            refuse_mixed_generations([_GEN1_FIXTURE, gen2])

    def test_a_single_generation_set_passes(self, tmp_path):
        from effects.infrastructure.shard_io import refuse_mixed_generations

        gen2 = self._gen2_shard(tmp_path / "g2.jsonl.gz")
        refuse_mixed_generations([gen2, gen2])
        refuse_mixed_generations([])


class TestReadRecordsMatching:
    """The envelope-head filter keeps exactly what the full parse would."""

    def _copy(self, tmp_path: Path) -> Path:
        target = tmp_path / "records"
        target.mkdir()
        (target / _GEN1_FIXTURE.name).write_bytes(_GEN1_FIXTURE.read_bytes())
        return target

    def test_it_matches_a_filter_over_every_parsed_record(self, tmp_path):
        from effects.infrastructure.shard_io import read_records_matching

        directory = self._copy(tmp_path)
        everything = list(read_records(directory))
        games = {everything[0].game_id, everything[-1].game_id}
        expected = [
            r.record_id for r in everything
            if r.game_id in games and r.kind is RecordKind.COMBAT
        ]
        got = [
            r.record_id for r in read_records_matching(
                directory, game_ids=games, kinds={"combat"},
            )
        ]
        assert got == expected and expected

    def test_no_filter_reads_everything(self, tmp_path):
        from effects.infrastructure.shard_io import read_records_matching

        directory = self._copy(tmp_path)
        assert len(list(read_records_matching(directory))) == len(
            list(read_records(directory))
        )

    def test_a_line_with_an_unexpected_head_is_parsed_not_dropped(self, tmp_path):
        from effects.infrastructure.shard_io import read_records_matching

        record = _record(**ALL_KINDS["playability-blockers"])
        data = record_to_dict(record)
        # A writer that put kind before game_id: the fast path cannot read it.
        reordered = {"kind": data["kind"], **data}
        path = tmp_path / "x.jsonl"
        path.write_text(json.dumps(reordered) + "\n", encoding="utf-8")
        got = list(read_records_matching(
            tmp_path, game_ids={record.game_id}, kinds={"playability"},
        ))
        assert [r.record_id for r in got] == [record.record_id]

    def test_the_head_is_read_off_a_written_line(self):
        from effects.infrastructure.record_codec import format_record_line
        from effects.infrastructure.shard_io import peek_envelope

        record = _record(**ALL_KINDS["playability-blockers"])
        head = peek_envelope(format_record_line(record))
        assert head == (record.record_id, record.game_id, "playability")

    def test_a_head_of_another_shape_reads_as_none(self):
        from effects.infrastructure.shard_io import peek_envelope

        assert peek_envelope('{"kind":"combat","game_id":"g","record_id":"r"}') is None


class TestTheWindow:
    """``validate-corpus`` reads the first records of a collection pass."""

    def _write(self, tmp_path: Path, count: int) -> None:
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for index in range(count):
                writer.write(_record(record_id=f"run.0-L1.{index}"))

    def test_limit_stops_reading(self, tmp_path):
        self._write(tmp_path, 5)
        assert len(list(read_window(tmp_path, limit=2))) == 2
        assert len(list(read_window(tmp_path))) == 5

    def test_each_raw_record_is_observed_under_its_shard_s_relative_name(self, tmp_path):
        self._write(tmp_path, 2)
        seen = []
        records = list(read_window(tmp_path, observe=lambda name, data: seen.append(
            (name, data["record_id"]),
        )))
        shard = iter_shards(tmp_path)[0].relative_to(tmp_path).as_posix()
        assert seen == [(shard, record.record_id) for record in records]
