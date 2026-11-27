"""Shard IO: filtered reads, the window and values off the closed lists.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.records import RecordKind
from effects.infrastructure.record_codec import record_to_dict
from effects.infrastructure.shard_io import (
    ReadWindow,
    iter_shards,
    read_records,
    read_shard,
    read_window,
)
from tests.unit.effects.builders import GEN3_RECORDS
from tests.unit.effects.file_builders import RawShardWriter
from tests.unit.effects.infrastructure.record_codec_support import (
    ALL_KINDS,
    _record,
)


class TestReadRecordsMatching:
    """The envelope-head filter keeps exactly what the full parse would."""

    def _copy(self, tmp_path: Path) -> Path:
        target = tmp_path / "records"
        target.mkdir()
        (target / GEN3_RECORDS.name).write_bytes(GEN3_RECORDS.read_bytes())
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
                directory, game_ids=games, kinds={RecordKind.COMBAT},
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
            tmp_path, game_ids={record.game_id}, kinds={RecordKind.PLAYABILITY},
        ))
        assert [r.record_id for r in got] == [record.record_id]

    def test_where_sees_each_head_once_after_the_other_filters(self, tmp_path):
        """Once per line, so a predicate may keep state; parsed or peeked alike."""
        from effects.infrastructure.shard_io import iter_shard_matching

        directory = self._copy(tmp_path)
        shard = directory / GEN3_RECORDS.name
        combats = [r.record_id for r in read_records(directory) if r.kind is RecordKind.COMBAT]
        seen: list[str] = []

        def every_other(head) -> bool:
            seen.append(head.record_id)
            return len(seen) % 2 == 1

        got = [
            r.record_id
            for r in iter_shard_matching(shard, kinds={RecordKind.COMBAT}, where=every_other)
        ]
        assert seen == combats and combats
        assert got == combats[::2]

    def test_the_head_is_read_off_a_written_line(self):
        from effects.infrastructure.record_codec import format_record_line
        from effects.infrastructure.shard_io import peek_envelope

        record = _record(**ALL_KINDS["playability-blockers"])
        head = peek_envelope(format_record_line(record))
        assert head == (record.record_id, record.game_id, "playability")

    def test_the_head_is_read_off_every_line_the_java_writer_wrote(self):
        """The head regex is written against the Java writer's layout: on every
        characterization line, of every kind, it reads what parsing reads."""
        from effects.infrastructure.shard_io import peek_envelope
        from tests.unit.effects.builders import JAVA_GOLDENS

        lines = [line for path in sorted(JAVA_GOLDENS.glob("*.jsonl"))
                 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kinds = set()
        for line in lines:
            raw = json.loads(line)
            assert peek_envelope(line) == (raw["record_id"], raw["game_id"], raw["kind"]), line[:80]
            kinds.add(raw["kind"])
        assert {"resolution", "rewrite", "trigger", "combat", "playability", "continuous",
                "turn_action"} <= kinds

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
        assert len(list(read_window(tmp_path, ReadWindow(limit=2)))) == 2
        assert len(list(read_window(tmp_path))) == 5

    def test_each_raw_record_is_observed_under_its_shard_s_relative_name(self, tmp_path):
        self._write(tmp_path, 2)
        seen = []
        records = list(read_window(tmp_path, ReadWindow(observe=lambda name, data: seen.append(
            (name, data["record_id"]),
        ))))
        shard = iter_shards(tmp_path)[0].relative_to(tmp_path).as_posix()
        assert seen == [(shard, record.record_id) for record in records]


class TestValuesOutsideTheClosedLists:
    """A value outside the wire's closed lists decodes, and is counted, the
    first sighting reported by shard and line."""

    def test_it_is_counted_and_reported_once(self, tmp_path, caplog):
        from effects.infrastructure import shard_io

        raw = record_to_dict(_record(record_id="r.1"))
        raw["state"]["entities"][0]["zone"] = "nowhere-zone"
        line = json.dumps(raw)
        path = tmp_path / "run.0-L1.jsonl"
        path.write_text(f"{line}\n{line}\n", encoding="utf-8")
        tally = shard_io.DecodeTally()
        with caplog.at_level("WARNING", logger=shard_io.__name__):
            records = list(read_shard(path, tally=tally))
        assert [r.state.entities[0].zone for r in records] == ["nowhere-zone"] * 2
        assert tally.off_list[("entity.zone", "'nowhere-zone'")] == 2
        warned = [r.getMessage() for r in caplog.records if "nowhere-zone" in r.getMessage()]
        assert len(warned) == 1 and "run.0-L1.jsonl, line 1: entity.zone" in warned[0]

    def test_a_taken_tally_travels_and_sums(self, tmp_path):
        """What a pool worker hands back with a shard's result, its caller adds
        to its own: the tally is the reader's, emptied when taken."""
        from effects.infrastructure import shard_io

        raw = record_to_dict(_record(record_id="r.1"))
        raw["state"]["entities"][0]["zone"] = "elsewhere-zone"
        path = tmp_path / "run.0-L1.jsonl"
        path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        tally = shard_io.DecodeTally()
        list(read_shard(path, tally=tally))
        taken = tally.take()
        assert not tally
        total = shard_io.DecodeTally()
        total.merge(taken)
        total.merge(taken)
        assert total.off_list[("entity.zone", "'elsewhere-zone'")] == 2
        assert "  entity.zone 'elsewhere-zone': 2" in total.report_lines()
