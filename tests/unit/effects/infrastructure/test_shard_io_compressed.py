"""Shard IO: compressed shards, truncation and damage (T027).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

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
from tests.unit.effects.infrastructure.record_codec_support import (
    _record,
)


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

    def test_a_block_cut_midway_keeps_every_whole_record_before_the_cut(self, tmp_path):
        """The Java writer's members are blocks of many records; a kill cuts
        one midway. Every record whole before the cut reads, in order, and the
        one the cut went through is dropped, never read as a broken line."""
        path = tmp_path / "run.0.jsonl.gz"
        block = [self._line(f"run.0.{i}") for i in range(2, 40)]
        self._write_members(path, [[self._line("run.0.1")], block])
        data = path.read_bytes()
        first = len(gzip.compress((self._line("run.0.1") + "\n").encode("utf-8")))
        path.write_bytes(data[: first + (len(data) - first) * 3 // 4])

        read = [r.record_id for r in read_shard(path)]

        expected = [f"run.0.{i}" for i in range(1, 40)]
        assert 2 < len(read) < len(expected)
        assert read == expected[: len(read)]

    def test_a_half_written_line_inside_a_member_is_skipped(self, tmp_path):
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
        assert count_records(tmp_path) == 2

    def test_both_spellings_yield_lines_of_one_shape(self, tmp_path):
        """Stripped of the newline and blank-free, whichever spelling: a writer
        copying lines back adds the newline itself."""
        line = format_record_line(_record())
        body = f"{line}\n\n{line}\n"
        plain = tmp_path / "a.jsonl"
        plain.write_text(body, encoding="utf-8")
        packed = tmp_path / "b.jsonl.gz"
        packed.write_bytes(gzip.compress(body.encode("utf-8")))
        assert list(iter_shard_lines(plain)) == list(iter_shard_lines(packed)) == [line, line]


class TestDamagedShards:
    """A killed worker can only cut the last member short, which ends the read
    quietly. Damage in the last member (power lost mid-write, a flipped bit)
    ends it too, with a warning and a tally entry, since nothing follows to
    lose; damage before the last would silently drop every later member, so
    it stops the read naming the shard and the byte the damaged member starts
    at. A line that does not decode names its shard and line."""

    def _line(self, record_id: str) -> str:
        return format_record_line(_record(record_id=record_id))

    def _members(self, *record_ids: str) -> list[bytes]:
        return [gzip.compress((self._line(r) + "\n").encode("utf-8")) for r in record_ids]

    def test_a_member_with_a_bad_header_mid_shard_stops_the_read_loudly(self, tmp_path):
        from effects.infrastructure.shard_io import CorruptShardError

        first, second, third = self._members("r.1", "r.2", "r.3")
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + b"XX" + second[2:] + third)
        read = []
        with pytest.raises(CorruptShardError, match=rf"run\.0-L1\.jsonl\.gz.*byte {len(first)}"):
            read.extend(iter_shard_lines(path))
        assert read == [self._line("r.1")]

    def test_a_flipped_bit_mid_shard_stops_the_read_loudly(self, tmp_path):
        """The member's check sum catches it, so no later member is lost unsaid."""
        from effects.infrastructure.shard_io import CorruptShardError

        first, second, third = self._members("r.1", "r.2", "r.3")
        damaged = bytearray(second)
        damaged[len(damaged) - 6] ^= 0x01  # inside the CRC-32 trailer
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + bytes(damaged) + third)
        with pytest.raises(CorruptShardError, match=rf"byte {len(first)}"):
            list(iter_shard_lines(path))

    def test_a_truncated_final_member_still_ends_the_read_quietly(self, tmp_path):
        first, second = self._members("r.1", "r.2")
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + second[:-8])
        assert list(iter_shard_lines(path)) == [self._line("r.1"), self._line("r.2")]
        path.write_bytes(first + second[:5])
        assert list(iter_shard_lines(path)) == [self._line("r.1")]

    def test_a_line_that_does_not_decode_names_its_shard_and_line(self, tmp_path):
        from effects.infrastructure.shard_io import RecordDecodeError, iter_shard_matching

        path = tmp_path / "run.0-L1.jsonl.gz"
        broken = json.loads(self._line("r.2"))
        broken["kind"] = "no_such_kind"
        body = f"{self._line('r.1')}\n\n{json.dumps(broken)}\n"
        path.write_bytes(gzip.compress(body.encode("utf-8")))
        for read in (lambda: list(read_shard(path)),
                     lambda: list(read_window(tmp_path)),
                     lambda: list(iter_shard_matching(path))):
            with pytest.raises(RecordDecodeError,
                               match=r"run\.0-L1\.jsonl\.gz, line 3: .*no_such_kind"):
                read()

    def _ends_quietly(self, path, caplog, expected):
        from effects.infrastructure import shard_io

        tally = shard_io.DecodeTally()
        with caplog.at_level("WARNING", logger=shard_io.__name__):
            assert list(iter_shard_lines(path, tally=tally)) == expected
        assert tally.damaged_tails == [f"{path}, byte {self._at}"]
        assert any("is corrupt" in r.getMessage() and str(path) in r.getMessage()
                   for r in caplog.records)

    def test_a_flipped_check_sum_in_the_last_member_ends_the_read_quietly(
            self, tmp_path, caplog):
        first, second = self._members("r.1", "r.2")
        damaged = bytearray(second)
        damaged[len(damaged) - 6] ^= 0x01  # inside the CRC-32 trailer
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + bytes(damaged))
        self._at = len(first)
        self._ends_quietly(path, caplog, [self._line("r.1")])

    def test_garbage_after_the_last_member_ends_the_read_quietly(self, tmp_path, caplog):
        """Power lost mid-write can leave anything at the end of the file."""
        first, second = self._members("r.1", "r.2")
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + second + b"\xffgarbage\x01\x02")
        self._at = len(first) + len(second)
        self._ends_quietly(path, caplog, [self._line("r.1"), self._line("r.2")])

    def test_a_bad_header_on_the_last_member_ends_the_read_quietly(self, tmp_path, caplog):
        first, second = self._members("r.1", "r.2")
        path = tmp_path / "run.0-L1.jsonl.gz"
        path.write_bytes(first + b"XX" + second[2:])
        self._at = len(first)
        self._ends_quietly(path, caplog, [self._line("r.1")])

    def test_the_corrupt_shard_error_pickles_with_its_location(self):
        import pickle

        from effects.infrastructure.shard_io import RecordDecodeError

        error = RecordDecodeError.at(Path("s.jsonl.gz"), 7, KeyError("kind"))
        back = pickle.loads(pickle.dumps(error))
        assert (str(back), back.path, back.line) == (str(error), Path("s.jsonl.gz"), 7)

    @pytest.mark.parametrize("line", [
        "[1, 2]", '"a string"',
        None,  # an entity that is a string: built below
    ])
    def test_a_malformed_shape_names_its_shard_and_line(self, tmp_path, line):
        from effects.infrastructure.shard_io import RecordDecodeError

        if line is None:
            raw = record_to_dict(_record(record_id="r.2"))
            raw["state"]["entities"] = ["E1"]
            line = json.dumps(raw)
        path = tmp_path / "run.0-L1.jsonl"
        path.write_text(f"{self._line('r.1')}\n{line}\n", encoding="utf-8")
        for read in (lambda: list(read_shard(path)), lambda: list(read_window(tmp_path))):
            with pytest.raises(RecordDecodeError, match=r"run\.0-L1\.jsonl, line 2"):
                read()

    def test_a_line_of_bad_utf8_in_a_gzip_shard_names_its_shard_and_line(self, tmp_path):
        from effects.infrastructure.shard_io import RecordDecodeError

        path = tmp_path / "run.0-L1.jsonl.gz"
        body = (self._line("r.1") + "\n").encode("utf-8") + b'{"record_id": "\xff\xfe"}\n'
        path.write_bytes(gzip.compress(body))
        with pytest.raises(RecordDecodeError, match=r"run\.0-L1\.jsonl\.gz, line 2"):
            list(read_shard(path))

    def test_a_line_that_is_no_json_names_its_shard_and_line(self, tmp_path):
        from effects.infrastructure.shard_io import RecordDecodeError

        path = tmp_path / "run.0-L1.jsonl"
        path.write_text(f"{self._line('r.1')}\n{{not json\n", encoding="utf-8")
        with pytest.raises(RecordDecodeError, match=r"run\.0-L1\.jsonl, line 2"):
            list(read_shard(path))
