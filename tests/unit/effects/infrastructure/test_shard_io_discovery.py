"""Shard IO: shards in subdirectories and Java-written shards (T027).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from effects.domain.records import RecordKind
from effects.infrastructure.shard_io import (
    iter_shards,
)


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

    def test_the_order_is_path_order_across_directories_and_spellings(self, tmp_path) -> None:
        """Shard order fixes the epoch walk and the split's accumulation: one
        path order over every subdirectory, both spellings interleaved."""
        for name in ("dirb/run.0-b.jsonl", "dira/run.0-z.jsonl", "dirc/run.0-c.jsonl.gz",
                     "dira/run.0-a.jsonl.gz", "top.0-t.jsonl"):
            self._shard(tmp_path / name)

        assert [p.relative_to(tmp_path).as_posix() for p in iter_shards(tmp_path)] == [
            "dira/run.0-a.jsonl.gz", "dira/run.0-z.jsonl", "dirb/run.0-b.jsonl",
            "dirc/run.0-c.jsonl.gz", "top.0-t.jsonl",
        ]


class TestJavaWrittenShards:
    """The shard writer's own lines (``ShardCharacterizationTest``), read
    as a shard: what a gen-3 worker writes is what every reader here reads."""

    @staticmethod
    def _shard(tmp_path: Path, golden: str) -> Path:
        from tests.unit.effects.builders import JAVA_GOLDENS

        shard = tmp_path / "run.0-l1.jsonl.gz"
        lines = (JAVA_GOLDENS / golden).read_text(encoding="utf-8")
        shard.write_bytes(gzip.compress(lines.encode("utf-8")))
        return shard

    def test_a_finished_game_reads_whole_and_ends_in_its_marker(self, tmp_path):
        from effects.domain.records import TurnActionSubkind
        from effects.infrastructure.shard_io import ended_games, read_shard

        shard = self._shard(tmp_path, "shard-game-end.jsonl")
        records = list(read_shard(shard))

        assert [r.kind for r in records] == [RecordKind.RESOLUTION] * 3 + [RecordKind.TURN_ACTION]
        assert [r.record_id for r in records] == [f"run.0-l1.{i}" for i in range(4)]
        assert {r.game_id for r in records} == {"run.0-l1.0"}
        assert {r.generation for r in records} == {3}
        marker = records[-1]
        assert marker.subkind is TurnActionSubkind.GAME_END
        assert (marker.payload.events, marker.payload.records_withheld) == ((), 0)
        assert ended_games(shard) == {"run.0-l1.0"}

    def test_a_game_without_its_marker_reads_as_cut(self, tmp_path):
        from effects.infrastructure.shard_io import ended_games, read_shard

        shard = self._shard(tmp_path, "shard-spell-and-combat.jsonl")

        assert [r.kind for r in read_shard(shard)] == [
            RecordKind.RESOLUTION, RecordKind.RESOLUTION, RecordKind.COMBAT]
        assert ended_games(shard) == frozenset()

    @staticmethod
    def _gzip_shard(tmp_path: Path) -> tuple[Path, bytes]:
        """``shard-two-games.jsonl.gz``, byte for byte: Java's gzip members,
        one a game, as ``RecordShardWriter`` flushes them at each game end."""
        from tests.unit.effects.builders import JAVA_GOLDENS

        data = (JAVA_GOLDENS / "shard-two-games.jsonl.gz").read_bytes()
        shard = tmp_path / "run.0-l1.jsonl.gz"
        shard.write_bytes(data)
        return shard, data

    @staticmethod
    def _first_member_length(data: bytes) -> int:
        import zlib

        inflater = zlib.decompressobj(wbits=31)
        inflater.decompress(data)
        return len(data) - len(inflater.unused_data)

    @staticmethod
    def _assert_ended_by_its_marker(own: list) -> None:
        from effects.domain.records import TurnActionSubkind

        assert [r.subkind for r in own].count(TurnActionSubkind.GAME_END) == 1
        assert own[-1].subkind is TurnActionSubkind.GAME_END
        assert own[-1].payload.records_withheld == 0

    def test_a_java_written_shard_reads_game_by_game_each_ended_by_its_marker(self, tmp_path):
        from effects.infrastructure.shard_io import ended_games, read_shard

        shard, data = self._gzip_shard(tmp_path)
        records = list(read_shard(shard))

        games = list(dict.fromkeys(r.game_id for r in records))
        assert games == ["run.0-l1.0", "run.0-l1.1"]
        assert [r.record_id for r in records] == [f"run.0-l1.{i}" for i in range(len(records))]
        for game in games:
            self._assert_ended_by_its_marker([r for r in records if r.game_id == game])
        assert ended_games(shard) == {"run.0-l1.0", "run.0-l1.1"}
        # One member a game: the first ends exactly where game 0's marker does.
        first = gzip.decompress(data[:self._first_member_length(data)]).decode("utf-8")
        assert [json.loads(line)["game_id"] for line in first.splitlines()] == [
            "run.0-l1.0"] * sum(r.game_id == "run.0-l1.0" for r in records)

    def test_a_java_written_shard_cut_in_its_second_game_keeps_the_first_whole(self, tmp_path):
        """A worker killed while its second game's block was being appended:
        the first game, flushed whole at its end, still reads as ended."""
        from effects.infrastructure.shard_io import ended_games, read_shard

        shard, data = self._gzip_shard(tmp_path)
        first = self._first_member_length(data)
        shard.write_bytes(data[:first + (len(data) - first) // 2])

        assert ended_games(shard) == {"run.0-l1.0"}
        assert {r.game_id for r in read_shard(shard)} == {"run.0-l1.0"}
