"""Repacking per-source parts into uniform shards cut at game boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.record_io import read_records, read_shard, repack_shards, write_shard


@pytest.fixture
def parts(tmp_path, make_record) -> list[Path]:
    """Three parts: 3, 7 and 2 records; games of 2, 2, 3, 2, 2, 1, 1 records."""
    def game(gid, n):
        return [make_record(record_id=f"{gid}.{i}", game_id=gid) for i in range(n)]
    a = game("g1", 2) + game("g2", 1)
    b = game("g2", 1) + game("g3", 3) + game("g4", 2) + game("g5", 1)
    c = game("g6", 1) + game("g7", 1)
    out = []
    for name, records in (("a", a), ("b", b), ("c", c)):
        path = tmp_path / "parts" / f"{name}.jsonl.gz"
        write_shard(path, records)
        out.append(path)
    return out


def test_every_record_survives_in_order(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=4)
    before = [r.record_id for p in parts for r in read_shard(p)]
    after = [r.record_id for r in read_records(tmp_path / "out")]
    assert after == before
    assert [p.name for p in written] == sorted(p.name for p in written)


def test_a_shard_closes_at_a_game_boundary_once_full(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=4)
    games = [[r.game_id for r in read_shard(p)] for p in written]
    # 4 records fill the first shard; it closes at the first game boundary once full.
    assert games[0] == ["g1", "g1", "g2", "g2"]
    assert games[1] == ["g3", "g3", "g3", "g4", "g4"]
    assert games[2] == ["g5", "g6", "g7"]
    # No game spans two shards.
    seen = {}
    for index, shard in enumerate(games):
        for gid in shard:
            assert seen.setdefault(gid, index) == index


def test_zero_means_one_shard_per_part(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=0)
    assert len(written) == 3
    assert [sum(1 for _ in read_shard(p)) for p in written] == [3, 7, 2]


def test_names_are_zero_padded_and_sequential(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=2)
    assert written[0].name == "shard-00001.jsonl.gz"
    assert written[1].name == "shard-00002.jsonl.gz"


def test_no_parts_writes_nothing(tmp_path):
    assert repack_shards([], tmp_path / "out", shard_records=4) == []
    assert not (tmp_path / "out").exists()
