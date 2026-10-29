"""Repacking per-source parts into uniform shards of whole, mixed games."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from effects.domain.corpus_curation import record_hash
from effects.infrastructure import record_io
from effects.infrastructure.record_io import (
    iter_shard_lines,
    parse_record_line,
    read_records,
    read_shard,
    repack_shards,
    write_shard,
)


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


@pytest.fixture
def runs(tmp_path, make_record) -> list[Path]:
    """Eight parts, one per collection run, each eight games of five records.

    The shape the write pass leaves: a part per source shard, so a part is one
    JVM lifetime of one run and its games sit side by side.
    """
    out = []
    for run in range(8):
        records = [
            make_record(record_id=f"r{run}.g{game}.{n}", game_id=f"r{run}.g{game}")
            for game in range(8) for n in range(5)
        ]
        path = tmp_path / "parts" / f"run{run}.jsonl.gz"
        write_shard(path, records)
        out.append(path)
    return out


def _games(path: Path) -> list[str]:
    return [record.game_id for record in read_shard(path)]


def test_every_record_survives_once(tmp_path, parts):
    repack_shards(parts, tmp_path / "out", shard_records=4)
    before = Counter(r.record_id for p in parts for r in read_shard(p))
    after = Counter(r.record_id for r in read_records(tmp_path / "out"))
    assert after == before


def test_no_game_spans_two_shards(tmp_path, runs):
    written = repack_shards(runs, tmp_path / "out", shard_records=40)
    seen: dict[str, int] = {}
    for index, path in enumerate(written):
        for game in _games(path):
            assert seen.setdefault(game, index) == index


def test_a_shard_mixes_the_games_of_many_runs(tmp_path, runs):
    """Not a run of consecutive lifetimes: the trainer draws whole shards and
    shuffles only inside the resident one, so a shard holding one run's
    games makes every batch drawn from it correlated by run."""
    written = repack_shards(runs, tmp_path / "out", shard_records=40)
    runs_per_shard = [len({game.split(".")[0] for game in _games(p)}) for p in written]
    assert max(runs_per_shard) >= 4
    assert sum(runs_per_shard) / len(runs_per_shard) > 2


def test_records_are_shuffled_within_a_shard(tmp_path, runs):
    written = repack_shards(runs, tmp_path / "out", shard_records=40)
    largest = max(written, key=lambda p: len(_games(p)))
    ids = [record.record_id for record in read_shard(largest)]
    assert ids != sorted(ids)


def test_the_shard_count_follows_the_size(tmp_path, runs):
    written = repack_shards(runs, tmp_path / "out", shard_records=40)
    total = sum(1 for _ in read_records(tmp_path / "out"))
    assert total == 320
    assert 6 <= len(written) <= 8


def test_the_deal_is_a_function_of_the_seed(tmp_path, runs):
    def deal(name: str, seed: int) -> list[list[str]]:
        written = repack_shards(runs, tmp_path / name, shard_records=40, seed=seed)
        return [[r.record_id for r in read_shard(p)] for p in written]

    assert deal("a", 1) == deal("b", 1)
    assert deal("a2", 1) != deal("c", 2)


def test_scratch_buckets_are_removed(tmp_path, runs):
    out = tmp_path / "stratum"
    repack_shards(runs, out, shard_records=40)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["parts", "stratum"]


def test_zero_means_one_shard_per_part(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=0)
    assert len(written) == 3
    assert [sum(1 for _ in read_shard(p)) for p in written] == [3, 7, 2]


def test_names_are_zero_padded_and_sequential(tmp_path, runs):
    written = repack_shards(runs, tmp_path / "out", shard_records=40)
    assert [p.name for p in written] == [
        f"shard-{index:05d}.jsonl.gz" for index in range(1, len(written) + 1)
    ]


@pytest.mark.parametrize("groups", [3, 256])
def test_the_deal_is_the_in_memory_deal(tmp_path, runs, monkeypatch, groups):
    """Every game to ``hash(game_id) % count``, each shard shuffled under
    ``seed:index`` from the parts' order, numbered in index order.

    Three groups make every group carry several shards, the path a stratum
    with more shards than :data:`record_io._GROUPS` takes, without writing
    hundreds of files to get there.
    """
    monkeypatch.setattr(record_io, "_GROUPS", groups)
    seed, count = 7, 32
    lines = [line for part in runs for line in iter_shard_lines(part)]
    expected: dict[int, list[str]] = defaultdict(list)
    for line in lines:
        game = parse_record_line(line)["game_id"]
        expected[record_hash(game, seed=seed) % count].append(line)
    for index, bucket in expected.items():
        random.Random(f"{seed}:{index}").shuffle(bucket)

    written = repack_shards(
        runs, tmp_path / "out", shard_records=10, seed=seed, records=len(lines),
    )

    assert [p.name for p in written] == [
        f"shard-{number:05d}.jsonl.gz" for number in range(1, len(expected) + 1)
    ]
    assert [list(iter_shard_lines(p)) for p in written] == [
        expected[index] for index in sorted(expected)
    ]


def test_no_parts_writes_nothing(tmp_path):
    assert repack_shards([], tmp_path / "out", shard_records=4) == []
    assert not (tmp_path / "out").exists()
