"""Shard IO: writing a shard and finding the games it ended (T027).
"""

from __future__ import annotations

import gzip
import json

from effects.infrastructure.record_codec import record_to_dict
from tests.unit.effects.infrastructure.record_codec_support import (
    _record,
)


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


def test_ended_games_are_the_games_whose_marker_the_shard_holds(tmp_path):
    """WP7: what the build reads for cut games, a reordered envelope included."""
    from dataclasses import replace

    from effects.infrastructure.shard_io import ended_games, write_shard
    from tests.unit.effects.builders import game_end_marker

    ended = replace(_record(), game_id="g1")
    cut = replace(_record(), record_id="run-uuid.4.2", game_id="g2")
    reordered = replace(_record(), record_id="run-uuid.4.3", game_id="g3")
    shard = tmp_path / "run.0-a.jsonl.gz"
    write_shard(shard, [ended, game_end_marker(ended), cut, reordered])
    marker = record_to_dict(game_end_marker(reordered))
    with gzip.open(shard, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(reversed(list(marker.items())))) + "\n")
    assert ended_games(shard) == {"g1", "g3"}
