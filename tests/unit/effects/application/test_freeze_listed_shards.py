"""The freeze reads the shards the manifest lists, and nothing else (finding 11.4)."""

from __future__ import annotations

import dataclasses

import pytest

from effects.application.freeze_selection_set import run, stratum_reader
from effects.domain.corpus_manifest import CuratedShard
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.curated_shards import CorpusShardsError
from effects.infrastructure.shard_io import write_shard
from tests.unit.effects.application.test_freeze_selection_set import GD, _manifest, _two_halves


@pytest.fixture
def store(tmp_path, make_record):
    """A corpus whose game-disjoint stratum holds the one shard its manifest lists."""
    store = CorpusStore(tmp_path / "corpus")
    write_shard(store.output_dir("game-disjoint") / "shard-00001.jsonl.gz",
                [make_record(record_id="r", game_id="g0")])
    manifest = dataclasses.replace(
        _manifest(_two_halves(1)),
        shards={"game-disjoint": (CuratedShard("shard-00001.jsonl.gz", 1),)},
    )
    store.save(manifest)
    return store


def test_an_extra_shard_in_a_validation_stratum_is_refused_by_name(store, tmp_path, make_record):
    write_shard(store.output_dir("game-disjoint") / "stale.jsonl.gz",
                [make_record(record_id="old", game_id="g0")])
    with pytest.raises(CorpusShardsError, match=f"{GD}: stale.jsonl.gz is not listed"):
        run(store.directory, reports_dir=tmp_path / "reports", min_games_per_half=1)
    assert not (tmp_path / "reports").exists()


def test_a_reader_walks_the_paths_it_is_given_and_no_directory(store, make_record):
    stale = store.output_dir("game-disjoint") / "stale.jsonl.gz"
    write_shard(stale, [make_record(record_id="old", game_id="g0")])
    listed = (store.output_dir("game-disjoint") / "shard-00001.jsonl.gz",)
    ids = [r.record_id for shard in stratum_reader(listed)() for r in shard]
    assert ids == ["r"]
