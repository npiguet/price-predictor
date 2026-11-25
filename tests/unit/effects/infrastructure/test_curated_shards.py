"""The shards on disk against the manifest's listing (finding 11.4)."""

from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import CuratedShard
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.curated_shards import (
    CorpusShardsError,
    count_records,
    require_listed_shards,
)
from effects.infrastructure.shard_io import write_shard
from tests.unit.effects.application.test_train_startup import _manifest


@pytest.fixture
def store(tmp_path, make_record):
    store = CorpusStore(tmp_path / "corpus")
    for name in ("shard-00001.jsonl.gz", "shard-00002.jsonl.gz"):
        write_shard(store.training_dir / name, [
            make_record(record_id=f"{name}-{i}", game_id="g") for i in range(2)
        ])
    write_shard(store.output_dir("game-disjoint") / "shard-00001.jsonl.gz",
                [make_record(record_id="v", game_id="v")])
    return store


def _listing(**strata):
    return {
        stratum: tuple(CuratedShard(name, records) for name, records in shards)
        for stratum, shards in strata.items()
    }


def test_the_listed_shards_are_returned_by_stratum_in_the_manifests_order(store):
    manifest = _manifest(shards=_listing(
        training=[("shard-00001.jsonl.gz", 2), ("shard-00002.jsonl.gz", 2)],
        **{"game-disjoint": [("shard-00001.jsonl.gz", 1)]},
    ))
    listed = require_listed_shards(store, manifest)
    assert listed["training"] == (
        store.training_dir / "shard-00001.jsonl.gz", store.training_dir / "shard-00002.jsonl.gz",
    )
    assert listed["game-disjoint"] == (store.output_dir("game-disjoint") / "shard-00001.jsonl.gz",)
    assert listed["card-disjoint"] == ()


def test_every_difference_is_named_in_one_refusal(store):
    (store.training_dir / "shard-00002.jsonl.gz").unlink()
    manifest = _manifest(shards=_listing(
        training=[("shard-00001.jsonl.gz", 3), ("shard-00002.jsonl.gz", 2)],
        **{"game-disjoint": [("shard-00001.jsonl.gz", 1)]},
    ))
    with pytest.raises(CorpusShardsError) as refused:
        require_listed_shards(store, manifest)
    message = str(refused.value)
    assert "training: shard-00001.jsonl.gz holds 2 record(s), the manifest 3" in message
    assert "training: shard-00002.jsonl.gz is listed but absent" in message


def test_a_shard_in_a_subdirectory_of_a_stratum_is_not_listed(store, make_record):
    write_shard(store.training_dir / "old" / "shard-00001.jsonl.gz", [make_record()])
    manifest = _manifest(shards=_listing(
        training=[("shard-00001.jsonl.gz", 2), ("shard-00002.jsonl.gz", 2)],
        **{"game-disjoint": [("shard-00001.jsonl.gz", 1)]},
    ))
    with pytest.raises(CorpusShardsError, match="training: old/shard-00001.jsonl.gz is not listed"):
        require_listed_shards(store, manifest)


def test_records_are_counted_by_line_without_decoding_them(tmp_path):
    import gzip

    with gzip.open(tmp_path / "a.jsonl.gz", "wt", encoding="utf-8") as out:
        out.write("not json" + chr(10) * 2 + "nor this" + chr(10))
    assert count_records(tmp_path / "a.jsonl.gz") == 2
