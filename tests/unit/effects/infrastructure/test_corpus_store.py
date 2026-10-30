from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import SourceShard
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import list_shards
from tests.unit.effects.domain.test_corpus_manifest import manifest


@pytest.fixture
def a_manifest():
    return manifest()


def test_a_manifest_round_trips_through_the_store(tmp_path, a_manifest):
    store = CorpusStore(tmp_path)
    store.save(a_manifest)
    assert store.load() == a_manifest


def test_load_fails_loudly_when_no_manifest_was_written(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        CorpusStore(tmp_path).load()


def test_the_store_names_the_three_output_directories(tmp_path):
    store = CorpusStore(tmp_path)
    assert store.training_dir == tmp_path / "training"
    assert store.card_disjoint_dir == tmp_path / "validation" / "card-disjoint"
    assert store.game_disjoint_dir == tmp_path / "validation" / "game-disjoint"


def test_the_listing_names_shards_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    (tmp_path / "depleted" / "run.0-a.jsonl.gz").write_bytes(b"x" * 7)
    assert list_shards(tmp_path) == (
        SourceShard(name="depleted/run.0-a.jsonl.gz", size=7),
    )


def test_clearing_the_outputs_moves_the_manifest_aside(tmp_path, a_manifest):
    """A build that fails after clearing must not leave a manifest over empty strata.

    Its digest would still match every checkpoint trained on the old dataset,
    so the evaluator would score them against whatever the failed build left.
    """
    store = CorpusStore(tmp_path)
    store.save(a_manifest)
    store.clear_outputs()
    with pytest.raises(FileNotFoundError):
        store.load()
    assert store.previous_manifest_path.exists()


def test_saving_a_manifest_removes_the_one_set_aside(tmp_path, a_manifest):
    store = CorpusStore(tmp_path)
    store.save(a_manifest)
    store.clear_outputs()
    store.save(a_manifest)
    assert store.load() == a_manifest
    assert not store.previous_manifest_path.exists()
