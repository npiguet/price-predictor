from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import SourceShard
from effects.infrastructure.corpus_store import CorpusStore, current_sources
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


def test_current_sources_lists_shards_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    (tmp_path / "depleted" / "run.0-a.jsonl.gz").write_bytes(b"x" * 7)
    assert current_sources(tmp_path) == (
        SourceShard(name="depleted/run.0-a.jsonl.gz", size=7),
    )
