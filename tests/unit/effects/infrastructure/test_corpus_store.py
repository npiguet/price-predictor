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


@pytest.mark.parametrize(("path_of", "relative"), [
    (lambda store: store.manifest_path, "manifest.json"),
    (lambda store: store.previous_manifest_path, "manifest.previous.json"),
    (lambda store: store.training_dir, "training"),
    (lambda store: store.card_disjoint_dir, "validation/card-disjoint"),
    (lambda store: store.game_disjoint_dir, "validation/game-disjoint"),
    (lambda store: store.gate_one_dir, "validation/gate-one"),
    (lambda store: store.samples_dir, "validation/samples"),
    (lambda store: store.sample_path("card-disjoint"),
     "validation/samples/card-disjoint.jsonl.gz"),
    (lambda store: store.parts_dir_for("training"), ".parts/training"),
], ids=[
    "manifest", "previous-manifest", "training", "card-disjoint", "game-disjoint",
    "gate-one", "samples", "sample", "parts",
])
def test_the_store_lays_out_the_dataset(tmp_path, path_of, relative):
    """Every reader of a curated dataset finds its parts by these paths."""
    assert path_of(CorpusStore(tmp_path)) == tmp_path / relative


def test_clearing_the_outputs_removes_every_stratum_sample_and_part(tmp_path):
    store = CorpusStore(tmp_path)
    for directory in (store.gate_one_dir, store.samples_dir, store.parts_dir_for("training")):
        directory.mkdir(parents=True)
        (directory / "x").write_text("x")
    store.clear_outputs()
    assert not store.gate_one_dir.exists()
    assert not store.samples_dir.exists()
    assert not store.parts_dir.exists()


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
