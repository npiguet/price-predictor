from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, SourceShard


def manifest(**overrides) -> CorpusManifest:
    base = dict(
        seed=42,
        surface="script",
        vocab_path="models/effects/vocab-script.txt",
        variant_scripts="output/effects/variant-scripts",
        holdout_permille=20,
        holdout_max_carriers=8,
        text_cap=200,
        card_disjoint_text_cap=50,
        game_disjoint_target=1000,
        training_records=0,
        class_mix={"rewrite": 1.0},
        delivered_mix={"rewrite": 1.0},
        held_out_cards=("soul echo",),
        held_out_texts=("deals 3 damage",),
        card_disjoint_games=("run.0-a.1",),
        game_disjoint_games=("run.0-a.2",),
        rarity={"deals 3 damage": 7},
        sources=(SourceShard(name="depleted/run.0-a.jsonl.gz", size=1234),),
        per_class={"rewrite": ClassCounts(read=10, kept=4, dropped_by_cap=6, unique_texts=2)},
        per_stratum={
            "training": 4, "card-disjoint": 3, "game-disjoint": 2,
            "dropped-held-out": 1,
        },
        unique_texts={"training": 2, "card-disjoint": 1, "game-disjoint": 1},
        shortfall={},
    )
    base.update(overrides)
    return CorpusManifest(**base)


@pytest.fixture
def manifest_dict() -> dict:
    return manifest().as_dict()


def test_round_trips_through_a_dict():
    original = manifest()
    assert CorpusManifest.from_dict(original.as_dict()) == original


def test_digest_is_stable_across_instances():
    assert manifest().digest() == manifest().digest()


def test_digest_changes_when_a_recorded_decision_changes():
    assert manifest().digest() != manifest(text_cap=100).digest()


def test_digest_ignores_the_order_a_source_list_was_built_in():
    a = manifest(sources=(
        SourceShard(name="b.jsonl.gz", size=2), SourceShard(name="a.jsonl.gz", size=1),
    ))
    b = manifest(sources=(
        SourceShard(name="a.jsonl.gz", size=1), SourceShard(name="b.jsonl.gz", size=2),
    ))
    assert a.digest() == b.digest()


def test_drift_names_added_and_removed_shards():
    current = (
        SourceShard(name="a.jsonl.gz", size=1),
        SourceShard(name="c.jsonl.gz", size=3),
    )
    added, removed, resized = manifest().drift(current)
    assert added == ("a.jsonl.gz", "c.jsonl.gz")
    assert removed == ("depleted/run.0-a.jsonl.gz",)
    assert resized == ()


def test_drift_reports_a_shard_that_grew():
    current = (SourceShard(name="depleted/run.0-a.jsonl.gz", size=9999),)
    added, removed, resized = manifest().drift(current)
    assert (added, removed) == ((), ())
    assert resized == ("depleted/run.0-a.jsonl.gz",)


def test_digest_changes_when_the_delivered_mixture_changes():
    """Two datasets holding different class proportions are different
    datasets, so a checkpoint pinned to one must refuse the other."""
    assert manifest().digest() != manifest(
        delivered_mix={"rewrite": 0.5, "combat": 0.5},
    ).digest()


def test_a_manifest_written_before_the_rework_still_loads(manifest_dict):
    """Every field this rework added has a default (FR-143 compatibility)."""
    for key in (
        "quality_dropped", "games_by_source", "held_out_games_by_source",
        "shard_records", "validation_sample", "max_events_per_record",
    ):
        manifest_dict.pop(key, None)
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.quality_dropped == {}
    assert loaded.games_by_source == {}
    assert loaded.held_out_games_by_source == {}
    assert loaded.shard_records == 0
    assert loaded.validation_sample == 0
    assert loaded.max_events_per_record == 0


def test_the_new_fields_round_trip(manifest_dict):
    manifest_dict.update({
        "quality_dropped": {"no-ability": 3},
        "games_by_source": {"depleted": 10, "full-strength": 4},
        "held_out_games_by_source": {"depleted": 1, "full-strength": 4},
        "shard_records": 2000,
        "validation_sample": 2048,
        "max_events_per_record": 64,
    })
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.as_dict()["quality_dropped"] == {"no-ability": 3}
    assert CorpusManifest.from_dict(loaded.as_dict()) == loaded
