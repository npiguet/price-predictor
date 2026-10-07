from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
        "unattributed_records", "token_keys_remapped", "token_keys_ambiguous",
        "forge_tokenscripts",
    ):
        manifest_dict.pop(key, None)
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.quality_dropped == {}
    assert loaded.games_by_source == {}
    assert loaded.held_out_games_by_source == {}
    assert loaded.shard_records == 0
    assert loaded.validation_sample == 0
    assert loaded.max_events_per_record == 0
    assert loaded.unattributed_records == 0
    assert loaded.token_keys_remapped == 0
    assert loaded.token_keys_ambiguous == {}
    assert loaded.forge_tokenscripts == ""


def test_the_new_fields_round_trip(manifest_dict):
    manifest_dict.update({
        "quality_dropped": {"no-ability": 3},
        "games_by_source": {"depleted": 10, "full-strength": 4},
        "held_out_games_by_source": {"depleted": 1, "full-strength": 4},
        "shard_records": 2000,
        "validation_sample": 2048,
        "max_events_per_record": 64,
        "unattributed_records": 512,
        "token_keys_remapped": 17,
        "token_keys_ambiguous": {"goblin_token": 4},
        "forge_tokenscripts": "../forge/forge-gui/res/tokenscripts",
    })
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.as_dict()["quality_dropped"] == {"no-ability": 3}
    assert loaded.unattributed_records == 512
    assert loaded.token_keys_remapped == 17
    assert loaded.token_keys_ambiguous == {"goblin_token": 4}
    assert loaded.forge_tokenscripts == "../forge/forge-gui/res/tokenscripts"
    assert CorpusManifest.from_dict(loaded.as_dict()) == loaded


def test_the_digest_changes_when_the_remap_did_something_else():
    """Two datasets whose old token keys resolved differently are different
    datasets: the same raw shards, keyed at different abilities."""
    assert manifest().digest() != manifest(token_keys_remapped=1).digest()


def test_the_no_acting_text_tally_stays_out_of_the_digest(manifest_dict):
    """A tally added after the fact must not invalidate older checkpoints.

    The digest is what FR-147 pins a checkpoint to, and it stands for the
    split and the contents. ``no_acting_text_scripts`` says which scripts the
    build refused records through, which an operator reads and no sampling
    decision depends on, so a manifest written before the field existed has to
    hash exactly as one that carries it.
    """
    manifest_dict.pop("no_acting_text_scripts", None)
    before = CorpusManifest.from_dict(manifest_dict)
    after = CorpusManifest.from_dict({
        **manifest_dict,
        "no_acting_text_scripts": {"cardsfolder/g/grizzly_bears.txt": 12},
    })

    assert before.no_acting_text_scripts == {}
    assert after.no_acting_text_scripts == {"cardsfolder/g/grizzly_bears.txt": 12}
    assert before.digest() == after.digest()


def test_a_different_held_out_text_still_moves_the_digest():
    """The exclusion is one named field, not a hole in the hash."""
    assert manifest().digest() != manifest(
        held_out_texts=("gains 4 life",),
    ).digest()


# ── gen-2 fields (FR-053, FR-042; T082, T084) ───────────────────────────

_FIXTURE = Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-manifest-trimmed.json"
_REAL_GEN1 = Path("output/effects/corpus/manifest.json")
#: The digest the gen-1 checkpoint (run 09.17f) recorded for that manifest.
_GEN1_RECORDED_DIGEST = "e3c0c9bb913b9aeda0d07eb587b5445c"


def _canonical_digest(raw: dict) -> str:
    """The digest of a manifest's JSON as written, by the rule feature 023 used."""
    payload = {k: v for k, v in raw.items() if k not in CorpusManifest._OUTSIDE_THE_DIGEST}
    payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def test_a_real_gen1_manifest_keeps_its_digest():
    """Every checkpoint pinned to a gen-1 corpus must stay evaluable (FR-042)."""
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    loaded = CorpusManifest.from_dict(raw)

    assert loaded.holdout_unit == "text"
    assert loaded.game_disjoint_target == 1000
    assert loaded.digest() == _canonical_digest(raw)
    assert set(loaded.as_dict()) == set(raw)


@pytest.mark.skipif(not _REAL_GEN1.exists(), reason="the gen-1 corpus is not on this machine")
def test_the_gen1_corpus_still_matches_the_digest_its_checkpoint_recorded():
    raw = json.loads(_REAL_GEN1.read_text(encoding="utf-8"))
    assert CorpusManifest.from_dict(raw).digest() == _GEN1_RECORDED_DIGEST


def test_gen2_fields_are_left_out_at_their_gen1_values(manifest_dict):
    for name in CorpusManifest._GEN1_DEFAULTS:
        if name != "game_disjoint_target":
            assert name not in manifest_dict


def test_a_manifest_without_the_retired_count_omits_it():
    data = manifest(game_disjoint_target=None).as_dict()
    assert "game_disjoint_target" not in data
    assert CorpusManifest.from_dict(data).game_disjoint_target is None


def test_the_gen2_fields_round_trip_and_move_the_digest():
    gen2 = manifest(
        game_disjoint_target=None,
        holdout_unit="template",
        families={"rewrite": {"Moved": {
            "available": 10, "share": 5, "written": 8, "repeats": 3, "shortfall": 0,
        }}},
        signatures={"rewrite": {"Moved": {"died:changed": 8}}},
        policy_counts={"rewrite": {"on_policy": 6, "off_policy": 2}},
        legality_counts={"blockers": {"real": 4, "what_if": 9}},
        keyword_threshold_games=("run.0-a.9",),
        held_out_texts_without_resolution=("deals 3 damage",),
        held_out_texts_under_five_games={"deals 3 damage": 2},
        game_disjoint_share=0.01,
        game_disjoint_keyword_share=0.15,
        game_disjoint_keywords=("trample",),
        reuse_cap=4,
    )
    restored = CorpusManifest.from_dict(json.loads(json.dumps(gen2.as_dict())))

    assert restored == gen2
    assert gen2.digest() != manifest(game_disjoint_target=None).digest()
