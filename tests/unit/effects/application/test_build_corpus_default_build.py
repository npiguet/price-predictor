"""What the default build writes: withheld games, held-out texts, strata, shards."""

from __future__ import annotations

from effects.domain.records import (
    RecordKind,
)
from effects.infrastructure.shard_io import (
    read_records,
    read_shard,
)
from tests.unit.effects.application.build_corpus_support import (
    _HELD_OUT_KEY,
    _HELD_OUT_TEXT,
)
from tests.unit.effects.record_builders import held_out_cards


def test_no_training_record_belongs_to_a_withheld_game(default_build):
    store = default_build
    manifest = store.load()
    withheld = set(manifest.card_disjoint_games) | set(manifest.game_disjoint_games)
    assert withheld
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert not trained & withheld


def test_no_training_record_names_a_held_out_card(default_build):
    """FR-088: the exclusion is by game, and every held-out game is dropped."""
    from effects.domain.split import record_names_held_out_card

    manifest = default_build.load()
    held_out = held_out_cards(manifest.held_out_cards)
    for record in read_records(default_build.training_dir):
        assert not record_names_held_out_card(record, held_out)


def test_the_manifest_records_the_held_out_texts(default_build):
    """Fix round 1, Finding 1: ``build()`` must populate ``held_out_texts``.

    A ``--corpus`` training run's stratum-sizing guard (``check_holdout``)
    reads ``HeldOutCards.texts`` alone, built from this field — not from
    ``held_out_cards``. A manifest that recorded only the held-out cards would
    make every ``--corpus`` run see an empty text set and report the
    card-disjoint stratum as empty regardless of the dataset's actual health.
    """
    assert default_build.load().held_out_texts == (_HELD_OUT_TEXT,)


def test_a_validation_stratum_keeps_whole_games(default_build):
    """A probe and the combat record it mirrors land together (FR-136)."""
    store = default_build
    mirrored = 0
    for directory in (store.card_disjoint_dir, store.game_disjoint_dir):
        records = list(read_records(directory))
        by_id = {r.record_id for r in records}
        for record in records:
            if record.mirror_of is not None:
                mirrored += 1
                assert record.mirror_of in by_id
    assert mirrored, "the fixture must contain a mirrored probe or this proves nothing"


def test_the_cap_trims_a_text_that_exceeds_it(tmp_path, shared_corpus):
    manifest = shared_corpus.build(tmp_path / "curated", text_cap=2).load()
    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) > 0


def test_two_builds_of_one_corpus_agree(tmp_path, shared_corpus, default_build):
    second = shared_corpus.build(tmp_path / "again")

    assert default_build.load().digest() == second.load().digest()
    assert (
        [r.record_id for r in read_records(default_build.training_dir)]
        == [r.record_id for r in read_records(second.training_dir)]
    )


def test_the_manifest_records_per_stratum_counts_and_unique_texts(default_build):
    """FR-143/FR-145: per-class counts alone leave both validation strata
    with no unique-text figure anywhere in the manifest. A test that only
    checked the two fields exist would not catch them being populated empty,
    so this checks actual values against strata ``a_corpus`` is built to
    populate: the tainted game (card-disjoint) and whichever clean game the
    seeded game-disjoint draw takes both carry a resolvable ability.
    """
    manifest = default_build.load()
    assert manifest.per_stratum["training"] > 0
    assert manifest.unique_texts["training"] > 0
    # The counts are what each stratum's directory holds, not merely positive.
    for stratum, directory in (("training", default_build.training_dir),
                               ("card-disjoint", default_build.card_disjoint_dir),
                               ("game-disjoint", default_build.game_disjoint_dir)):
        written = {r.record_id for r in read_records(directory)}
        assert manifest.per_stratum[stratum] == len(written), stratum
    assert "card-disjoint" in manifest.per_stratum
    assert "game-disjoint" in manifest.per_stratum
    assert manifest.unique_texts["card-disjoint"] > 0
    assert manifest.unique_texts["game-disjoint"] > 0


# ── the delivered mixture is the mixture the manifest records ─────────


def test_no_empty_output_shard_is_written(default_build):
    """One output shard per source shard per stratum, written unconditionally,
    turned 701 source shards into 2,103 files, most of them empty.

    They are not free. `train-effect-model --corpus` lists both validation
    directories and reads every shard in them before the first training step,
    logging a line each; an empty *training* shard drawn into an epoch
    forfeits its share of the step budget, because `_train_on_shard` returns
    early on a shard with nothing in it.
    """
    store = default_build
    written = [
        shard
        for directory in (
            store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir,
        )
        for shard in directory.glob("*.jsonl.gz")
    ]
    assert written, "the fixture must write something or this proves nothing"
    empty = [shard for shard in written if not any(True for _ in read_shard(shard))]
    assert not empty, f"empty shards written: {[str(s) for s in empty]}"


def test_the_gate_one_slice_holds_held_out_resolutions_of_card_disjoint_games(default_build):
    store = default_build
    manifest = store.load()
    slice_ = list(read_records(store.gate_one_dir))
    assert slice_, "g-tainted resolves the held-out text, so the slice is not empty"
    assert all(r.kind is RecordKind.RESOLUTION for r in slice_)
    assert all(r.game_id in manifest.card_disjoint_games for r in slice_)
    assert all(_HELD_OUT_KEY in (r.ability or ()) for r in slice_)
    card_disjoint_ids = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert {r.record_id for r in slice_} <= card_disjoint_ids
    assert manifest.per_stratum["gate-one"] == len(slice_)


def test_games_are_reported_per_source_directory(shared_corpus, default_build):
    """Shards live under depleted/ and full-strength/ in the fixture."""
    manifest = default_build.load()
    assert set(manifest.games_by_source) == set(shared_corpus.source_dirs)
    assert sum(manifest.games_by_source.values()) == shared_corpus.game_count
    assert all(manifest.held_out_games_by_source.get(s, 0) <= n
               for s, n in manifest.games_by_source.items())


def test_output_shards_are_repacked_and_the_parts_removed(tmp_path, shared_corpus):
    # ``game_disjoint_target=1`` as everywhere else in this file: the default
    # 1000 takes all three clean games into the game-disjoint stratum, leaving
    # the training output — the one this test reads back — empty.
    store = shared_corpus.build(tmp_path / "out", shard_records=2)
    assert not store.parts_dir.exists()
    names = sorted(p.name for p in store.training_dir.glob("*.jsonl.gz"))
    assert names and names[0] == "shard-00001.jsonl.gz"
    assert store.load().shard_records == 2


def test_the_build_writes_a_validation_sample_per_stratum(tmp_path, shared_corpus):
    store = shared_corpus.build(tmp_path / "out", validation_sample=4)
    for stratum in ("card-disjoint", "game-disjoint"):
        assert store.sample_path(stratum).exists()
        assert 0 < sum(1 for _ in read_shard(store.sample_path(stratum))) <= 4 * 8
    assert store.load().validation_sample == 4
