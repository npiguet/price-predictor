"""The checkpoint contract (T082).

A checkpoint has to describe the model *as trained*, not as remembered: its
split, the vocabulary and keyword files it actually read, and the keyword it was
trained without. Everything the evaluator does rests on reading those back
rather than recomputing them against a corpus that has grown since.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from effects.domain.ability_encoder import AbilityEncoderConfig
from effects.domain.aux_heads import TRAINING_ONLY_MODULES
from effects.domain.baselines import (
    BASELINES,
    Baseline,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.split_provenance import SplitProvenance
from effects.domain.text_holdout import HoldoutUnit
from effects.infrastructure.effect_model_store import (
    DEFAULT_MODEL_OUTPUT,
    EffectCheckpoint,
    EffectModelStore,
    HashMismatchError,
    SplitMismatchError,
    content_hash,
    filter_training_only,
    load_for_inference,
    model_output_for,
    require_same_split,
    resolve_inference_paths,
    verify_hashes,
)

_MODEL_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
    e_dim=4, vocab_size=32, n_api_types=5, n_param_keys=7,
    d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)
_ENCODER_CONFIG = AbilityEncoderConfig(
    vocab_size=32, e_dim=4, d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)


@pytest.fixture
def files(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    keywords = tmp_path / "keyword-definitions.json"
    keywords.write_text("{}", encoding="utf-8")
    return vocab, keywords


def _provenance(files, **overrides) -> SplitProvenance:
    vocab, keywords = files
    defaults = {
        "held_out_cards": ("Serra Angel",),
        "card_disjoint_games": ("g1", "g2"),
        "game_disjoint_games": ("g3",),
        "vocab_path": str(vocab),
        "keyword_definitions_path": str(keywords),
        "vocab_hash": content_hash(vocab),
        "keyword_definitions_hash": content_hash(keywords),
        "withheld_keyword": "cascade",
    }
    defaults.update(overrides)
    return SplitProvenance(**defaults)


def _checkpoint(files, **overrides) -> EffectCheckpoint:
    torch.manual_seed(0)
    model = EffectModel(_MODEL_CONFIG)
    defaults = {
        "encoder_config": _ENCODER_CONFIG,
        "model_config": _MODEL_CONFIG,
        "encoder_state": {"token_embedding.weight": torch.zeros(32, 16)},
        "model_state": model.state_dict(),
        "provenance": _provenance(files),
    }
    defaults.update(overrides)
    return EffectCheckpoint(**defaults)


def test_two_saves_in_one_second_keep_both_checkpoints(tmp_path, files, monkeypatch):
    """A fast epoch under --save-every-epoch saves twice in one second."""
    from datetime import UTC, datetime

    import effects.infrastructure.effect_model_store as store_module

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(store_module, "datetime", _Frozen)
    store = EffectModelStore(tmp_path / "effect-model")
    saved = [store.save(_checkpoint(files)).name for _ in range(3)]
    assert saved == ["20261029-120000.pt", "20261029-120000-1.pt", "20261029-120000-2.pt"]


class TestSplitRoundTrip:
    def test_the_split_survives_a_save_and_load(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        loaded = store.load()
        assert loaded.provenance == _provenance(files)

    def test_the_withheld_keyword_is_checkpoint_state_not_a_flag(
        self, tmp_path, files,
    ):
        """The zero-shot check must measure the model that was trained."""
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        assert store.load().provenance.withheld_keyword == "cascade"

    def test_a_run_that_withheld_nothing_records_none(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(
            files, provenance=_provenance(files, withheld_keyword=None),
        ))
        assert store.load().provenance.withheld_keyword is None

    def test_the_corpus_provenance_comes_back(self, tmp_path, files):
        """FR-147: a `--corpus` run's manifest path and digest are checkpoint
        state, the same as the split and the vocabulary hashes above."""
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files, provenance=_provenance(
            files, corpus_path="output/effects/corpus", corpus_digest="abc123",
        )))
        provenance = store.load().provenance
        assert provenance.corpus_path == "output/effects/corpus"
        assert provenance.corpus_digest == "abc123"


class TestCorpusProvenance:
    """FR-147: a checkpoint pins the curated corpus it read.

    A rebuild is a different split, so scoring the gates against the rebuilt
    one would score them partly on games the model trained on — the same
    failure the held-out ``game_id`` sets exist to prevent, arriving through
    the corpus instead of the split.
    """

    def test_split_provenance_round_trips_the_corpus_and_holdout_it_read(self):
        """The holdout flags too (T169, FR-134): a depleted corpus was composed
        against them, and a run inheriting the split must be able to tell."""
        original = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
            holdout_permille=20, holdout_max_carriers=8,
            holdout_unit=HoldoutUnit.TEMPLATE,
        )
        restored = SplitProvenance.from_dict(original.as_dict())
        assert restored == original
        assert restored.holdout_unit is HoldoutUnit.TEMPLATE

    def test_the_text_unit_is_written_as_no_field_and_read_back(self):
        """FR-042: a checkpoint that records no unit reads as it trained."""
        text = SplitProvenance(held_out_cards=("Alpha",))
        assert "holdout_unit" not in text.as_dict()
        assert SplitProvenance.from_dict(text.as_dict()).holdout_unit is HoldoutUnit.TEXT

    def test_a_checkpoint_saved_before_curated_corpora_still_loads(self):
        assert SplitProvenance.from_dict({}).corpus_digest == ""

    def test_an_empty_provenance_reads_as_every_default(self):
        assert SplitProvenance.from_dict({}) == SplitProvenance()

    def test_stored_lists_read_back_as_tuples(self):
        provenance = SplitProvenance.from_dict({
            "held_out_cards": ["Shock"], "card_disjoint_games": ["g1"],
            "holdout_permille": 20,
        })
        assert provenance.held_out_cards == ("Shock",)
        assert provenance.card_disjoint_games == ("g1",)
        assert provenance.holdout_permille == 20


class TestHashChecking:
    def test_matching_hashes_pass(self, files):
        vocab, keywords = files
        verify_hashes(
            _provenance(files), vocab_path=vocab, keyword_path=keywords,
        )

    def test_a_changed_vocabulary_fails_fast(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        vocab.write_text("[PAD]\n[UNK]\ndifferent\n", encoding="utf-8")
        with pytest.raises(HashMismatchError, match="vocabulary"):
            verify_hashes(provenance, vocab_path=vocab, keyword_path=keywords)

    def test_a_changed_keyword_file_fails_fast(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        keywords.write_text('{"Flying": {}}', encoding="utf-8")
        with pytest.raises(HashMismatchError, match="keyword definitions"):
            verify_hashes(provenance, vocab_path=vocab, keyword_path=keywords)

    def test_the_error_names_both_hashes(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        vocab.write_text("changed", encoding="utf-8")
        with pytest.raises(HashMismatchError) as excinfo:
            verify_hashes(provenance, vocab_path=vocab, keyword_path=keywords)
        assert provenance.vocab_hash[:12] in str(excinfo.value)

    def test_a_missing_file_hashes_as_empty_rather_than_raising(self, tmp_path):
        assert content_hash(tmp_path / "absent.json") == ""

    def test_an_unrecorded_hash_is_not_checked(self, files):
        """A run with no keyword file is degraded, not broken."""
        vocab, keywords = files
        verify_hashes(
            SplitProvenance(vocab_hash=content_hash(vocab)),
            vocab_path=vocab, keyword_path=keywords,
        )


class TestVariantIsolation:
    def test_full_writes_to_the_shipping_directory(self):
        assert model_output_for(Baseline.FULL) == DEFAULT_MODEL_OUTPUT

    def test_every_other_variant_writes_under_its_own_name(self):
        for variant in BASELINES:
            if variant == Baseline.FULL:
                continue
            assert model_output_for(variant) == DEFAULT_MODEL_OUTPUT / variant

    def test_a_matching_variant_checkpoint_is_accepted(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(files, variant="identity")
        require_same_split(
            main, variant, main_path=tmp_path / "a.pt", baseline_path=tmp_path / "b.pt",
        )

    def test_a_different_split_fails_fast_naming_both(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(
            files, variant="identity",
            provenance=_provenance(files, game_disjoint_games=("g9",)),
        )
        with pytest.raises(SplitMismatchError) as excinfo:
            require_same_split(
                main, variant,
                main_path=tmp_path / "main.pt", baseline_path=tmp_path / "var.pt",
            )
        assert "main.pt" in str(excinfo.value)
        assert "var.pt" in str(excinfo.value)

    def test_a_different_vocabulary_hash_fails_fast(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(
            files, variant="identity",
            provenance=_provenance(files, vocab_hash="deadbeef"),
        )
        with pytest.raises(SplitMismatchError, match="vocabulary"):
            require_same_split(
                main, variant,
                main_path=tmp_path / "a.pt", baseline_path=tmp_path / "b.pt",
            )

    def test_a_different_keyword_definition_hash_fails_fast(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(
            files, variant="identity",
            provenance=_provenance(files, keyword_definitions_hash="deadbeef"),
        )
        with pytest.raises(SplitMismatchError, match="keyword definitions"):
            require_same_split(
                main, variant,
                main_path=tmp_path / "a.pt", baseline_path=tmp_path / "b.pt",
            )

    def test_a_different_held_out_card_list_fails_fast(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(
            files, variant="identity",
            provenance=_provenance(files, held_out_cards=("Shock",)),
        )
        with pytest.raises(SplitMismatchError):
            require_same_split(
                main, variant,
                main_path=tmp_path / "a.pt", baseline_path=tmp_path / "b.pt",
            )


class TestTrainingOnlyHeadsAreFiltered:
    def test_the_saved_artifact_carries_no_training_only_head(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        keys = store.load().model_state.keys()
        for name in TRAINING_ONLY_MODULES:
            assert not any(key.startswith(f"{name}.") for key in keys), name

    def test_the_shipped_heads_survive(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        keys = store.load().model_state.keys()
        for name in ("per_entity_head", "created_objects_head", "verdict_head"):
            assert any(key.startswith(f"{name}.") for key in keys), name

    def test_the_trunk_survives(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        assert any(k.startswith("trunk.") for k in store.load().model_state)

    def test_filtering_leaves_an_unrelated_key_alone(self):
        state = {"trunk.weight": 1, "mlm_head.weight": 2, "mlm_headache": 3}
        filtered = filter_training_only(state)
        assert "trunk.weight" in filtered
        assert "mlm_head.weight" not in filtered
        assert "mlm_headache" in filtered  # prefix match is on "mlm_head."


class TestStoreFiles:
    def test_a_save_writes_a_timestamp_and_refreshes_latest(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        path = store.save(_checkpoint(files))
        assert path.exists()
        assert store.latest_path().read_bytes() == path.read_bytes()
        assert path.name != "latest.pt"

    def test_loading_a_missing_checkpoint_says_what_to_run(self, tmp_path):
        store = EffectModelStore(tmp_path / "never-trained")
        with pytest.raises(FileNotFoundError, match="train-effect-model"):
            store.load()

    def test_the_configs_round_trip(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        loaded = store.load()
        assert loaded.model_config == _MODEL_CONFIG
        assert loaded.encoder_config == _ENCODER_CONFIG


class TestExtraRoundTrip:
    """The identity baseline's ``e`` lives in an embedding table, not in the
    encoder. A checkpoint that dropped it would reload as random vectors and
    hand gate 1 a margin nobody earned — and gate 1 blocks shipping."""

    def test_a_variants_extra_payload_survives_a_round_trip(
        self, tmp_path, files,
    ):
        from effects.application.surface_batching import IdentityTable

        table = IdentityTable(["flying", "trample"], 4)
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(replace(
            _checkpoint(files),
            variant="identity",
            extra={"identity_table": table.state()},
        ))
        loaded = IdentityTable.from_state(store.load().extra["identity_table"])
        assert loaded.texts == table.texts
        assert torch.equal(loaded.embedding.weight, table.embedding.weight)

    def test_a_checkpoint_with_no_extra_loads_an_empty_one(
        self, tmp_path, files,
    ):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        assert store.load().extra == {}


class TestInferencePathResolution:
    def test_the_paths_default_to_what_the_checkpoint_recorded(self, files):
        vocab, keywords = files
        resolved = resolve_inference_paths(
            _provenance(files), vocab_path=None, keyword_path=None,
        )
        assert resolved == (vocab, keywords)

    def test_the_resolved_paths_pass_the_hash_check(self, files):
        provenance = _provenance(files)
        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=None, keyword_path=None,
        )
        verify_hashes(provenance, vocab_path=vocab_path, keyword_path=keyword_path)

    def test_an_explicit_path_overrides(self, tmp_path, files):
        _vocab, keywords = files
        chosen = tmp_path / "other-vocab.txt"
        vocab_path, keyword_path = resolve_inference_paths(
            _provenance(files), vocab_path=chosen, keyword_path=None,
        )
        assert vocab_path == chosen
        assert keyword_path == keywords

    def test_an_override_is_still_hash_checked(self, tmp_path, files):
        """Overriding to a different file must fail, not silently re-encode."""
        other = tmp_path / "other-vocab.txt"
        other.write_text("[PAD]\n[UNK]\nextra\n", encoding="utf-8")
        provenance = _provenance(files)
        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=other, keyword_path=None,
        )
        with pytest.raises(HashMismatchError):
            verify_hashes(
                provenance, vocab_path=vocab_path, keyword_path=keyword_path,
            )


class TestLoadForInference:
    """Loading for inference resolves and hash-checks in one step, so no
    inference command can resolve its files and forget to verify them."""

    def test_it_returns_the_checkpoint_and_its_verified_paths(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        path = store.save(_checkpoint(files))
        loaded = load_for_inference(path)
        assert (loaded.vocab_path, loaded.keyword_path) == files
        assert loaded.checkpoint.provenance.withheld_keyword == "cascade"

    def test_a_file_that_moved_since_training_fails_the_load(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        path = store.save(_checkpoint(files))
        files[0].write_text("[PAD]\nextra\n", encoding="utf-8")
        with pytest.raises(HashMismatchError):
            load_for_inference(path)

    def test_an_override_is_verified_too(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        path = store.save(_checkpoint(files))
        other = tmp_path / "other-keywords.json"
        other.write_text('{"flying": "x"}', encoding="utf-8")
        with pytest.raises(HashMismatchError):
            load_for_inference(path, keyword_path=other)

_GEN1_CHECKPOINT = (
    __import__("pathlib").Path(__file__).parents[4]
    / "models" / "effects" / "runs" / "2026-09-17-full-textless-corpus" / "latest.pt"
)


class TestGen2Checkpoint:
    """What a gen-2 checkpoint records and drops (T101, FR-059, FR-061, FR-063)."""

    def test_the_sweep_flags_and_roots_round_trip(self, tmp_path, files):
        """Spec Story 5 scenario 6: unit, size, noise ratio, value weight."""
        encoder = AbilityEncoderConfig(vocab_size=32, e_dim=4, d_model=32, n_layers=2)
        settings = {"e_noise": 0.2, "value_weight": 0.05, "mlm_weight": 0.1,
                    "mlm_mask_prob": 0.15, "api_weight": 0.05,
                    "api_types": ["Draw"], "param_keys": ["NumCards"]}
        store = EffectModelStore(tmp_path)
        store.save(_checkpoint(
            files, encoder_config=encoder,
            encoder_state={},
            provenance=_provenance(files, holdout_unit="template"),
            cards_folders=("output/gen1-cardsfolder", "output/gen1-tokenscripts"),
            training_settings=settings,
        ))
        loaded = store.load()
        assert (loaded.encoder_config.n_layers, loaded.encoder_config.d_model) == (2, 32)
        assert loaded.encoder_config.ff_dim == 128  # follows d_model x 4
        assert loaded.provenance.holdout_unit == "template"
        assert loaded.training_settings == settings
        assert loaded.cards_folders == (
            "output/gen1-cardsfolder", "output/gen1-tokenscripts",
        )

    def test_an_encoder_config_recording_e_noise_loads_with_it_ignored(self, tmp_path, files):
        store = EffectModelStore(tmp_path)
        store.save(_checkpoint(files))
        path = store.latest_path()
        payload = torch.load(path, weights_only=False)
        payload["encoder_config"]["e_noise"] = 0.05
        torch.save(payload, path)
        assert store.load().encoder_config == _ENCODER_CONFIG

    def test_a_config_recording_no_size_builds_at_four_layers_and_256(self):
        from effects.infrastructure.effect_model_store import encoder_config_from

        config = encoder_config_from({"vocab_size": 32, "e_dim": 64, "e_noise": 0.05})
        assert (config.n_layers, config.d_model, config.ff_dim) == (4, 256, 1024)

    @pytest.mark.skipif(not _GEN1_CHECKPOINT.exists(), reason="gen-1 checkpoint absent")
    def test_the_gen1_checkpoint_loads_at_its_recorded_size(self):
        """Spec Story 5 scenario 8, on the real gen-1 payload."""
        from effects.domain.ability_encoder import AbilityEncoder

        checkpoint = EffectModelStore(_GEN1_CHECKPOINT.parent).load(_GEN1_CHECKPOINT)
        config = checkpoint.encoder_config
        assert (config.n_layers, config.d_model) == (4, 256)
        encoder = AbilityEncoder(config)
        encoder.load_state_dict(checkpoint.encoder_state)
        model = EffectModel(checkpoint.model_config)
        missing, unexpected = model.load_state_dict(checkpoint.model_state, strict=False)
        assert not unexpected
        assert set(missing) <= {
            key for key in model.state_dict()
            if key.startswith(("option_kind_embedding.", *(
                f"{name}." for name in TRAINING_ONLY_MODULES
            )))
        }
        assert checkpoint.training_settings == {} and checkpoint.cards_folders == ()
