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
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.infrastructure.effect_model_store import (
    DEFAULT_MODEL_OUTPUT,
    VARIANT_FULL,
    VARIANTS,
    EffectCheckpoint,
    EffectModelStore,
    HashMismatchError,
    SplitMismatchError,
    SplitProvenance,
    content_hash,
    filter_training_only,
    model_output_for,
    require_same_split,
    resolve_inference_paths,
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


class TestSplitRoundTrip:
    def test_the_split_survives_a_save_and_load(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        loaded = store.load()
        assert loaded.provenance == _provenance(files)

    def test_both_strata_come_back(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        provenance = store.load().provenance
        assert provenance.card_disjoint_games == ("g1", "g2")
        assert provenance.game_disjoint_games == ("g3",)
        assert provenance.validation_games == {"g1", "g2", "g3"}

    def test_the_held_out_card_list_comes_back(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        assert store.load().provenance.held_out_cards == ("Serra Angel",)

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


class TestHashChecking:
    def test_matching_hashes_pass(self, files):
        vocab, keywords = files
        _provenance(files).verify_hashes(
            vocab_path=vocab, keyword_path=keywords,
        )

    def test_a_changed_vocabulary_fails_fast(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        vocab.write_text("[PAD]\n[UNK]\ndifferent\n", encoding="utf-8")
        with pytest.raises(HashMismatchError, match="vocabulary"):
            provenance.verify_hashes(vocab_path=vocab, keyword_path=keywords)

    def test_a_changed_keyword_file_fails_fast(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        keywords.write_text('{"Flying": {}}', encoding="utf-8")
        with pytest.raises(HashMismatchError, match="keyword definitions"):
            provenance.verify_hashes(vocab_path=vocab, keyword_path=keywords)

    def test_the_error_names_both_hashes(self, files):
        vocab, keywords = files
        provenance = _provenance(files)
        vocab.write_text("changed", encoding="utf-8")
        with pytest.raises(HashMismatchError) as excinfo:
            provenance.verify_hashes(vocab_path=vocab, keyword_path=keywords)
        assert provenance.vocab_hash[:12] in str(excinfo.value)

    def test_a_missing_file_hashes_as_empty_rather_than_raising(self, tmp_path):
        assert content_hash(tmp_path / "absent.json") == ""

    def test_an_unrecorded_hash_is_not_checked(self, files):
        """A run with no keyword file is degraded, not broken."""
        vocab, keywords = files
        SplitProvenance(vocab_hash=content_hash(vocab)).verify_hashes(
            vocab_path=vocab, keyword_path=keywords,
        )


class TestVariantIsolation:
    def test_full_writes_to_the_shipping_directory(self):
        assert model_output_for(VARIANT_FULL) == DEFAULT_MODEL_OUTPUT

    def test_every_other_variant_writes_under_its_own_name(self):
        for variant in VARIANTS:
            if variant == VARIANT_FULL:
                continue
            assert model_output_for(variant) == DEFAULT_MODEL_OUTPUT / variant

    def test_the_four_baselines_are_all_named(self):
        assert set(VARIANTS) == {
            "full", "identity", "state-only", "no-state", "taxonomy",
        }

    def test_a_matching_variant_checkpoint_is_accepted(self, tmp_path, files):
        main = _checkpoint(files)
        variant = _checkpoint(files, variant="identity")
        require_same_split(
            main, variant, main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
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
                main_path=tmp_path / "main.pt", variant_path=tmp_path / "var.pt",
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
                main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
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
                main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
            )


class TestTrainingOnlyHeadsAreFiltered:
    def test_the_saved_artifact_carries_no_training_only_head(self, tmp_path, files):
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(_checkpoint(files))
        keys = store.load().model_state.keys()
        for name in EffectModel.TRAINING_ONLY_HEADS:
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
        assert store.latest_path().exists()
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
        table = torch.nn.Embedding(8, 4).state_dict()
        store = EffectModelStore(tmp_path / "effect-model")
        store.save(replace(
            _checkpoint(files),
            variant="identity",
            extra={"identity_table": table},
        ))
        loaded = store.load()
        assert "identity_table" in loaded.extra
        assert torch.equal(
            loaded.extra["identity_table"]["weight"], table["weight"],
        )

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

    def test_an_explicit_path_overrides(self, tmp_path, files):
        chosen = tmp_path / "other-vocab.txt"
        vocab_path, _ = resolve_inference_paths(
            _provenance(files), vocab_path=chosen, keyword_path=None,
        )
        assert vocab_path == chosen

    def test_an_override_is_still_hash_checked(self, tmp_path, files):
        """Overriding to a different file must fail, not silently re-encode."""
        _vocab, keywords = files
        other = tmp_path / "other-vocab.txt"
        other.write_text("[PAD]\n[UNK]\nextra\n", encoding="utf-8")
        provenance = _provenance(files)
        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=other, keyword_path=None,
        )
        with pytest.raises(HashMismatchError):
            provenance.verify_hashes(
                vocab_path=vocab_path, keyword_path=keyword_path,
            )
