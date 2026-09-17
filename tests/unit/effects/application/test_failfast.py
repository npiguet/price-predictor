"""The exit-code contract's fail-fast rows (T081).

Each of these is a case where continuing would produce a number that looks like
a result and is not one: a baseline that saw different games, or a cache encoded
against a vocabulary the model never trained on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.train_effect_model import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    VARIANT_FULL,
    VARIANTS,
    ContextCache,
    EarlyStopper,
    MissingSplitError,
    TrainEffectModelConfig,
    learning_rate_at,
    require_split_from,
    variant_masks,
    warmup_steps,
    withheld_keyword_rules,
)
from effects.domain.ability_encoder import AbilityEncoderConfig
from effects.domain.effect_model import EffectModelConfig
from effects.infrastructure.effect_model_store import (
    DEFAULT_MODEL_OUTPUT,
    EffectCheckpoint,
    HashMismatchError,
    SplitMismatchError,
    SplitProvenance,
    content_hash,
    require_same_split,
)

_MODEL_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
)
_ENCODER_CONFIG = AbilityEncoderConfig(vocab_size=32)


def _checkpoint(provenance: SplitProvenance) -> EffectCheckpoint:
    return EffectCheckpoint(
        encoder_config=_ENCODER_CONFIG, model_config=_MODEL_CONFIG,
        encoder_state={}, model_state={}, provenance=provenance,
    )


class TestVariantRequiresASplit:
    def test_a_variant_run_without_split_from_fails_fast(self):
        for variant in VARIANTS:
            if variant == VARIANT_FULL:
                continue
            with pytest.raises(MissingSplitError, match="--split-from"):
                require_split_from(variant, None)

    def test_the_error_says_why_the_comparison_would_be_meaningless(self):
        with pytest.raises(MissingSplitError) as excinfo:
            require_split_from("identity", None)
        assert "saw different games" in str(excinfo.value)

    def test_a_full_run_computes_its_own_split(self):
        require_split_from(VARIANT_FULL, None)

    def test_a_variant_run_with_split_from_is_accepted(self):
        require_split_from("identity", Path("models/effects/effect-model/latest.pt"))

    def test_a_variant_run_with_corpus_instead_is_accepted(self):
        """A shared curated dataset is as firm a baseline as a checkpoint's
        split (FR-146): its manifest enumerates the games directly."""
        require_split_from("identity", None, corpus="output/effects/corpus")

    def test_a_variant_run_with_neither_route_still_fails_fast(self):
        with pytest.raises(MissingSplitError, match="--corpus"):
            require_split_from("identity", None)


class TestVariantCheckpointDisagreement:
    def _provenance(self, **overrides) -> SplitProvenance:
        defaults = {
            "held_out_cards": ("Serra Angel",),
            "card_disjoint_games": ("g1",),
            "game_disjoint_games": ("g2",),
            "vocab_hash": "abc123",
            "keyword_definitions_hash": "def456",
        }
        defaults.update(overrides)
        return SplitProvenance(**defaults)

    def test_a_split_disagreement_names_both_checkpoints(self, tmp_path):
        main = _checkpoint(self._provenance())
        variant = _checkpoint(self._provenance(card_disjoint_games=("g9",)))
        with pytest.raises(SplitMismatchError) as excinfo:
            require_same_split(
                main, variant,
                main_path=tmp_path / "full.pt",
                variant_path=tmp_path / "identity.pt",
            )
        message = str(excinfo.value)
        assert "full.pt" in message and "identity.pt" in message

    def test_a_vocabulary_disagreement_fails_fast(self, tmp_path):
        main = _checkpoint(self._provenance())
        variant = _checkpoint(self._provenance(vocab_hash="different"))
        with pytest.raises(SplitMismatchError, match="vocabulary"):
            require_same_split(
                main, variant,
                main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
            )

    def test_a_keyword_definition_disagreement_fails_fast(self, tmp_path):
        main = _checkpoint(self._provenance())
        variant = _checkpoint(
            self._provenance(keyword_definitions_hash="different"),
        )
        with pytest.raises(SplitMismatchError, match="keyword definitions"):
            require_same_split(
                main, variant,
                main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
            )

    def test_matching_checkpoints_pass(self, tmp_path):
        main = _checkpoint(self._provenance())
        variant = _checkpoint(self._provenance())
        require_same_split(
            main, variant,
            main_path=tmp_path / "a.pt", variant_path=tmp_path / "b.pt",
        )


class TestHashMismatchAtInference:
    def test_a_changed_vocabulary_fails_before_encoding(self, tmp_path):
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
        keywords = tmp_path / "kw.json"
        keywords.write_text("{}", encoding="utf-8")
        provenance = SplitProvenance(
            vocab_hash=content_hash(vocab),
            keyword_definitions_hash=content_hash(keywords),
        )
        vocab.write_text("[PAD]\n[UNK]\nmore\n", encoding="utf-8")
        with pytest.raises(HashMismatchError, match="mean nothing"):
            provenance.verify_hashes(vocab_path=vocab, keyword_path=keywords)


class TestVariantMasks:
    def test_full_masks_nothing(self):
        masks = variant_masks("full")
        assert not (masks.zero_e or masks.zero_state)
        assert not (masks.identity_embedding or masks.taxonomy_embedding)

    def test_identity_replaces_the_encoder(self):
        assert variant_masks("identity").identity_embedding

    def test_state_only_zeroes_every_e(self):
        masks = variant_masks("state-only")
        assert masks.zero_e
        assert not masks.zero_state

    def test_no_state_zeroes_the_ability_tokens_too(self):
        """Otherwise it would read the board through the abilities on it."""
        masks = variant_masks("no-state")
        assert masks.zero_state
        assert masks.zero_e

    def test_taxonomy_swaps_e_for_the_sidecars_api_fields(self):
        assert variant_masks("taxonomy").taxonomy_embedding

    def test_an_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="unknown --variant"):
            variant_masks("oracle-text")


class TestScheduleConstants:
    def test_warmup_is_five_percent_of_the_scheduled_steps(self):
        assert warmup_steps(epochs=40, steps_per_epoch=5_000) == 10_000

    def test_warmup_is_at_least_one_step(self):
        assert warmup_steps(epochs=1, steps_per_epoch=1) == 1

    def test_the_rate_ramps_linearly_then_holds(self):
        warmup = 100
        assert learning_rate_at(0, warmup=warmup) == pytest.approx(
            LEARNING_RATE / warmup
        )
        assert learning_rate_at(49, warmup=warmup) == pytest.approx(
            LEARNING_RATE * 0.5
        )
        assert learning_rate_at(warmup, warmup=warmup) == LEARNING_RATE
        assert learning_rate_at(10_000, warmup=warmup) == LEARNING_RATE

    def test_the_pinned_constants_are_the_specs(self):
        assert LEARNING_RATE == 1e-4
        assert MAX_GRAD_NORM == 1.0


class TestEarlyStopping:
    def test_a_new_best_resets_the_counter(self):
        stopper = EarlyStopper(patience=2)
        assert stopper.update(1.0)
        assert stopper.update(0.9)
        assert stopper.since_best == 0

    def test_it_stops_after_patience_epochs_without_a_best(self):
        stopper = EarlyStopper(patience=2)
        stopper.update(1.0)
        assert not stopper.update(1.1)
        assert not stopper.should_stop
        assert not stopper.update(1.2)
        assert stopper.should_stop

    def test_selection_is_by_card_disjoint_loss(self):
        """The number the model ships on, not in-distribution fit."""
        stopper = EarlyStopper(patience=5)
        stopper.update(0.5)
        assert not stopper.update(0.6)
        assert stopper.best == 0.5


class TestContextCache:
    def test_it_starts_empty_and_due_only_after_enough_batches(self):
        cache = ContextCache(refresh_every=3)
        assert len(cache) == 0
        assert not cache.due_for_refresh()
        for _ in range(3):
            cache.note_batch()
        assert cache.due_for_refresh()

    def test_a_refresh_seeds_unseen_texts_outright(self):
        cache = ContextCache()
        cache.refresh({"bolt": (1.0, 2.0)})
        assert cache.get("bolt") == (1.0, 2.0)

    def test_a_refresh_blends_a_seen_text_rather_than_snapping(self):
        cache = ContextCache(momentum=0.5)
        cache.refresh({"bolt": (0.0, 0.0)})
        cache.refresh({"bolt": (2.0, 4.0)})
        assert cache.get("bolt") == (1.0, 2.0)

    def test_a_refresh_resets_the_counter(self):
        cache = ContextCache(refresh_every=1)
        cache.note_batch()
        assert cache.due_for_refresh()
        cache.refresh({})
        assert not cache.due_for_refresh()

    def test_an_unseen_text_reads_as_absent(self):
        assert ContextCache().get("never encoded") is None


class TestWithheldKeyword:
    def test_no_withholding_hides_nothing(self):
        assert withheld_keyword_rules(None) == (frozenset(), False)

    def test_a_withheld_keyword_is_hidden_and_force_expanded(self):
        hidden, force = withheld_keyword_rules("cascade")
        assert hidden == {"cascade"}
        assert force

    def test_the_token_form_matches_the_vocabularys(self):
        hidden, _ = withheld_keyword_rules("First Strike")
        assert hidden == {"first_strike"}


class TestConfigDefaults:
    def test_the_defaults_are_the_contracts(self):
        config = TrainEffectModelConfig()
        assert config.corpus is None
        assert config.e_dim == 64
        assert config.e_noise == 0.05
        assert config.keyword_expand_p == 0.25
        assert config.context_dropout == 0.15
        assert (config.mlm_weight, config.mlm_mask_prob) == (0.1, 0.15)
        assert config.api_weight == 0.05
        assert config.curriculum_step == 10_000
        assert config.curriculum_epoch == 3
        assert (config.batch_size, config.grad_accum) == (32, 1)
        assert (config.steps_per_epoch, config.epochs, config.patience) == (
            5_000, 40, 5
        )
        assert config.cache_refresh == 500
        assert config.context_cache is False
        assert config.withhold_keyword is None

    def test_the_cards_folders_default_to_both_converted_trees(self):
        assert TrainEffectModelConfig().cards_folders == (
            Path("output/cardsfolder/"), Path("output/tokenscripts/"),
        )

    def test_variant_scripts_and_split_from_default_to_absent(self):
        config = TrainEffectModelConfig()
        assert config.variant_scripts is None
        assert config.split_from is None

    def test_full_writes_to_the_shipping_directory(self):
        assert TrainEffectModelConfig().resolved_model_output() == (
            DEFAULT_MODEL_OUTPUT
        )

    def test_a_variant_writes_under_its_own_name(self):
        config = TrainEffectModelConfig(variant="identity")
        assert config.resolved_model_output() == DEFAULT_MODEL_OUTPUT / "identity"

    def test_an_explicit_model_output_wins(self, tmp_path):
        config = TrainEffectModelConfig(variant="identity", model_output=tmp_path)
        assert config.resolved_model_output() == tmp_path


class TestHoldoutGuard:
    """An empty card-disjoint stratum must stop the run (T160, FR-088b).

    Left alone it produces a training run that writes nothing: `_validate` over
    no records returns `nan`, `nan < inf` is False, so `EarlyStopper` never
    records a best, the checkpoint is never saved, and the run early-stops after
    its patience and exits 0. Hours of GPU time and no artifact.
    """

    def test_an_empty_card_disjoint_stratum_raises(self) -> None:
        from effects.application.train_effect_model import (
            EmptyHoldoutError,
            check_holdout,
        )

        with pytest.raises(EmptyHoldoutError) as raised:
            check_holdout(card_disjoint_records=0, unique_text_records=0, minimum=2000)

        assert "full-strength" in str(raised.value)

    def test_a_thin_stratum_warns_and_names_the_shortfall(self) -> None:
        from effects.application.train_effect_model import check_holdout

        warning = check_holdout(
            card_disjoint_records=5_000, unique_text_records=120, minimum=2000,
        )

        assert warning is not None
        assert "120" in warning and "2000" in warning

    def test_a_healthy_stratum_warns_about_nothing(self) -> None:
        from effects.application.train_effect_model import check_holdout

        assert check_holdout(
            card_disjoint_records=50_000, unique_text_records=9_000, minimum=2000,
        ) is None


class TestCorpusMismatch:
    """``evaluate-effect-model`` refuses a corpus rebuilt since training
    (FR-147) — the same discipline as the vocabulary-hash guard above, for the
    dataset rather than the vocabulary file."""

    def test_evaluate_refuses_a_corpus_rebuilt_since_training(self, tmp_path):
        from effects.application.evaluate_effect_model import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError, match="rebuilt"):
            check_corpus(provenance, actual_digest="something-else")

    def test_evaluate_accepts_a_checkpoint_that_read_no_curated_corpus(self):
        from effects.application.evaluate_effect_model import check_corpus

        check_corpus(SplitProvenance(), actual_digest="anything")  # no raise

    def test_matching_digests_pass(self, tmp_path):
        from effects.application.evaluate_effect_model import check_corpus

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="same",
        )
        check_corpus(provenance, actual_digest="same")  # no raise

    def test_the_error_names_the_corpus_path_and_both_digests(self, tmp_path):
        from effects.application.evaluate_effect_model import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError) as excinfo:
            check_corpus(provenance, actual_digest="something-else")
        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert "trained-against-this" in message
        assert "something-else" in message


class TestCorpusPathResolution:
    """``--corpus`` on ``evaluate-effect-model`` defaults to what the
    checkpoint recorded, an explicit value overriding it (FR-147). Unlike
    ``resolve_inference_paths``, which hash-checks every resolved path
    afterwards whether it came from the checkpoint or an override, an
    override here is discarded — with a warning — when the checkpoint
    recorded no digest to check it against."""

    _CHECKPOINT = Path("models/effects/effect-model/latest.pt")

    def test_defaults_to_the_checkpoints_recorded_path(self):
        from effects.application.evaluate_effect_model import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=None, checkpoint=self._CHECKPOINT,
        ) == Path("output/effects/corpus")

    def test_an_explicit_corpus_overrides(self, tmp_path):
        from effects.application.evaluate_effect_model import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=tmp_path, checkpoint=self._CHECKPOINT,
        ) == tmp_path

    def test_returns_none_when_the_checkpoint_read_no_curated_corpus(self, tmp_path):
        """No digest recorded means nothing to check, whatever --corpus says."""
        from effects.application.evaluate_effect_model import resolve_corpus_path

        assert resolve_corpus_path(
            SplitProvenance(), corpus=None, checkpoint=self._CHECKPOINT,
        ) is None
        assert resolve_corpus_path(
            SplitProvenance(), corpus=tmp_path, checkpoint=self._CHECKPOINT,
        ) is None

    def test_an_override_on_a_legacy_checkpoint_warns_that_it_is_ignored(
        self, tmp_path, caplog,
    ):
        """Silence here would be the flag-with-no-effect-and-no-signal
        pattern this repo keeps paying for: refusing would be too strong (it
        would break a batch-eval script passing --corpus uniformly across a
        mix of curated and legacy checkpoints), and honouring it would be
        meaningless (there is no digest to compare freshness against)."""
        from effects.application.evaluate_effect_model import resolve_corpus_path

        with caplog.at_level("WARNING"):
            resolve_corpus_path(
                SplitProvenance(), corpus=tmp_path, checkpoint=self._CHECKPOINT,
            )
        assert str(self._CHECKPOINT) in caplog.text
        assert "no curated corpus" in caplog.text

    def test_a_curated_checkpoints_override_warns_about_nothing(
        self, tmp_path, caplog,
    ):
        """A recorded digest means the override has something to check
        against, isolating corpus_digest as the one thing that decides
        whether the warning fires."""
        from effects.application.evaluate_effect_model import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        with caplog.at_level("WARNING"):
            resolve_corpus_path(
                provenance, corpus=tmp_path, checkpoint=self._CHECKPOINT,
            )
        assert caplog.text == ""


class TestCorpusSurface:
    """A curated dataset's rarity table is keyed on one encoding surface.

    ``CorpusManifest.surface`` records which, and nothing read it: the
    ``--corpus`` branch took its surface from ``--vocab-path`` alone. A
    prose-built dataset read with ``--vocab-path
    models/effects/vocab-script.txt`` misses on every rarity lookup, and
    ``sample_weights`` reads a miss as "a shard collected after the dataset was
    built" and falls back to the resident shard's own count — reverting
    silently to exactly the per-shard weighting FR-141 exists to replace.
    """

    def test_a_vocabulary_on_the_other_surface_is_refused(self) -> None:
        from effects.application.train_effect_model import (
            SurfaceMismatchError,
            require_matching_surface,
        )

        with pytest.raises(SurfaceMismatchError, match="script"):
            require_matching_surface(
                manifest_surface="script",
                vocab_path=Path("models/effects/vocab.txt"),
                corpus="output/effects/corpus",
            )

    def test_the_surface_the_dataset_was_built_on_is_accepted(self) -> None:
        from effects.application.train_effect_model import require_matching_surface

        require_matching_surface(
            manifest_surface="script",
            vocab_path=Path("models/effects/vocab-script.txt"),
            corpus="output/effects/corpus",
        )
        require_matching_surface(
            manifest_surface="prose",
            vocab_path=Path("models/effects/vocab.txt"),
            corpus="output/effects/corpus",
        )


class TestRarityCoverage:
    """How much of a shard the rarity table actually names.

    Reported once per run because a table that matches nothing is invisible
    otherwise: every weight silently falls back to the resident shard's count,
    and the run looks exactly like a healthy one.
    """

    def test_it_counts_distinct_texts_rather_than_records(self) -> None:
        from effects.application.train_effect_model import rarity_coverage

        texts = ["a", "a", "a", "b", None, None]
        assert rarity_coverage(texts, {"a": 3}) == (1, 2)

    def test_a_table_naming_nothing_reads_as_zero_of_the_texts_present(self) -> None:
        from effects.application.train_effect_model import rarity_coverage

        assert rarity_coverage(["a", "b"], {"other": 1}) == (0, 2)

    def test_a_shard_with_no_acting_text_at_all_looks_up_nothing(self) -> None:
        from effects.application.train_effect_model import rarity_coverage

        assert rarity_coverage([None, None], {"a": 1}) == (0, 0)


class TestEmptyHoldoutIsAnExitCodeNotATraceback:
    """An empty card-disjoint stratum is a corpus problem, not a crash.

    ``check_holdout`` raises it from inside the training loop, and the CLI let
    it out as a traceback — the one exit-code row an operator meets after
    building a corpus without a full-strength run in it.
    """

    def test_it_is_logged_and_reported_as_two(self, monkeypatch, caplog):
        import argparse
        import logging

        from effects.application import train_effect_model
        from effects.infrastructure import cli

        def refuse(config):
            raise train_effect_model.EmptyHoldoutError(
                "The card-disjoint stratum holds no records"
            )

        monkeypatch.setattr(train_effect_model, "run", refuse)
        monkeypatch.setattr(cli, "train_config_from", lambda args: None)
        args = argparse.Namespace(split_from=None, variant="full", corpus="x")

        with caplog.at_level(logging.ERROR):
            assert cli.run_train_effect_model(args) == 2

        assert "card-disjoint stratum holds no records" in caplog.text
