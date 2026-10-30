"""The trainer's configuration, schedule, stopping rule and startup refusals.

Each refusal is a case where continuing would produce a run that looks like a
result and is not one: no checkpoint saved, or weights read against the other
encoding surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.train_effect_model import (
    LEARNING_RATE,
    EarlyStopper,
    TrainEffectModelConfig,
    learning_rate_at,
    warmup_steps,
)
from effects.infrastructure.effect_model_store import (
    DEFAULT_MODEL_OUTPUT,
)


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


class TestModelOutput:
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

    Left alone it produces a training run that writes nothing: `Validator.score`
    over no records returns `nan`, `nan < inf` is False, so `EarlyStopper` never
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


class TestProvenanceRecordsTheCorpus:
    """FR-147: a checkpoint records the curated corpus it read, alongside its
    split, so ``evaluate-effect-model`` can refuse one rebuilt since."""

    def test_a_corpus_runs_provenance_records_the_manifest_digest(self):
        from effects.application.train_effect_model import (
            CorpusFacts,
            TrainEffectModelConfig,
        )

        provenance = CorpusFacts(digest="abc123").provenance(
            TrainEffectModelConfig(corpus="output/effects/corpus"),
        )

        assert provenance.corpus_path == "output/effects/corpus"
        assert provenance.corpus_digest == "abc123"

    def test_a_run_without_a_digest_records_no_corpus(self):
        """Nothing is pinned until a manifest has been read."""
        from effects.application.train_effect_model import (
            CorpusFacts,
            TrainEffectModelConfig,
        )

        provenance = CorpusFacts().provenance(TrainEffectModelConfig())

        assert provenance.corpus_path == ""
        assert provenance.corpus_digest == ""


def test_early_stopping_with_no_patience_never_stops():
    from effects.application.train_effect_model import EarlyStopper

    stopper = EarlyStopper(0)
    stopper.update(1.0)
    for _ in range(50):
        stopper.update(2.0)
    assert not stopper.should_stop
