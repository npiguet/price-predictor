"""The trainer's configuration, schedule, stopping rule and startup refusals.

Each refusal is a case where continuing would produce a run that looks like a
result and is not one: no checkpoint saved, or weights read against the other
encoding surface.
"""

from __future__ import annotations

import pytest

from effects.application.train_effect_model import TrainEffectModelConfig
from effects.application.training_schedule import (
    LEARNING_RATE,
    ControlScoreStopper,
    StopperStateError,
    learning_rate_at,
    warmup_steps,
)
from effects.domain.baselines import Baseline
from tests.unit.effects.builders import write_vocabulary


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

    def test_lr_scale_multiplies_the_whole_schedule(self):
        for step in (0, 49, 100, 10_000):
            assert learning_rate_at(step, warmup=100, scale=0.5) == (
                learning_rate_at(step, warmup=100) * 0.5
            )


class TestControlScoreStopper:
    """FR-069: higher is better, and patience counts epochs without a new best."""

    def test_a_higher_control_score_is_a_new_best_and_resets_the_counter(self):
        stopper = ControlScoreStopper(patience=2)
        assert stopper.best_score is None
        assert stopper.update(0.1)
        assert stopper.update(0.3)
        assert stopper.since_best == 0
        assert stopper.best_score == 0.3

    def test_it_stops_after_patience_epochs_without_a_best(self):
        stopper = ControlScoreStopper(patience=2)
        stopper.update(0.5)
        assert not stopper.update(0.4)
        assert not stopper.should_stop
        assert not stopper.update(0.5)          # a tie is no new best
        assert stopper.should_stop

    def test_a_score_below_zero_is_still_a_first_best(self):
        stopper = ControlScoreStopper(patience=1)
        assert stopper.update(-0.7)
        assert stopper.best_score == -0.7

    def test_a_nan_score_sets_no_best_and_counts_toward_patience(self):
        stopper = ControlScoreStopper(patience=1)
        assert not stopper.update(float("nan"))
        assert stopper.best_score is None and stopper.should_stop

    def test_its_state_round_trips_and_names_what_it_measures(self):
        stopper = ControlScoreStopper(patience=3)
        stopper.update(0.2)
        stopper.update(0.1)
        again = ControlScoreStopper(patience=3)
        again.load_state_dict(stopper.state_dict())
        assert (again.best, again.since_best) == (0.2, 1)
        assert stopper.state_dict()["measure"] == "control_score"

    def test_a_stopper_state_that_kept_a_loss_is_refused(self):
        with pytest.raises(StopperStateError, match="validation loss"):
            ControlScoreStopper(5).load_state_dict({"best": 0.4, "since_best": 1})


class TestModelOutput:
    def test_the_model_output_is_the_directory_the_run_names(self, tmp_path):
        config = TrainEffectModelConfig.unpathed(variant=Baseline.IDENTITY, model_output=tmp_path)
        assert config.resolved_model_output() == tmp_path


class TestHoldoutGuard:
    """An empty card-disjoint stratum must stop the run (T160, FR-088b).

    Left alone it produces a training run that writes nothing: `Validator.score`
    over no records returns `nan`, and gate 1 and the reported losses would
    measure nothing for the whole run. Hours of GPU time and no gate-1 reading.
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
    prose-built dataset read with a script vocabulary misses on every rarity
    lookup, and
    ``sample_weights`` reads a miss as "a shard collected after the dataset was
    built" and falls back to the resident shard's own count — reverting
    silently to exactly the per-shard weighting FR-141 exists to replace.
    """

    def test_a_vocabulary_on_the_other_surface_is_refused(self, tmp_path) -> None:
        from effects.application.train_effect_model import (
            SurfaceMismatchError,
            require_matching_surface,
        )

        with pytest.raises(SurfaceMismatchError, match="script"):
            require_matching_surface(
                manifest_surface="script",
                vocab_path=write_vocabulary(tmp_path / "vocab-script.txt", "prose"),
                corpus="output/effects/corpus",
            )

    def test_the_surface_the_dataset_was_built_on_is_accepted(self, tmp_path) -> None:
        from effects.application.train_effect_model import require_matching_surface

        for surface in ("script", "prose"):
            require_matching_surface(
                manifest_surface=surface,
                vocab_path=write_vocabulary(tmp_path / f"{surface}.txt", surface),
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
            TrainEffectModelConfig.unpathed(corpus="output/effects/corpus"), {},
        )

        assert provenance.corpus_path == "output/effects/corpus"
        assert provenance.corpus_digest == "abc123"

    def test_a_run_without_a_digest_records_no_corpus(self):
        """Nothing is pinned until a manifest has been read."""
        from effects.application.train_effect_model import (
            CorpusFacts,
            TrainEffectModelConfig,
        )

        provenance = CorpusFacts().provenance(TrainEffectModelConfig.unpathed(), {})

        assert provenance.corpus_path == ""
        assert provenance.corpus_digest == ""


def test_early_stopping_with_no_patience_never_stops():
    stopper = ControlScoreStopper(0)
    stopper.update(1.0)
    for _ in range(50):
        stopper.update(0.5)
    assert not stopper.should_stop


#: Each class a field's metadata may declare (``run_settings.setting``).
ON_RESUME = {"compared", "may change"}
RECORDED = {"always", "when set", "never"}


def test_every_flag_declares_how_a_resume_and_the_settings_treat_it():
    """The classification sits beside each field's help, so a flag added later
    cannot be left out of the settings or the resume's comparison unseen."""
    from dataclasses import fields

    unclassified = [
        spec.name for spec in fields(TrainEffectModelConfig)
        if spec.metadata.get("resume") not in ON_RESUME
        or spec.metadata.get("recorded") not in RECORDED
    ]
    assert unclassified == []


def test_a_flag_a_resume_compares_is_one_the_settings_record():
    """The comparison reads the recorded settings, so a compared flag recorded
    nowhere would be compared against nothing."""
    from dataclasses import fields

    assert [
        spec.name for spec in fields(TrainEffectModelConfig)
        if spec.metadata.get("resume") == "compared"
        and spec.metadata.get("recorded") == "never"
    ] == []


def test_the_settings_record_every_flag_that_shapes_a_step(tmp_path):
    """Finding 19.1: these were in no checkpoint. The variant tree is the one
    the run read, the manifest's when no flag named one (finding 16.6)."""
    from effects.application.run_context import training_settings
    from effects.application.train_effect_model import CorpusFacts, RunSpec

    config = TrainEffectModelConfig.unpathed(
        batch_size=8, grad_accum=2, steps_per_epoch=50, shards_per_epoch=4,
        curriculum_epoch=2, context_dropout=0.2, keyword_expand_p=0.5, lr_scale=0.5,
        variant_scripts=tmp_path / "variants",
    )
    settings = training_settings(
        RunSpec(config, CorpusFacts(digest="d"), seed=3), api_types=[], param_keys=[],
    )
    assert {name: settings[name] for name in (
        "batch_size", "grad_accum", "steps_per_epoch", "shards_per_epoch",
        "curriculum_epoch", "context_dropout", "keyword_expand_p", "lr_scale",
        "variant_scripts", "rule_mask", "variant", "seed",
    )} == {
        "batch_size": 8, "grad_accum": 2, "steps_per_epoch": 50, "shards_per_epoch": 4,
        "curriculum_epoch": 2, "context_dropout": 0.2, "keyword_expand_p": 0.5,
        "lr_scale": 0.5, "variant_scripts": str(tmp_path / "variants"),
        "rule_mask": "hard", "variant": "full", "seed": 3,
    }
    # Paths a run names but compares by digest or hash, and speed-only flags,
    # are not settings.
    assert not {"corpus", "model_output", "prefetch_workers", "resume"} & settings.keys()
