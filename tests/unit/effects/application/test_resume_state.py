"""Which settings a resume compares: every recorded flag unless its field
declares that a resume may change it, and the digests the run recorded."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from effects.application.resume_state import settings_mismatches
from effects.application.run_context import training_settings
from effects.application.run_settings import free_on_resume
from effects.application.train_effect_model import (
    CorpusFacts,
    RunSpec,
    TrainEffectModelConfig,
)
from effects.domain.rule_mask import RuleMask
from effects.infrastructure.effect_model_store import tokenizer_hashes

FLAGS = {spec.name for spec in fields(TrainEffectModelConfig)}


def _settings(config: TrainEffectModelConfig, facts: CorpusFacts | None = None) -> dict:
    return training_settings(
        RunSpec(
            config, facts or CorpusFacts(digest="d"), seed=7,
            tokenizer_hashes=tokenizer_hashes(config.tokenizer_files),
        ),
        api_types=[], param_keys=[],
    )


def test_the_flags_a_resume_may_change_place_no_step():
    """The paths are compared by digest or hash; the rest change speed, the
    learning rate's factor, or how long the run goes on."""
    assert free_on_resume(TrainEffectModelConfig) == {
        "epochs", "patience", "lr_scale", "save_every_epoch", "model_output",
        "prefetch_workers", "resume", "corpus", "cards_folders", "variant_scripts",
        "vocab_path", "keyword_definitions", "reports_dir",
    }


def test_every_other_recorded_flag_is_compared():
    compared = set(_settings(TrainEffectModelConfig.unpathed())) & FLAGS
    assert compared - free_on_resume(TrainEffectModelConfig) == {
        "seed", "variant", "e_dim", "relation_values", "rule_mask", "deviance_floor",
        "ema_epochs", "stratum_weight", "e_noise", "value_weight", "encoder_layers",
        "encoder_d_model", "keyword_expand_p", "context_dropout", "mlm_weight",
        "mlm_mask_prob", "api_weight", "curriculum_epoch", "curriculum_rewarm_steps",
        "batch_size", "grad_accum", "steps_per_epoch", "shards_per_epoch",
        "withhold_keyword",
    }


@pytest.mark.parametrize("flag, value", [
    ("rule_mask", RuleMask.SOFT), ("deviance_floor", 0.05), ("ema_epochs", 2.0),
    ("stratum_weight", 0.25), ("relation_values", "off"), ("curriculum_rewarm_steps", 100),
    ("e_noise", 0.2), ("batch_size", 16), ("context_dropout", 0.3),
])
def test_a_resume_changing_a_flag_that_shapes_a_step_is_a_mismatch(flag, value):
    """Story 7 acceptance 5, for every flag contracts/cli.md lists, and more."""
    config = TrainEffectModelConfig.unpathed()
    mismatches = settings_mismatches(
        _settings(config), _settings(replace(config, **{flag: value})),
    )
    assert len(mismatches) == 1 and mismatches[0].startswith(f"{flag}:")


def test_a_free_flag_that_moved_is_no_mismatch():
    config = TrainEffectModelConfig.unpathed()
    assert not settings_mismatches(
        _settings(config), _settings(replace(config, epochs=80, patience=0, lr_scale=0.5)),
    )


def test_a_resume_dropping_the_copies_of_the_run_it_continues_is_a_mismatch():
    """Recorded only when set, so the saved side alone can hold the key."""
    config = TrainEffectModelConfig.unpathed()
    saved = _settings(replace(config, lethality_copies=3))
    assert "lethality_copies" not in _settings(config)
    assert settings_mismatches(saved, _settings(config))
    assert settings_mismatches(_settings(config), saved)


def test_a_vocabulary_rebuilt_at_the_same_path_is_a_mismatch(tmp_path):
    """Rebuilt at the same size the embedding shapes match, so only the bytes
    say every token id may have moved."""
    vocab, keywords = tmp_path / "vocab.json", tmp_path / "keywords.json"
    vocab.write_text('{"a": 0}')
    keywords.write_text("{}")
    config = TrainEffectModelConfig.unpathed(vocab_path=vocab, keyword_definitions=keywords)
    saved = _settings(config)

    vocab.write_text('{"b": 0}')
    assert any("vocab" in line for line in settings_mismatches(saved, _settings(config)))
    vocab.write_text('{"a": 0}')
    keywords.write_text('{"flying": "x"}')
    assert any(
        "keyword" in line for line in settings_mismatches(saved, _settings(config))
    )


def test_a_resume_against_another_shard_listing_is_a_mismatch():
    """The listing sits outside the manifest's digest, so a rebuilt corpus
    with the same decisions still moves the shards a resume would train on."""
    config = TrainEffectModelConfig.unpathed()

    def settings(listing_digest: str) -> dict:
        return _settings(config, CorpusFacts(digest="d", shards_digest=listing_digest))

    saved = settings("listing-a")
    assert settings_mismatches(saved, settings("listing-a")) == []
    mismatches = settings_mismatches(saved, settings("listing-b"))
    assert [line.split(":")[0] for line in mismatches] == ["corpus_shards_digest"]


def test_a_resume_without_a_seed_takes_the_one_the_resume_file_holds(tmp_path, monkeypatch):
    from effects.application import resume_state

    monkeypatch.setattr(resume_state, "load_resume_state", lambda d: {"settings": {"seed": 99}})
    config = TrainEffectModelConfig.unpathed(resume=True, model_output=tmp_path)
    assert resume_state.run_seed(config) == 99
    assert resume_state.run_seed(replace(config, seed=5)) == 5


def test_a_fresh_run_without_a_seed_draws_one(tmp_path):
    from effects.application import resume_state

    config = TrainEffectModelConfig.unpathed(model_output=tmp_path)
    assert isinstance(resume_state.run_seed(config), int)


def _built_run(monkeypatch, tmp_path):
    """A loop over a stand-in preparer at the smallest encoder, and its run."""
    from types import SimpleNamespace

    from effects.application.run_context import build_run
    from effects.domain.input_layout import SlotKind
    from tests.unit.effects.builders import make_cpu_training_loop

    loop = make_cpu_training_loop(
        monkeypatch, encoder_layers=1, encoder_d_model=8, model_output=tmp_path,
    )
    return loop, build_run(loop.spec, SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=8),
        surface_spec=SimpleNamespace(widths=dict.fromkeys(SlotKind, 4)),
        api_types=[], param_keys=[],
    ), device=loop.device)


def test_a_checkpoint_records_the_settings_the_resume_file_holds(monkeypatch, tmp_path):
    """One record of the run's settings (finding 19.1): what a checkpoint
    holds is what a resume compares."""
    from effects.application import resume_state

    loop, run = _built_run(monkeypatch, tmp_path)
    written = {}
    monkeypatch.setattr(
        resume_state, "save_resume_state", lambda directory, state: written.update(state),
    )
    resume_state.save_resume(run, 1)
    assert loop._checkpoint(1, run).training_settings == written["settings"]


def test_a_resume_file_recording_the_settings_in_the_old_shape_is_refused(
    monkeypatch, tmp_path,
):
    """No backward compatibility (D19): refused by name, never compared key by key."""
    from effects.application import resume_state
    from effects.infrastructure.effect_model_store import ResumeMismatchError

    _loop, run = _built_run(monkeypatch, tmp_path)
    old = {name: value for name, value in run.training_settings.items()
           if name != "settings_shape"}
    monkeypatch.setattr(resume_state, "load_resume_state", lambda d: {"settings": old})
    with pytest.raises(ResumeMismatchError, match="settings_shape"):
        resume_state.restore_resume(run)
