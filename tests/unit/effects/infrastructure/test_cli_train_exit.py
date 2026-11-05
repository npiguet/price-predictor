"""``train-effect-model``'s exit codes for an operator's error."""

from __future__ import annotations

import pytest

from effects.domain.baselines import Baseline


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
        monkeypatch.setattr(cli, "config_from_args", lambda cls, args: None)
        args = argparse.Namespace(
            variant=Baseline.FULL, corpus="x",
            encoder_d_model=256, encoder_layers=4,
        )

        with caplog.at_level(logging.ERROR):
            assert cli.run_train_effect_model(args) == 2

        assert "card-disjoint stratum holds no records" in caplog.text


# ── the gen-3 flags and startup checks (contracts/cli.md, T149) ──────────


def _parse(*argv: str):
    from effects.infrastructure.cli import build_parser

    return build_parser().parse_args(["train-effect-model", "--corpus", "c", *argv])


def _config(*argv: str):
    from effects.application.train_effect_model import TrainEffectModelConfig
    from effects.infrastructure.cli import config_from_args

    return config_from_args(TrainEffectModelConfig, _parse(*argv))


def test_the_gen3_flags_default_as_the_contract_says():
    from effects.domain.rule_mask import RuleMask

    config = _config()
    assert config.rule_mask is RuleMask.HARD
    assert (config.deviance_floor, config.ema_epochs, config.stratum_weight) == (0.04, 1.0, 0.5)
    assert (config.patience, config.relation_values, config.e_dim) == (5, "on", 256)


def test_the_gen3_flags_reach_the_config():
    from effects.domain.rule_mask import RuleMask

    config = _config(
        "--rule-mask", "soft", "--deviance-floor", "0.1", "--ema-epochs", "2.5",
        "--stratum-weight", "1", "--patience", "0",
    )
    assert config.rule_mask is RuleMask.SOFT
    assert (config.deviance_floor, config.ema_epochs, config.stratum_weight) == (0.1, 2.5, 1.0)
    assert config.patience == 0


def test_a_rule_mask_other_than_hard_or_soft_is_refused():
    with pytest.raises(SystemExit):
        _parse("--rule-mask", "medium")


@pytest.mark.parametrize("argv, message", [
    (("--stratum-weight", "1.5"), "--stratum-weight"),
    (("--stratum-weight", "-0.1"), "--stratum-weight"),
    (("--ema-epochs", "0"), "--ema-epochs"),
    (("--deviance-floor", "0"), "--deviance-floor"),
])
def test_an_out_of_range_value_exits_before_training(monkeypatch, caplog, argv, message):
    import logging

    from effects.application import train_effect_model
    from effects.infrastructure import cli

    started = []
    monkeypatch.setattr(train_effect_model, "run", lambda config: started.append(1))
    with caplog.at_level(logging.ERROR):
        assert cli.run_train_effect_model(_parse(*argv)) == 2
    assert not started and message in caplog.text


@pytest.mark.parametrize("failure", [
    "UnacceptedGenerationError", "MissingTrainingInputError", "ResumeMismatchError",
])
def test_the_startup_refusals_exit_one(monkeypatch, caplog, failure):
    """A record lacking ``generation: 3``, a manifest without per-rule counts
    (``MissingTrainingInputError``) and a resume changing a compared flag."""
    from effects.application import train_effect_model
    from effects.domain.shard_generation import UnacceptedGenerationError
    from effects.infrastructure import cli, effect_model_store

    kind = {
        "UnacceptedGenerationError": UnacceptedGenerationError,
        "MissingTrainingInputError": train_effect_model.MissingTrainingInputError,
        "ResumeMismatchError": effect_model_store.ResumeMismatchError,
    }[failure]

    def refuse(config):
        raise kind("refused for the test")

    monkeypatch.setattr(train_effect_model, "run", refuse)
    assert cli.run_train_effect_model(_parse()) == 1
