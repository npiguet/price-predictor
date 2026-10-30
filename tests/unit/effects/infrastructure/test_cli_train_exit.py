"""``train-effect-model``'s exit codes for an operator's error."""

from __future__ import annotations

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
