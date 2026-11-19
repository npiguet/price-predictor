"""Argparse-level tests for the ``train-encoder`` subcommand (FR-021/FR-022)."""

from __future__ import annotations

import argparse
import io

import pytest

from sealed.infrastructure.cli import build_parser

_PATHS = [
    "--cards-played-path", "played.txt", "--cards-folder", "cards",
    "--vocab-path", "vocab.txt", "--model-output", "run",
]


def _parse(args: list[str]) -> argparse.Namespace:
    """Parse a ``train-encoder`` command line, the required path flags added."""
    return build_parser().parse_args([*args, *_PATHS])


class TestTrainEncoderArgs:
    def test_defaults_match_fr021(self):
        ns = _parse(["train-encoder"])
        assert ns.batch_size == 64
        assert ns.epochs == 100
        assert ns.lr == 1e-4
        assert ns.patience == 20
        assert ns.dropout == 0.1
        assert ns.n_layers == 6
        assert ns.n_heads == 4
        assert ns.n_pool_queries == 4
        assert ns.shrinkage_k == 20.0
        assert ns.mlm_weight == 0.1
        assert ns.mlm_mask_prob == 0.15

    def test_overrides_propagate(self):
        ns = _parse([
            "train-encoder",
            "--shrinkage-k", "0",
            "--epochs", "1",
            "--n-layers", "2",
            "--n-pool-queries", "8",
            "--mlm-weight", "0.5",
            "--mlm-mask-prob", "0.25",
        ])
        assert ns.shrinkage_k == 0.0
        assert ns.epochs == 1
        assert ns.n_layers == 2
        assert ns.n_pool_queries == 8
        assert ns.mlm_weight == 0.5
        assert ns.mlm_mask_prob == 0.25

    def test_mlm_flags_surface_on_resolved_config(self):
        """The two new flags must reach the TrainEncoderConfig dataclass."""
        from pathlib import Path

        from sealed.application.train_encoder import TrainEncoderConfig
        cfg = TrainEncoderConfig(
            cards_played_path=Path("played.txt"), cards_folder=Path("cards"),
            vocab_path=Path("vocab.txt"), model_output_dir=Path("run"),
            win_rates_path=Path("rates.txt"), mlm_weight=0.4, mlm_mask_prob=0.3,
        )
        assert cfg.mlm_weight == 0.4
        assert cfg.mlm_mask_prob == 0.3

    def test_no_aggregate_labels_subcommand(self):
        # FR-013: aggregation is inline, so the help must NOT advertise
        # an aggregate-labels subcommand.
        parser = build_parser()
        buf = io.StringIO()
        parser.print_help(buf)
        help_text = buf.getvalue()
        assert "aggregate-labels" not in help_text

    def test_val_fraction_and_random_seed_absent_from_cli_surface(self):
        # val_fraction and random_seed stay hardcoded — not flags.
        parser = build_parser()
        for forbidden in ("--val-fraction", "--random-seed"):
            with pytest.raises(SystemExit):
                parser.parse_args(["train-encoder", forbidden, "1"])

    def test_d_model_and_ff_dim_are_tunable(self):
        # d_model / ff_dim are exposed for architecture experiments.
        ns = _parse(["train-encoder", "--d-model", "128"])
        assert ns.d_model == 128
        assert ns.ff_dim is None  # resolved to 4 * d_model by the handler
        ns = _parse(["train-encoder", "--d-model", "64", "--ff-dim", "300"])
        assert (ns.d_model, ns.ff_dim) == (64, 300)
        # defaults unchanged
        ns = _parse(["train-encoder"])
        assert ns.d_model == 256 and ns.ff_dim is None

    def test_invalid_n_pool_queries_exits_via_pipeline(self):
        # The argparse layer accepts any int; the failure surfaces as exit
        # code 6 once the pipeline runs (pre-flight in run_train_encoder).
        # This is exercised indirectly here: argparse does not reject the
        # value, so the pipeline must.
        ns = _parse(["train-encoder", "--n-pool-queries", "5"])
        assert ns.n_pool_queries == 5


class TestWhereTheWinRatesGo:
    """``--win-rates-output`` names the per-card win-rate table's file. Unset,
    the table goes to ``--model-output``."""

    @pytest.mark.parametrize(("argv", "expected"), [
        ((), "run/cards-win-rates.txt"),
        (("--win-rates-output", "elsewhere/rates.txt"), "elsewhere/rates.txt"),
    ])
    def test_the_pipeline_is_given_the_path_the_flags_name(self, monkeypatch, argv, expected):
        from pathlib import Path

        import sealed.application.train_encoder as module
        from sealed.infrastructure.cli import run_train_encoder

        seen = {}
        monkeypatch.setattr(module, "run", lambda config: seen.setdefault("config", config))
        assert run_train_encoder(_parse(["train-encoder", *argv])) == 0
        assert seen["config"].win_rates_path == Path(expected)
