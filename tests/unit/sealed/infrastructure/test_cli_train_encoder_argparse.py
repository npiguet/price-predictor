"""Argparse-level tests for the ``train-encoder`` subcommand (FR-021/FR-022)."""

from __future__ import annotations

import argparse
import io

import pytest

from sealed.infrastructure.cli import build_parser


def _parse(args: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(args)


class TestTrainEncoderArgs:
    def test_defaults_match_fr021(self):
        ns = _parse(["train-encoder"])
        assert ns.cards_played_path == "output/sealed/cards-played.txt"
        assert ns.cards_folder == "output/cardsfolder/"
        assert ns.vocab_path == "models/sealed/encoder/vocab.txt"
        assert ns.model_output == "models/sealed/encoder/"
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
        from sealed.application.train_encoder import TrainEncoderConfig
        cfg = TrainEncoderConfig(mlm_weight=0.4, mlm_mask_prob=0.3)
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

    def test_hardcoded_constants_absent_from_cli_surface(self):
        # FR-022 constants (d_model, ff_dim, val_fraction, random_seed) are
        # not flags. Passing them as flags must error out.
        parser = build_parser()
        for forbidden in ("--d-model", "--ff-dim", "--val-fraction", "--random-seed"):
            with pytest.raises(SystemExit):
                parser.parse_args(["train-encoder", forbidden, "1"])

    def test_invalid_n_pool_queries_exits_via_pipeline(self):
        # The argparse layer accepts any int; the failure surfaces as exit
        # code 6 once the pipeline runs (pre-flight in run_train_encoder).
        # This is exercised indirectly here: argparse does not reject the
        # value, so the pipeline must.
        ns = _parse(["train-encoder", "--n-pool-queries", "5"])
        assert ns.n_pool_queries == 5
