"""The ``train-encoder`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_train_encoder_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-encoder",
        help=(
            "Train the sealed encoder + regression head on per-card "
            "winnability and save only encoder weights"
        ),
    )
    parser.set_defaults(func=run_train_encoder)
    parser.add_argument(
        "--cards-played-path", required=True,
        help="Path to cards-played.txt",
    )
    parser.add_argument(
        "--cards-folder", required=True,
        help="Directory of converted card .txt files",
    )
    parser.add_argument(
        "--vocab-path", required=True,
        help="Tokenizer vocab file",
    )
    parser.add_argument(
        "--model-output", required=True,
        help="Output directory for {timestamp}.pt + latest.pt (the run directory)",
    )
    parser.add_argument(
        "--win-rates-output", default=None,
        help=(
            f"The per-card win-rate table's file (default: {_WIN_RATES_NAME} in "
            "--model-output)"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early-stop after this many epochs without improvement (default: 20)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=6,
        help="Number of transformer encoder layers (default: 6)",
    )
    parser.add_argument(
        "--d-model", type=int, default=256,
        help=(
            "Model / embedding width (default: 256). Must be divisible by "
            "--n-heads and --n-pool-queries."
        ),
    )
    parser.add_argument(
        "--ff-dim", type=int, default=None,
        help="Transformer feed-forward hidden dim (default: 4 * --d-model)",
    )
    parser.add_argument(
        "--n-heads", type=int, default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--n-pool-queries", type=int, default=4,
        help=(
            "Number of multi-query attention pool queries (default: 4). "
            "Must divide d_model (256)."
        ),
    )
    parser.add_argument(
        "--pool-mode", choices=["dual", "attn"], default="dual",
        help=(
            "Card-vector pooling: 'dual' = multi-query attention pool "
            "concatenated with max-pool over the token outputs (default, "
            "pooled width 2*d_model); 'attn' = attention pool only (pooled "
            "width d_model)."
        ),
    )
    parser.add_argument(
        "--shrinkage-k", type=float, default=20.0,
        help=(
            "Bayesian shrinkage strength toward each head's neutral point "
            "(default: 20). 0 = raw labels; higher = more aggressive "
            "shrinkage on low-observation cells. Drives both label "
            "shrinkage and FR-017a per-head sample weighting."
        ),
    )
    parser.add_argument(
        "--mlm-weight", type=float, default=0.1,
        help=(
            "Weight on the MLM auxiliary cross-entropy loss "
            "(default: 0.1). Full training loss = L_reg + "
            "(--mlm-weight) * L_mlm."
        ),
    )
    parser.add_argument(
        "--mlm-mask-prob", type=float, default=0.15,
        help=(
            "Per-token probability of being masked for the MLM head "
            "(default: 0.15). Only non-special, non-pad positions are "
            "eligible."
        ),
    )


_WIN_RATES_NAME = "cards-win-rates.txt"


def _win_rates_output(args: argparse.Namespace) -> Path:
    """``train-encoder``'s win-rate table: ``--win-rates-output``, else the
    ``--model-output`` run directory."""
    if args.win_rates_output:
        return Path(args.win_rates_output)
    return Path(args.model_output) / _WIN_RATES_NAME


def run_train_encoder(args: argparse.Namespace) -> int:
    """Execute the train-encoder command."""
    from sealed.application.train_encoder import (
        TrainEncoderConfig,
        _PreFlightError,
    )
    from sealed.application.train_encoder import (
        run as run_train_encoder_pipeline,
    )

    ff_dim = args.ff_dim if args.ff_dim is not None else 4 * args.d_model
    config = TrainEncoderConfig(
        cards_played_path=Path(args.cards_played_path),
        cards_folder=Path(args.cards_folder),
        vocab_path=Path(args.vocab_path),
        model_output_dir=Path(args.model_output),
        win_rates_path=_win_rates_output(args),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        dropout=args.dropout,
        n_layers=args.n_layers,
        d_model=args.d_model,
        ff_dim=ff_dim,
        n_heads=args.n_heads,
        n_pool_queries=args.n_pool_queries,
        pool_mode=args.pool_mode,
        shrinkage_k=args.shrinkage_k,
        mlm_weight=args.mlm_weight,
        mlm_mask_prob=args.mlm_mask_prob,
    )
    try:
        run_train_encoder_pipeline(config)
    except _PreFlightError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        # SealedEncoderConfig validation (n_heads / d_model, etc.)
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
