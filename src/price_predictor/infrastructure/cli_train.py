"""The ``train`` commands of the ``price_predictor`` CLI: parsers and handlers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from price_predictor.infrastructure.cli_common import (
    add_shared_dataset_args,
    require_path_exists,
)

logger = logging.getLogger(__name__)


def add_train(subparsers: argparse._SubParsersAction) -> None:
    """Register ``train {sklearn,transformer}``."""
    train_parser = subparsers.add_parser("train", help="Train model on card data")
    train_subparsers = train_parser.add_subparsers(dest="model", required=True)
    _add_train_sklearn(train_subparsers)
    _add_train_transformer(train_subparsers)


def _add_train_sklearn(train_subparsers: argparse._SubParsersAction) -> None:
    train_sklearn = train_subparsers.add_parser("sklearn",
                                                 help="Train sklearn model")
    add_shared_dataset_args(train_sklearn)
    train_sklearn.add_argument("--model-output", type=str,
                               default="./models/price-predictor/sklearn/")
    train_sklearn.set_defaults(func=run_train_sklearn)


def _add_train_transformer(train_subparsers: argparse._SubParsersAction) -> None:
    train_transformer = train_subparsers.add_parser("transformer",
                                                     help="Train transformer model")
    add_shared_dataset_args(train_transformer)
    train_transformer.add_argument("--model-output", type=str,
                                   default="./models/price-predictor/transformer/")
    train_transformer.add_argument("--batch-size", type=int, default=64)
    train_transformer.add_argument("--epochs", type=int, default=100)
    train_transformer.add_argument("--lr", type=float, default=1e-4)
    train_transformer.add_argument("--patience", type=int, default=20)
    train_transformer.add_argument("--vocab-path", type=str,
                                   default="models/price-predictor/transformer/vocab.txt",
                                   help="Path to vocab.txt built by 'vocabulary' command")
    train_transformer.add_argument(
        "--log-offset", type=float, default=2.0,
        help="Offset for log(price+offset) transform (use 0.5 for high-price signal)",
    )
    train_transformer.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1; try 0.05 or 0.0 for small datasets)",
    )
    train_transformer.add_argument(
        "--sampler-exponent", type=float, default=1.0,
        help="Price-bucket oversampling exponent: 0=uniform, 0.5=sqrt, 1=full",
    )
    train_transformer.add_argument(
        "--d-model", type=int, default=256,
        help="Embedding dimension (default: 256). Must divide by n-heads.",
    )
    train_transformer.add_argument(
        "--n-layers", type=int, default=2,
        help="Number of transformer encoder layers (default: 2).",
    )
    train_transformer.add_argument(
        "--n-heads", type=int, default=4,
        help="Number of attention heads (default: 4). Must divide d-model.",
    )
    train_transformer.add_argument(
        "--ff-dim", type=int, default=1024,
        help="Feed-forward inner dim (default: 1024, typically 4x d-model).",
    )
    train_transformer.set_defaults(func=run_train_transformer)


def run_train_sklearn(args: argparse.Namespace) -> int:
    """Train the sklearn model using converted card text files."""
    from price_predictor.application.train import TrainModelUseCase

    output_dir = Path(args.output_dir)
    if not require_path_exists(output_dir, "Converted cards directory"):
        logger.error("Run 'convert' first to generate converted card text files.")
        return 1
    prices_path = Path(args.prices_path)
    if not require_path_exists(prices_path, "Prices file"):
        return 1
    printings_path = Path(args.printings_path)
    if not require_path_exists(printings_path, "Printings file"):
        return 1
    try:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=output_dir,
            prices_path=prices_path,
            printings_path=printings_path,
            output_path=Path(args.model_output),
            held_out_fraction=args.test_split,
            random_seed=args.random_seed,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 2
    output = {
        "model_version": result.trained_model.model_version,
        "model_path": str(result.model_path),
        "cards_used": result.trained_model.card_count,
        "cards_skipped": result.cards_skipped,
        "skipped_reasons": result.skipped_reasons,
        "price_range_eur": {
            "min": result.trained_model.price_range_min_eur,
            "max": result.trained_model.price_range_max_eur,
        },
    }
    print(json.dumps(output, indent=2))
    return 0


def run_train_transformer(args: argparse.Namespace) -> int:
    """Train the transformer model using converted card text files."""
    from price_predictor.application.train_transformer import train_transformer

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        logger.error(
            "Vocabulary file not found at %s. "
            "Run 'python -m price_predictor vocabulary' first.",
            vocab_path,
        )
        return 1

    try:
        train_transformer(
            output_dir=Path(args.output_dir),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            model_output=Path(args.model_output),
            vocab_path=vocab_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            random_seed=args.random_seed,
            log_offset=args.log_offset,
            dropout=args.dropout,
            sampler_exponent=args.sampler_exponent,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ff_dim=args.ff_dim,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1
    except RuntimeError as e:
        logger.error("%s", e)
        return 2
    return 0
