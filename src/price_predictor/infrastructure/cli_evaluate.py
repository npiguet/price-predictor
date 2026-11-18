"""The ``evaluate`` commands of the ``price_predictor`` CLI: parsers and handlers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from price_predictor.infrastructure.cli_common import add_shared_dataset_args

logger = logging.getLogger(__name__)


def add_evaluate(subparsers: argparse._SubParsersAction) -> None:
    """Register ``evaluate {sklearn,transformer}``."""
    evaluate_parser = subparsers.add_parser("evaluate",
                                            help="Evaluate model accuracy")
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="model", required=True)
    _add_evaluate_sklearn(evaluate_subparsers)
    _add_evaluate_transformer(evaluate_subparsers)


def _add_evaluate_sklearn(evaluate_subparsers: argparse._SubParsersAction) -> None:
    eval_sklearn = evaluate_subparsers.add_parser("sklearn",
                                                   help="Evaluate sklearn model")
    add_shared_dataset_args(eval_sklearn)
    eval_sklearn.add_argument("--model-path", type=str,
                              default="./models/price-predictor/sklearn/latest.joblib")
    eval_sklearn.add_argument("--output-csv", type=str, default=None)
    eval_sklearn.set_defaults(func=run_evaluate_sklearn)


def _add_evaluate_transformer(evaluate_subparsers: argparse._SubParsersAction) -> None:
    eval_transformer = evaluate_subparsers.add_parser(
        "transformer", help="Evaluate transformer model")
    add_shared_dataset_args(eval_transformer)
    eval_transformer.add_argument("--model-path", type=str,
                                  default="./models/price-predictor/transformer/")
    eval_transformer.add_argument("--vocab-path", type=str,
                                  default="models/price-predictor/transformer/vocab.txt",
                                  help="Path to vocab.txt")
    eval_transformer.set_defaults(func=run_evaluate_transformer)


def _write_per_card_csv(path: Path, per_card: list[dict]) -> None:
    """Write per-card evaluation rows to CSV at ``path``."""
    import csv

    fieldnames = [
        "name", "actual_price_eur",
        "predicted_price_eur", "absolute_error_eur",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_card)


def _writes_csv(args: argparse.Namespace, per_card: list[dict]) -> bool:
    """Whether ``--output-csv`` was given and there are rows to write."""
    return bool(getattr(args, "output_csv", None)) and bool(per_card)


def run_evaluate_sklearn(args: argparse.Namespace) -> int:
    """Evaluate the sklearn model on held-out test data."""
    from price_predictor.application.evaluate import EvaluateModelUseCase
    from price_predictor.infrastructure.model_store import ModelNotFoundError

    model_path = Path(args.model_path)
    try:
        use_case = EvaluateModelUseCase()
        result = use_case.execute(
            model_path=model_path,
            output_dir=Path(args.output_dir),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            held_out_fraction=args.test_split,
            random_seed=args.random_seed,
        )
    except ModelNotFoundError:
        logger.error("Model file not found at %s", model_path)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 2

    if _writes_csv(args, result.per_card):
        _write_per_card_csv(Path(args.output_csv), result.per_card)

    output = {
        "model_name": "sklearn",
        "model_version": result.model_version,
        "mean_absolute_error_eur": result.metrics.mean_absolute_error_eur,
        "median_percentage_error": result.metrics.median_percentage_error,
        "median_abs_error_log": result.metrics.median_log_error,
        "top_20_overlap": result.metrics.top_20_overlap,
        "sample_count": result.metrics.sample_count,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_evaluate_transformer(args: argparse.Namespace) -> int:
    """Evaluate the price transformer on the held-out split."""
    from price_predictor.application.evaluate_transformer import evaluate_transformer

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        logger.error(
            "Vocabulary file not found at %s. "
            "Run 'python -m price_predictor vocabulary' first.",
            vocab_path,
        )
        return 1

    model_path = Path(args.model_path)
    try:
        result = evaluate_transformer(
            model_dir=model_path,
            output_dir=Path(args.output_dir),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            vocab_path=vocab_path,
            random_seed=args.random_seed,
        )
    except FileNotFoundError:
        logger.error("Model not found at %s", model_path)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 2

    output = {
        "model_name": "transformer",
        "model_version": result.model_version,
        "mean_absolute_error_eur": result.mean_absolute_error_eur,
        "median_percentage_error": result.median_percentage_error,
        "median_abs_error_log": result.median_log_error,
        "top_20_overlap": result.top_20_overlap,
        "sample_count": result.sample_count,
    }
    print(json.dumps(output, indent=2))

    if result.per_bucket:
        print()
        print(result.format_per_bucket_table())

    return 0
