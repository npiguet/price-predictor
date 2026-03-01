"""CLI interface for the card price predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="price_predictor",
        description="MTG card price predictor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict price for a card")
    predict_parser.add_argument("--mana-cost", type=str, default=None)
    predict_parser.add_argument("--types", type=str, default=None)
    predict_parser.add_argument("--supertypes", type=str, default=None)
    predict_parser.add_argument("--subtypes", type=str, default=None)
    predict_parser.add_argument("--oracle-text", type=str, default=None)
    predict_parser.add_argument("--keywords", type=str, default=None)
    predict_parser.add_argument("--power", type=str, default=None)
    predict_parser.add_argument("--toughness", type=str, default=None)
    predict_parser.add_argument("--loyalty", type=str, default=None)
    predict_parser.add_argument("--colors", type=str, default=None)
    predict_parser.add_argument(
        "--model-path", type=str, default="models/latest.joblib"
    )

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train model on card data")
    train_parser.add_argument(
        "--forge-cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder",
    )
    train_parser.add_argument(
        "--prices-path", type=str, default="resources/AllPricesToday.json"
    )
    train_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json"
    )
    train_parser.add_argument("--output-path", type=str, default="models/")
    train_parser.add_argument("--test-split", type=float, default=0.2)
    train_parser.add_argument("--random-seed", type=int, default=42)

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model accuracy")
    eval_parser.add_argument(
        "--model-path", type=str, default="models/latest.joblib"
    )
    eval_parser.add_argument(
        "--forge-cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder",
    )
    eval_parser.add_argument(
        "--prices-path", type=str, default="resources/AllPricesToday.json"
    )
    eval_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json"
    )
    eval_parser.add_argument("--test-split", type=float, default=0.2)
    eval_parser.add_argument("--random-seed", type=int, default=42)
    eval_parser.add_argument("--output-csv", type=str, default=None)

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the prediction REST service")
    serve_parser.add_argument(
        "--model-path", type=str, default="models/latest.joblib"
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def run_serve(args: argparse.Namespace) -> int:
    """Execute the serve command."""
    import uvicorn

    from price_predictor.infrastructure.model_store import ModelNotFoundError, load_model
    from price_predictor.infrastructure.server import create_app

    model_path = Path(args.model_path)
    print(f"Loading model from {model_path}...", file=sys.stderr)

    try:
        artifact = load_model(model_path)
    except ModelNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        return 2

    # Extract model version from filename
    model_version = model_path.stem
    if model_version == "latest":
        model_version = "latest"
    artifact["model_version"] = model_version

    app = create_app(artifact)
    print(f"Prediction service started on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _parse_comma_list(value: str | None) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def run_predict(args: argparse.Namespace) -> int:
    """Execute the predict command."""
    from price_predictor.application.predict import PredictPriceUseCase
    from price_predictor.infrastructure.model_store import ModelNotFoundError

    types = _parse_comma_list(args.types)
    if not types:
        print("Error: No card types provided. At least one --types value is required.",
              file=sys.stderr)
        return 1

    mana_cost = ManaCost.parse(args.mana_cost) if args.mana_cost else None

    try:
        card = Card(
            name="prediction_input",
            types=types,
            supertypes=_parse_comma_list(args.supertypes),
            subtypes=_parse_comma_list(args.subtypes),
            mana_cost=mana_cost,
            oracle_text=args.oracle_text,
            keywords=_parse_comma_list(args.keywords),
            power=args.power,
            toughness=args.toughness,
            loyalty=args.loyalty,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    model_path = Path(args.model_path)
    try:
        use_case = PredictPriceUseCase()
        result = use_case.execute(card, model_path)
    except ModelNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    output = {
        "predicted_price_eur": result.predicted_price_eur,
        "model_version": result.model_version,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    from price_predictor.application.train import TrainModelUseCase

    forge_path = Path(args.forge_cards_path)
    if not forge_path.exists():
        print(f"Error: Forge cards path not found: {forge_path}", file=sys.stderr)
        return 1

    prices_path = Path(args.prices_path)
    if not prices_path.exists():
        print(f"Error: Prices file not found: {prices_path}", file=sys.stderr)
        return 1

    printings_path = Path(args.printings_path)
    if not printings_path.exists():
        print(f"Error: Printings file not found: {printings_path}", file=sys.stderr)
        return 1

    try:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            forge_cards_path=forge_path,
            prices_path=prices_path,
            printings_path=printings_path,
            output_path=Path(args.output_path),
            test_split=args.test_split,
            random_seed=args.random_seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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


def run_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    from price_predictor.application.evaluate import EvaluateModelUseCase
    from price_predictor.infrastructure.model_store import ModelNotFoundError

    model_path = Path(args.model_path)
    try:
        use_case = EvaluateModelUseCase()
        result = use_case.execute(
            model_path=model_path,
            forge_cards_path=Path(args.forge_cards_path),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            test_split=args.test_split,
            random_seed=args.random_seed,
        )
    except ModelNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Optionally write per-card CSV
    if args.output_csv and result.per_card:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "actual_price_eur",
                    "predicted_price_eur", "absolute_error_eur",
                ],
            )
            writer.writeheader()
            writer.writerows(result.per_card)

    output = {
        "model_version": result.model_version,
        "mean_absolute_error_eur": result.metrics.mean_absolute_error_eur,
        "median_percentage_error": result.metrics.median_percentage_error,
        "top_20_overlap": result.metrics.top_20_overlap,
        "sample_count": result.metrics.sample_count,
    }
    print(json.dumps(output, indent=2))
    return 0
