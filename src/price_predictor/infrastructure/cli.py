"""CLI interface for the card price predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

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

    # eval subcommand
    eval_card_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a card from a Forge script file via the prediction service",
    )
    eval_card_parser.add_argument(
        "file", type=str, help="Path to a Forge card script file"
    )
    eval_card_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/api/v1/evaluate",
        help="Prediction service endpoint URL",
    )

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert", help="Convert Forge card scripts to LLM-friendly text format"
    )
    convert_parser.add_argument(
        "--cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
        help="Path to Forge cardsfolder directory",
    )
    convert_parser.add_argument(
        "--output-path", type=str, default="./output",
        help="Output directory for converted files",
    )

    # train-transformer subcommand
    tt_parser = subparsers.add_parser(
        "train-transformer", help="Train transformer model on converted card texts"
    )
    tt_parser.add_argument("--output-dir", type=str, default="output/")
    tt_parser.add_argument(
        "--prices-path", type=str, default="resources/AllPricesToday.json"
    )
    tt_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json"
    )
    tt_parser.add_argument(
        "--forge-cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
    )
    tt_parser.add_argument("--model-output", type=str, default="models/transformer/")
    tt_parser.add_argument("--batch-size", type=int, default=64)
    tt_parser.add_argument("--epochs", type=int, default=20)
    tt_parser.add_argument("--lr", type=float, default=1e-4)
    tt_parser.add_argument("--patience", type=int, default=5)
    tt_parser.add_argument("--random-seed", type=int, default=42)

    # evaluate-transformer subcommand
    et_parser = subparsers.add_parser(
        "evaluate-transformer", help="Evaluate transformer model accuracy"
    )
    et_parser.add_argument("--model-path", type=str, default="models/transformer/")
    et_parser.add_argument("--output-dir", type=str, default="output/")
    et_parser.add_argument(
        "--prices-path", type=str, default="resources/AllPricesToday.json"
    )
    et_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json"
    )
    et_parser.add_argument(
        "--forge-cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
    )
    et_parser.add_argument("--random-seed", type=int, default=42)
    et_parser.add_argument("--output-csv", type=str, default=None)

    # check-convert subcommand
    check_parser = subparsers.add_parser(
        "check-convert",
        help="Check converted files against Oracle text from Forge scripts",
    )
    check_parser.add_argument(
        "--cards-path", type=str,
        default="../forge/forge-gui/res/cardsfolder/",
        help="Path to Forge cardsfolder directory",
    )
    check_parser.add_argument(
        "--output-path", type=str, default="./output",
        help="Path to converted output directory",
    )
    check_parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Similarity threshold below which cards are flagged (0.0-1.0)",
    )
    check_parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of results to show (0 = all)",
    )

    return parser


def run_eval(args: argparse.Namespace) -> int:
    """Execute the eval command — send a card file to the prediction endpoint."""
    from urllib.error import HTTPError, URLError

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    if not file_path.is_file():
        print(f"Error: Path is not a file: {file_path}", file=sys.stderr)
        return 1

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error: Could not read file: {e}", file=sys.stderr)
        return 1

    req = Request(
        args.endpoint,
        data=content.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )

    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            error_data = json.loads(e.read().decode("utf-8"))
            msg = error_data.get("error", str(e))
        except Exception:
            msg = str(e)
        print(
            f"Error: Prediction service returned error ({e.code}): {msg}",
            file=sys.stderr,
        )
        return 2
    except URLError:
        print(
            f"Error: Could not connect to prediction service at {args.endpoint}",
            file=sys.stderr,
        )
        return 2

    # Handle dual-model response format (feature 007)
    if "sklearn" in data:
        sklearn = data["sklearn"]
        print("sklearn:")
        print(f"  Predicted price: \u20ac{sklearn['predicted_price_eur']}")
        print(f"  Model version:   {sklearn['model_version']}")
        transformer = data.get("transformer")
        if transformer is not None:
            print("transformer:")
            print(f"  Predicted price: \u20ac{transformer['predicted_price_eur']}")
            print(f"  Model version:   {transformer['model_version']}")
        else:
            print("transformer:")
            print("  not available")
    else:
        # Legacy flat format
        price = data["predicted_price_eur"]
        version = data["model_version"]
        print(f"Predicted price: \u20ac{price}")
        print(f"Model version:   {version}")
    return 0


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

    # Try loading transformer model (optional — graceful degradation)
    transformer_artifact = None
    transformer_dir = Path("models/transformer/")
    transformer_pt = transformer_dir / "model.pt"
    if transformer_pt.exists():
        try:
            from price_predictor.infrastructure.transformer_store import (
                load_model as load_transformer,
            )

            model, config = load_transformer(transformer_dir)
            transformer_artifact = {
                "model": model,
                "config": config,
                "model_version": "transformer-v1",
            }
            print("Transformer model loaded.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load transformer model: {e}", file=sys.stderr)
    else:
        print("No transformer model found — transformer predictions disabled.", file=sys.stderr)

    app = create_app(artifact, transformer_artifact=transformer_artifact)
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


def run_convert(args: argparse.Namespace) -> int:
    """Execute the convert command — launch Java batch converter."""
    import platform
    import subprocess

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    connector_jar = (
        project_root / "forge-connector" / "target"
        / "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
    )

    if not connector_jar.exists():
        print(
            f"Error: Connector JAR not found at {connector_jar}\n"
            "Build it first: cd forge-connector && mvn package -DskipTests",
            file=sys.stderr,
        )
        return 2

    forge_dir = project_root.parent / "forge"
    forge_game_jar = forge_dir / "forge-game" / "target" / "forge-game-2.0.10-SNAPSHOT.jar"
    forge_core_jar = forge_dir / "forge-core" / "target" / "forge-core-2.0.10-SNAPSHOT.jar"
    forge_deps = forge_dir / "forge-game" / "target" / "dependency" / "*"

    for jar in [forge_game_jar, forge_core_jar]:
        if not jar.exists():
            print(f"Error: Forge JAR not found at {jar}", file=sys.stderr)
            return 2

    sep = ";" if platform.system() == "Windows" else ":"
    classpath = sep.join([
        str(connector_jar), str(forge_game_jar), str(forge_core_jar), str(forge_deps),
    ])

    cmd = [
        "java", "-cp", classpath,
        "com.pricepredictor.connector.ConvertMain",
        "--cards-path", args.cards_path,
        "--output-path", args.output_path,
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: Java not found. Ensure java is on PATH.", file=sys.stderr)
        return 2


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


def run_train_transformer(args: argparse.Namespace) -> int:
    """Execute the train-transformer command."""
    from price_predictor.application.train_transformer import train_transformer

    try:
        train_transformer(
            output_dir=Path(args.output_dir),
            forge_cards_path=Path(args.forge_cards_path),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            model_output=Path(args.model_output),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            random_seed=args.random_seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0


def run_evaluate_transformer(args: argparse.Namespace) -> int:
    """Execute the evaluate-transformer command."""
    from price_predictor.application.evaluate_transformer import evaluate_transformer

    model_path = Path(args.model_path)
    try:
        result = evaluate_transformer(
            model_dir=model_path,
            output_dir=Path(args.output_dir),
            forge_cards_path=Path(args.forge_cards_path),
            prices_path=Path(args.prices_path),
            printings_path=Path(args.printings_path),
            random_seed=args.random_seed,
        )
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
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
        "model_path": str(result.model_path / "model.pt"),
        "mean_absolute_error_eur": result.mean_absolute_error_eur,
        "median_percentage_error": result.median_percentage_error,
        "sample_count": result.sample_count,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_check_convert(args: argparse.Namespace) -> int:
    """Execute the check-convert command."""
    from price_predictor.application.check_convert import check_all, format_report

    output_dir = Path(args.output_path)
    cards_dir = Path(args.cards_path)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        return 1
    if not cards_dir.exists():
        print(f"Error: Cards directory not found: {cards_dir}", file=sys.stderr)
        return 1

    print(f"Checking converted files in {output_dir} against {cards_dir}...",
          file=sys.stderr)

    results = check_all(output_dir, cards_dir, threshold=args.threshold)

    report_path = output_dir / "check_report.txt"
    report_path.write_text(format_report(results), encoding="utf-8")
    print(f"Report written to {report_path} ({len(results)} cards with issues)",
          file=sys.stderr)
    print(format_report(results, limit=args.limit))
    return 0
