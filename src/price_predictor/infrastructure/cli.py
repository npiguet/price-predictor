"""CLI interface for the card price predictor."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _require_path_exists(path: Path, label: str) -> bool:
    """Print an error and return False when ``path`` is missing; True otherwise."""
    if path.exists():
        return True
    logger.error("%s not found: %s", label, path)
    return False


def _add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add the input/split flags shared by every train/evaluate subcommand."""
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Converted card text directory")
    parser.add_argument("--prices-path", type=str,
                        default="resources/AllPricesToday.json")
    parser.add_argument("--printings-path", type=str,
                        default="resources/AllPrintings.json")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="price_predictor",
        description="MTG card price predictor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train {model} ─────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train model on card data")
    train_subparsers = train_parser.add_subparsers(dest="model", required=True)

    # train sklearn
    train_sklearn = train_subparsers.add_parser("sklearn",
                                                 help="Train sklearn model")
    _add_shared_dataset_args(train_sklearn)
    train_sklearn.add_argument("--model-output", type=str,
                               default="./models/price-predictor/sklearn/")
    train_sklearn.set_defaults(func=run_train_sklearn)

    # train transformer
    train_transformer = train_subparsers.add_parser("transformer",
                                                     help="Train transformer model")
    _add_shared_dataset_args(train_transformer)
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

    # ── predict {model} ──────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict",
                                           help="Predict price for a card")
    predict_subparsers = predict_parser.add_subparsers(dest="model", required=True)

    _PREDICT_HANDLERS = {
        "sklearn": run_predict_sklearn,
        "transformer": run_predict_transformer,
    }
    for model_name, handler in _PREDICT_HANDLERS.items():
        p = predict_subparsers.add_parser(model_name,
                                          help=f"Predict using {model_name} model")
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--file", "-f", type=str,
                           help="Path to converted card text file")
        group.add_argument("--card", "-c", type=str,
                           help="Inline multiline converted card text")
        if model_name == "transformer":
            p.add_argument("--vocab-path", type=str,
                           default="models/price-predictor/transformer/vocab.txt",
                           help="Path to vocab.txt")
        p.set_defaults(func=handler)

    # ── evaluate {model} ─────────────────────────────────────────
    evaluate_parser = subparsers.add_parser("evaluate",
                                            help="Evaluate model accuracy")
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="model", required=True)

    # evaluate sklearn
    eval_sklearn = evaluate_subparsers.add_parser("sklearn",
                                                   help="Evaluate sklearn model")
    _add_shared_dataset_args(eval_sklearn)
    eval_sklearn.add_argument("--model-path", type=str,
                              default="./models/price-predictor/sklearn/latest.joblib")
    eval_sklearn.add_argument("--output-csv", type=str, default=None)
    eval_sklearn.set_defaults(func=run_evaluate_sklearn)

    # evaluate transformer
    eval_transformer = evaluate_subparsers.add_parser(
        "transformer", help="Evaluate transformer model")
    _add_shared_dataset_args(eval_transformer)
    eval_transformer.add_argument("--model-path", type=str,
                                  default="./models/price-predictor/transformer/")
    eval_transformer.add_argument("--vocab-path", type=str,
                                  default="models/price-predictor/transformer/vocab.txt",
                                  help="Path to vocab.txt")
    eval_transformer.set_defaults(func=run_evaluate_transformer)

    # ── serve ─────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve",
                                         help="Start the prediction REST service")
    serve_parser.add_argument("--model-path", type=str,
                              default="models/price-predictor/sklearn/latest.joblib")
    serve_parser.add_argument("--printings-path", type=str,
                              default="resources/AllPrintings.json")
    serve_parser.add_argument("--prices-path", type=str,
                              default="resources/AllPricesToday.json")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--vocab-path", type=str,
                              default="models/price-predictor/transformer/vocab.txt",
                              help="Path to vocab.txt for transformer predictions")
    serve_parser.set_defaults(func=run_serve)

    # ── vocabulary ────────────────────────────────────────────────
    vocab_parser = subparsers.add_parser(
        "vocabulary", help="Build custom MTG tokenizer vocabulary from card corpus"
    )
    vocab_parser.add_argument(
        "--output-dir", type=str, default="models/price-predictor/transformer/",
        help="Directory where vocab.txt is written",
    )
    vocab_parser.add_argument(
        "--cards-path", type=str, default="./output/cardsfolder",
        help="Path to converted card corpus (directory of .txt files)",
    )
    vocab_parser.add_argument(
        "--freq-threshold", type=int, default=5,
        help="Minimum corpus occurrences for a word to be included as a token",
    )
    vocab_parser.add_argument(
        "--printings-path", type=str, default="resources/AllPrintings.json",
        help=(
            "Path to AllPrintings.json. When present, alphabetic fragments of "
            "every set code are seeded into the vocabulary so enriched training "
            "texts never produce UNK for set codes. Silently skipped if absent."
        ),
    )
    vocab_parser.set_defaults(func=run_vocabulary)

    # ── convert ───────────────────────────────────────────────────
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
    convert_parser.set_defaults(func=run_convert)

    # ── check-convert ─────────────────────────────────────────────
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
    check_parser.set_defaults(func=run_check_convert)

    return parser


def run_vocabulary(args: argparse.Namespace) -> int:
    """Execute the vocabulary command — build MTG tokenizer vocabulary from card corpus."""
    from price_predictor.application.build_vocabulary import build_vocabulary
    from price_predictor.infrastructure.tokenizer_store import save_vocabulary

    cards_path = Path(args.cards_path)
    if not _require_path_exists(cards_path, "Cards path"):
        return 1

    txt_files = list(cards_path.rglob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", cards_path)
        return 1

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Could not create output directory %s: %s", output_dir, e)
        return 2

    printings_path_raw = getattr(args, "printings_path", "resources/AllPrintings.json")
    printings_path: Path | None = Path(printings_path_raw)
    if not printings_path.exists():
        logger.warning(
            "AllPrintings.json not found at %s — set-code tokens will not be seeded.",
            printings_path,
        )
        printings_path = None

    result = build_vocabulary(
        cards_path,
        freq_threshold=args.freq_threshold,
        printings_path=printings_path,
    )
    vocab_path = output_dir / "vocab.txt"
    save_vocabulary(result.vocab, vocab_path)

    output = {
        "vocab_path": str(vocab_path),
        "vocab_size": result.vocab_size,
        "domain_token_count": result.domain_token_count,
        "freq_threshold_token_count": result.freq_threshold_token_count,
        "coverage_pct": result.coverage_pct,
        "unk_pct": result.unk_pct,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Execute the serve command."""
    import uvicorn

    from price_predictor.infrastructure.model_store import ModelNotFoundError, load_model
    from price_predictor.infrastructure.server import create_app

    model_path = Path(args.model_path)
    logger.info("Loading model from %s...", model_path)

    try:
        artifact = load_model(model_path)
    except ModelNotFoundError:
        logger.error("Model file not found at %s", model_path)
        return 2

    artifact["model_version"] = model_path.stem

    transformer_artifact, tokenizer = _load_optional_transformer_bundle(
        Path(args.vocab_path)
    )

    metadata_map = None
    printings_path = Path(args.printings_path)
    prices_path = Path(args.prices_path)
    if printings_path.exists() and prices_path.exists():
        try:
            from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
            metadata_map_result, _ = build_metadata_map(printings_path, prices_path)
            metadata_map = metadata_map_result
            logger.info("Metadata map loaded (%d cards).", len(metadata_map))
        except (OSError, ValueError) as e:
            logger.warning("Failed to build metadata map: %s", e)
    else:
        logger.info("Metadata files not found — auto-fill disabled.")

    app = create_app(artifact, transformer_artifact=transformer_artifact,
                      metadata_map=metadata_map, tokenizer=tokenizer)
    logger.info("Prediction service started on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def run_convert(args: argparse.Namespace) -> int:
    """Execute the convert command — launch Java batch converter."""
    import subprocess

    from price_predictor.infrastructure.forge_jvm import (
        build_forge_classpath,
        build_jvm_command,
    )

    try:
        # ConvertMain calls ForgeEnvironmentInitializer.initialize(), which
        # uses GuiBase / GuiHeadless / FModel — all in forge-gui (and FModel
        # pulls in forge-ai subsystems), so the full runtime is required.
        classpath = build_forge_classpath(
            include_full_runtime=True,
            include_dependency_glob=True,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    cmd = build_jvm_command(
        main_class="com.pricepredictor.connector.ConvertMain",
        classpath=classpath,
        main_args=[
            "--cards-path", args.cards_path,
            "--output-path", args.output_path,
        ],
    )

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        logger.error("Java not found. Ensure java is on PATH.")
        return 2


def run_check_convert(args: argparse.Namespace) -> int:
    """Execute the check-convert command."""
    from price_predictor.application.check_convert import check_all, format_report

    output_dir = Path(args.output_path)
    cards_dir = Path(args.cards_path)

    if not _require_path_exists(output_dir, "Output directory"):
        return 1
    if not _require_path_exists(cards_dir, "Cards directory"):
        return 1

    logger.info("Checking converted files in %s against %s...", output_dir, cards_dir)

    results = check_all(output_dir, cards_dir, threshold=args.threshold)

    report_path = output_dir / "check_report.txt"
    report_path.write_text(format_report(results), encoding="utf-8")
    logger.info("Report written to %s (%d cards with issues)", report_path, len(results))
    print(format_report(results, limit=args.limit))
    return 0


def run_train_sklearn(args: argparse.Namespace) -> int:
    """Train the sklearn model using converted card text files."""
    from price_predictor.application.train import TrainModelUseCase

    output_dir = Path(args.output_dir)
    if not _require_path_exists(output_dir, "Converted cards directory"):
        logger.error("Run 'convert' first to generate converted card text files.")
        return 1
    prices_path = Path(args.prices_path)
    if not _require_path_exists(prices_path, "Prices file"):
        return 1
    printings_path = Path(args.printings_path)
    if not _require_path_exists(printings_path, "Printings file"):
        return 1
    try:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=output_dir,
            prices_path=prices_path,
            printings_path=printings_path,
            output_path=Path(args.model_output),
            test_split=args.test_split,
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


def _load_metadata_map_optional() -> dict | None:
    """Try to load metadata map from default paths. Returns None if unavailable."""
    printings_path = Path("resources/AllPrintings.json")
    prices_path = Path("resources/AllPricesToday.json")
    if not (printings_path.exists() and prices_path.exists()):
        return None
    try:
        from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
        metadata_map, _ = build_metadata_map(printings_path, prices_path)
        return metadata_map
    except (OSError, ValueError) as e:
        logger.warning("Failed to load metadata map: %s", e)
        return None


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


def _resolve_optional_printing_data(card_name: str):
    """Look up PrintingData for a card name from the default metadata map.

    Returns None when the metadata map is unavailable or the card name is
    unknown — both cases mean the caller should fall back to its own default.
    """
    from price_predictor.domain.card_name_resolver import CardNameResolver

    metadata_map = _load_metadata_map_optional()
    if not metadata_map:
        return None
    resolver = CardNameResolver(metadata_map=metadata_map)
    if resolver.canonicalize(card_name) is None:
        return None
    return resolver.lookup_printing_data(card_name)


def _read_card_text(args: argparse.Namespace) -> str | None:
    """Resolve the card text from --file or --card. None means a printed error."""
    if args.file:
        file_path = Path(args.file)
        if not _require_path_exists(file_path, "File"):
            return None
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Could not read file: %s", e)
            return None
    return args.card


def _load_optional_transformer_bundle(
    vocab_path: Path,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Load the transformer artifact + tokenizer for `serve`. Either may be None."""
    transformer_artifact: dict[str, Any] | None = None
    transformer_dir = Path("models/price-predictor/transformer/")
    transformer_pt = transformer_dir / "latest.pt"
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
            logger.info("Transformer model loaded.")
        except Exception as e:
            logger.warning("Failed to load transformer model: %s", e)
    else:
        logger.info("No transformer model found — transformer predictions disabled.")

    tokenizer: Any | None = None
    if vocab_path.exists():
        try:
            from price_predictor.infrastructure.tokenizer_store import load_tokenizer
            tokenizer = load_tokenizer(vocab_path)
            logger.info(
                "Tokenizer loaded from %s (%d tokens).",
                vocab_path, tokenizer.vocab_size,
            )
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s", e)
    else:
        logger.warning(
            "Tokenizer vocab not found at %s — transformer encoding may fail.",
            vocab_path,
        )

    return transformer_artifact, tokenizer


def run_predict_sklearn(args: argparse.Namespace) -> int:
    """Predict with the sklearn model using converted card text."""
    from price_predictor.application.predict import PredictPriceUseCase
    from price_predictor.domain.card_text import ConvertedCardText
    from price_predictor.infrastructure.converted_card_parser import parse_converted_text
    from price_predictor.infrastructure.model_store import ModelNotFoundError

    card_text = _read_card_text(args)
    if card_text is None:
        return 1

    try:
        card = parse_converted_text(ConvertedCardText(card_text))
    except ValueError as e:
        logger.error("Failed to parse card text: %s", e)
        return 1

    printing_data = _resolve_optional_printing_data(card.name)
    if printing_data is not None:
        from dataclasses import replace
        card = replace(card, printing_data=printing_data)

    model_path = Path("models/price-predictor/sklearn/latest.joblib")
    try:
        use_case = PredictPriceUseCase()
        result = use_case.execute(card, model_path)
    except ModelNotFoundError:
        logger.error("Model not found at %s. Run 'train sklearn' first.", model_path)
        return 2

    output = {
        "predicted_price_eur": result.predicted_price_eur,
        "model_version": result.model_version,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_predict_transformer(args: argparse.Namespace) -> int:
    """Predict with the transformer model using raw card text."""
    from price_predictor.application.predict_transformer import PredictTransformerUseCase
    from price_predictor.domain.value_objects import PrintingData
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer

    card_text = _read_card_text(args)
    if card_text is None:
        return 1

    from price_predictor.domain.tokenizer import extract_card_name
    card_name = extract_card_name(card_text)
    printing_data: PrintingData | None = None
    if card_name:
        printing_data = _resolve_optional_printing_data(card_name)
    if printing_data is None:
        printing_data = PrintingData.defaults()

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        logger.error(
            "Vocabulary file not found at %s. "
            "Run 'python -m price_predictor vocabulary' first.",
            vocab_path,
        )
        return 1

    try:
        tokenizer = load_tokenizer(vocab_path)
    except (OSError, ValueError) as e:
        logger.error("Failed to load tokenizer: %s", e)
        return 1

    model_dir = Path("models/price-predictor/transformer/")
    try:
        use_case = PredictTransformerUseCase()
        result = use_case.execute(card_text, model_dir, tokenizer=tokenizer,
                                  printing_data=printing_data)
    except FileNotFoundError:
        logger.error("Model not found at %s. Run 'train transformer' first.", model_dir)
        return 2

    output = {
        "predicted_price_eur": result.predicted_price_eur,
        "model_version": result.model_version,
    }
    print(json.dumps(output, indent=2))
    return 0


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
            test_split=args.test_split,
            random_seed=args.random_seed,
        )
    except ModelNotFoundError:
        logger.error("Model file not found at %s", model_path)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 2

    if hasattr(args, 'output_csv') and args.output_csv and result.per_card:
        _write_per_card_csv(Path(args.output_csv), result.per_card)

    output = {
        "model_name": "sklearn",
        "model_version": result.model_version,
        "mean_absolute_error_eur": result.metrics.mean_absolute_error_eur,
        "median_percentage_error": result.metrics.median_percentage_error,
        "median_abs_error_log": result.metrics.median_abs_error_log,
        "top_20_overlap": result.metrics.top_20_overlap,
        "sample_count": result.metrics.sample_count,
    }
    print(json.dumps(output, indent=2))
    return 0


def run_evaluate_transformer(args: argparse.Namespace) -> int:
    """Evaluate the transformer model on held-out validation data."""
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
        "median_abs_error_log": result.median_abs_error_log,
        "top_20_overlap": result.top_20_overlap,
        "sample_count": result.sample_count,
    }
    print(json.dumps(output, indent=2))

    if result.per_bucket:
        print()
        print(result.format_per_bucket_table())

    return 0
