"""The ``serve`` command of the ``price_predictor`` CLI: parser and handler."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from price_predictor.infrastructure.cli_predict import _PREDICT_MODEL_PATHS

logger = logging.getLogger(__name__)


def add_serve(subparsers: argparse._SubParsersAction) -> None:
    """Register ``serve``."""
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
    serve_parser.add_argument("--transformer-path", type=str,
                              default=_PREDICT_MODEL_PATHS["transformer"],
                              help="Directory of the optional transformer's latest.pt "
                                   "(default: %(default)s)")
    serve_parser.set_defaults(func=run_serve)


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
        Path(args.vocab_path), Path(args.transformer_path),
    )

    printing_data_map = None
    printings_path = Path(args.printings_path)
    prices_path = Path(args.prices_path)
    if printings_path.exists() and prices_path.exists():
        try:
            from price_predictor.infrastructure.mtgjson_loader import build_printing_data_map
            printing_data_map_result, _ = build_printing_data_map(printings_path, prices_path)
            printing_data_map = printing_data_map_result
            logger.info("Printing data loaded (%d cards).", len(printing_data_map))
        except (OSError, ValueError) as e:
            logger.warning("Failed to build printing data: %s", e)
    else:
        logger.info("MTGJSON files not found — printing-data lookup disabled.")

    app = create_app(artifact, transformer_artifact=transformer_artifact,
                      printing_data_map=printing_data_map, tokenizer=tokenizer)
    logger.info("Prediction service started on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _load_optional_transformer_bundle(
    vocab_path: Path, transformer_dir: Path = Path(_PREDICT_MODEL_PATHS["transformer"]),
) -> tuple[dict[str, Any] | None, Any | None]:
    """Load the transformer artifact + tokenizer for `serve`. Either may be None."""
    return _load_optional_transformer(Path(transformer_dir)), _load_optional_tokenizer(vocab_path)


def _load_optional_transformer(transformer_dir: Path) -> dict[str, Any] | None:
    """The transformer artifact for `serve`, or None when absent or unloadable."""
    if not (transformer_dir / "latest.pt").exists():
        logger.info("No transformer model found — transformer predictions disabled.")
        return None
    try:
        from price_predictor.infrastructure.transformer_store import (
            load_model as load_transformer,
        )
        model, config = load_transformer(transformer_dir)
    except Exception as e:
        logger.warning("Failed to load transformer model: %s", e)
        return None
    logger.info("Transformer model loaded.")
    return {"model": model, "config": config, "model_version": "transformer-v1"}


def _load_optional_tokenizer(vocab_path: Path) -> Any | None:
    """The tokenizer for `serve`, or None when its vocabulary is absent or unloadable."""
    if not vocab_path.exists():
        logger.warning(
            "Vocabulary not found at %s — transformer encoding may fail.",
            vocab_path,
        )
        return None
    try:
        from price_predictor.infrastructure.tokenizer_store import load_tokenizer
        tokenizer = load_tokenizer(vocab_path)
    except Exception as e:
        logger.warning("Failed to load tokenizer: %s", e)
        return None
    logger.info(
        "Tokenizer loaded from %s (%d tokens).",
        vocab_path, tokenizer.vocab_size,
    )
    return tokenizer
