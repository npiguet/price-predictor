"""The ``predict`` commands of the ``price_predictor`` CLI: parsers and handlers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from price_predictor.infrastructure.cli_common import require_path_exists

logger = logging.getLogger(__name__)


#: The model each ``predict`` reads by default: the rolling checkpoint its
#: trainer writes by default.
_PREDICT_MODEL_PATHS = {
    "sklearn": "models/price-predictor/sklearn/latest.joblib",
    "transformer": "models/price-predictor/transformer/",
}


def add_predict(subparsers: argparse._SubParsersAction) -> None:
    """Register ``predict {sklearn,transformer}``."""
    predict_parser = subparsers.add_parser("predict",
                                           help="Predict price for a card")
    predict_subparsers = predict_parser.add_subparsers(dest="model", required=True)

    predict_handlers = {
        "sklearn": run_predict_sklearn,
        "transformer": run_predict_transformer,
    }
    for model_name, handler in predict_handlers.items():
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
        p.add_argument("--model-path", type=str, default=_PREDICT_MODEL_PATHS[model_name],
                       help="The model to predict with (default: %(default)s)")
        p.set_defaults(func=handler)


def _load_printing_data_map_optional() -> dict | None:
    """Try to load the printing-data map from default paths. Returns None if unavailable."""
    printings_path = Path("resources/AllPrintings.json")
    prices_path = Path("resources/AllPricesToday.json")
    if not (printings_path.exists() and prices_path.exists()):
        return None
    try:
        from price_predictor.infrastructure.mtgjson_loader import build_printing_data_map
        printing_data_map, _ = build_printing_data_map(printings_path, prices_path)
        return printing_data_map
    except (OSError, ValueError) as e:
        logger.warning("Failed to load printing data: %s", e)
        return None


def _resolve_optional_printing_data(card_name: str):
    """Look up PrintingData for a card name from the default printing-data map.

    Returns None when the printing-data map is unavailable or the card name is
    unknown — both cases mean the caller should fall back to its own default.
    """
    from price_predictor.domain.card_name_resolver import CardNameResolver

    printing_data_map = _load_printing_data_map_optional()
    if not printing_data_map:
        return None
    resolver = CardNameResolver(printing_data_map=printing_data_map)
    if resolver.canonicalize(card_name) is None:
        return None
    return resolver.lookup_printing_data(card_name)


def _read_card_text(args: argparse.Namespace) -> str | None:
    """Resolve the card text from --file or --card. None means a printed error."""
    if args.file:
        file_path = Path(args.file)
        if not require_path_exists(file_path, "File"):
            return None
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Could not read file: %s", e)
            return None
    return args.card


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

    model_path = Path(args.model_path)
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

    model_dir = Path(args.model_path)
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
