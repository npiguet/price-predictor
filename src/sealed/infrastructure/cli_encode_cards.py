"""The ``encode-cards`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.cli_helpers import add_dataclass_arg
from sealed.application.encode_cards import EncodeCardsConfig


def _build_encode_cards_parser(subparsers) -> None:
    encode_parser = subparsers.add_parser(
        "encode-cards",
        help="Encode card scripts to .npz embedding files",
    )
    encode_parser.set_defaults(func=run_encode_cards)
    encode_parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "Encoder weights source: a sealed encoder or a price-predictor "
            "transformer checkpoint. One of this and --scorer-checkpoint is "
            "required; they are mutually exclusive."
        ),
    )
    encode_parser.add_argument(
        "--scorer-checkpoint",
        default=None,
        help=(
            "Extract encoder weights from a Phase B scorer checkpoint "
            "Mutually exclusive with --encoder-checkpoint. Rejected if the supplied checkpoint is "
            "Phase A (no encoder_state_dict)."
        ),
    )
    encode_parser.add_argument(
        "--vocab-path",
        required=True,
        help="Path to the tokenizer vocabulary file (the one the encoder was trained with)",
    )
    encode_parser.add_argument(
        "--cards-path",
        required=True,
        help="Directory containing .txt card script files (searched recursively)",
    )
    encode_parser.add_argument(
        "--embeddings-path", default=None,
        help=(
            "Directory the .npz embeddings are written to, laid out like "
            "--cards-path (a cache run directory); default: beside each card "
            "in --cards-path. --clean empties this directory's .npz files only."
        ),
    )
    add_dataclass_arg(
        encode_parser, EncodeCardsConfig, "clean",
        help_text=(
            "Delete all existing .npz files where the embeddings are written "
            "before encoding (forces full re-encode)"
        ),
    )


def _load_encoder_for_encode_cards(encoder_path: Path):
    """Dispatch encoder loading by checkpoint shape.

    Sealed encoders carry ``n_pool_queries`` in their stored config; the
    price-predictor encoder does not. Picking by config schema lets the
    same ``encode-cards`` command consume both checkpoint families
    without a separate flag (FR-027).
    """
    import torch

    raw = torch.load(encoder_path, map_location="cpu", weights_only=False)
    config_dict = raw.get("config")
    if isinstance(config_dict, dict) and "n_pool_queries" in config_dict:
        from sealed.infrastructure.encoder_store import SealedEncoderStore
        return SealedEncoderStore().load_encoder(encoder_path)
    from price_predictor.infrastructure.transformer_store import load_model
    return load_model(encoder_path)


def _model_from_scorer(scorer_path: Path):
    """The encoder a Phase B scorer checkpoint carries, as ``(model, config)``;
    exit 2 when the checkpoint is missing or Phase A."""
    from price_predictor.domain.entities import TransformerConfig
    from price_predictor.infrastructure.transformer_model import (
        CardPriceTransformerModel,
    )
    from sealed.infrastructure.scorer_store import ScorerStore

    if not scorer_path.exists():
        print(f"Error: Scorer checkpoint not found: {scorer_path}", file=sys.stderr)
        return 2
    loaded = ScorerStore().load_checkpoint(scorer_path)
    if loaded.encoder_state_dict is None:
        print(
            f"Error: {scorer_path} is a Phase A scorer checkpoint and "
            "contains no encoder weights. Use --encoder-checkpoint for "
            "non-Phase-B sources.",
            file=sys.stderr,
        )
        return 2
    cfg = TransformerConfig(**loaded.encoder_config)
    model = CardPriceTransformerModel(cfg)
    model.load_state_dict(loaded.encoder_state_dict)
    return model, cfg


def _model_from_encoder(encoder_path: Path):
    """``--encoder-checkpoint``'s encoder as ``(model, config)``; exit 2 when missing."""
    if not encoder_path.exists():
        print(f"Error: Encoder model not found: {encoder_path}", file=sys.stderr)
        return 2
    return _load_encoder_for_encode_cards(encoder_path)


def _load_encode_model(args: argparse.Namespace):
    """The encoder as ``(model, config)`` from whichever of the two checkpoint
    flags is given; exit 2 unless exactly one is, or when it cannot be loaded."""
    if (args.encoder_checkpoint is None) == (args.scorer_checkpoint is None):
        print(
            "Error: --encoder-checkpoint and --scorer-checkpoint are mutually "
            "exclusive and one of them is required: they are the two sources of "
            "the encoder weights.",
            file=sys.stderr,
        )
        return 2

    if args.scorer_checkpoint is not None:
        return _model_from_scorer(Path(args.scorer_checkpoint))
    return _model_from_encoder(Path(args.encoder_checkpoint))


def _missing_input_error(vocab_path: Path, cards_path: Path) -> int | None:
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found: {vocab_path}", file=sys.stderr)
        return 2
    if not cards_path.exists():
        print(f"Error: Cards path not found: {cards_path}", file=sys.stderr)
        return 2
    return None


def _report_progress(processed: int, skipped: int) -> None:
    print(
        f"\rProgress: {processed} encoded ({skipped} skipped)",
        end="",
        flush=True,
    )


def _report_result(result, text_dim: int) -> int:
    """Print the run's totals and errors; exit 1 when a card failed."""
    from sealed.domain.card_embedding_layout import FEATURE_COUNT

    if result.processed > 0 or result.skipped > 0:
        print()

    print(
        f"Done: {result.processed} processed, "
        f"{result.skipped} skipped, {len(result.errors)} errors  "
        f"(embedding dim = {text_dim + FEATURE_COUNT}: {text_dim} text + {FEATURE_COUNT} features)"
    )

    for err in result.errors:
        print(f"  Error: {err}", file=sys.stderr)

    return 1 if result.errors else 0


def run_encode_cards(args: argparse.Namespace) -> int:
    """Execute the encode-cards command."""
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer
    from sealed.application.encode_cards import EncodeCardsUseCase
    from sealed.domain.card_encoder import CardEncoder
    from sealed.infrastructure.embedding_store import EmbeddingStore

    loaded = _load_encode_model(args)
    if isinstance(loaded, int):
        return loaded
    model, config = loaded

    vocab_path = Path(args.vocab_path)
    cards_path = Path(args.cards_path)
    embeddings_path = Path(args.embeddings_path) if args.embeddings_path else None
    input_err = _missing_input_error(vocab_path, cards_path)
    if input_err is not None:
        return input_err

    print(f"Encoding cards in {cards_path}")
    if embeddings_path is not None:
        print(f"Writing their embeddings to {embeddings_path}")

    model.eval()
    tokenizer = load_tokenizer(vocab_path)

    encoder = CardEncoder(model, tokenizer, max_seq_len=config.max_seq_len)
    store = EmbeddingStore()
    use_case = EncodeCardsUseCase()

    result = use_case.execute(
        EncodeCardsConfig(
            cards_path=cards_path, clean=args.clean, embeddings_path=embeddings_path,
        ),
        encoder,
        store,
        progress=_report_progress,
    )
    return _report_result(result, config.pooled_dim)  # SealedEncoderConfig | TransformerConfig
