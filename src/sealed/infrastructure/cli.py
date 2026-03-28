"""CLI interface for the sealed module (encode-cards, generate-pools)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sealed",
        description="Sealed dataset preparation tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── encode-cards ──────────────────────────────────────────────
    encode_parser = subparsers.add_parser(
        "encode-cards",
        help="Encode card scripts to .npz embedding files",
    )
    encode_parser.add_argument(
        "--encoder-path",
        default="models/price-predictor/transformer/latest.pt",
        help="Path to the pretrained transformer model .pt file",
    )
    encode_parser.add_argument(
        "--vocab-path",
        default="models/price-predictor/transformer/vocab.txt",
        help="Path to the tokenizer vocabulary file",
    )
    encode_parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory containing .txt card script files (searched recursively)",
    )

    # ── generate-pools ────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate-pools",
        help="Generate sealed pools using Forge's booster generation logic",
    )
    generate_parser.add_argument(
        "--set",
        default="RVR",
        dest="set_code",
        help="MTG set code to generate boosters from (e.g. RVR, MH3, BLB)",
    )
    generate_parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of sealed pools to generate",
    )
    generate_parser.add_argument(
        "--pools-path",
        default=None,
        help="Output directory; pools.txt is written here. Default: output/sealed/pools/{set}/",
    )

    return parser


def run_encode_cards(args: argparse.Namespace) -> int:
    """Execute the encode-cards command."""
    from price_predictor.infrastructure.transformer_store import load_model
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer
    from sealed.domain.card_encoder import CardEncoder
    from sealed.infrastructure.embedding_store import EmbeddingStore
    from sealed.application.encode_cards import EncodeCardsUseCase

    encoder_path = Path(args.encoder_path)
    vocab_path = Path(args.vocab_path)
    cards_path = Path(args.cards_path)

    if not encoder_path.exists():
        print(f"Error: Encoder model not found: {encoder_path}", file=sys.stderr)
        return 2
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found: {vocab_path}", file=sys.stderr)
        return 2
    if not cards_path.exists():
        print(f"Error: Cards path not found: {cards_path}", file=sys.stderr)
        return 2

    print(f"Encoding cards in {cards_path}")

    # Load model
    model, config = load_model(encoder_path)
    model.eval()

    # Load tokenizer
    tokenizer = load_tokenizer(vocab_path)

    encoder = CardEncoder(model, tokenizer, max_seq_len=config.max_seq_len)
    store = EmbeddingStore()
    use_case = EncodeCardsUseCase()

    result = use_case.execute(cards_path, encoder, store)

    print(f"Done: {result.processed} processed, {result.skipped} skipped, {len(result.errors)} errors")

    for err in result.errors:
        print(f"  Error: {err}", file=sys.stderr)

    return 1 if result.errors else 0


def run_generate_pools(args: argparse.Namespace) -> int:
    """Execute the generate-pools command."""
    import platform
    from sealed.infrastructure.pool_connector import PoolConnector
    from sealed.application.generate_pools import GeneratePoolsUseCase

    set_code = args.set_code
    pool_count = args.size

    if args.pools_path is None:
        pools_path = Path("output") / "sealed" / "pools" / set_code
    else:
        raw = args.pools_path.replace("{set}", set_code)
        pools_path = Path(raw)

    connector = PoolConnector()
    use_case = GeneratePoolsUseCase()

    try:
        use_case.execute(set_code, pool_count, pools_path, connector)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "encode-cards":
        sys.exit(run_encode_cards(args))
    elif args.command == "generate-pools":
        sys.exit(run_generate_pools(args))
    else:
        parser.print_help()
        sys.exit(1)
