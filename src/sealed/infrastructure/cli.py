"""CLI interface for the sealed module (encode-cards, generate-pools)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_cards_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory with .npz card embeddings (default: output/cardsfolder/)",
    )


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
    encode_parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing .npz files before encoding (forces full re-encode)",
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

    # ── train-scorer ──────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train-scorer",
        help="Train the deck scorer model on match outcome data",
    )
    train_parser.add_argument(
        "--outcomes-path",
        default="output/sealed/match-outcomes.txt",
        help="Path to match outcomes file (default: output/sealed/match-outcomes.txt)",
    )
    _add_cards_path(train_parser)
    train_parser.add_argument(
        "--checkpoint-dir",
        default="models/sealed/scorer/",
        help="Directory for saving checkpoints (default: models/sealed/scorer/)",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from (default: none)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=2,
        help="Number of Set Transformer SAB layers (default: 2)",
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=4,
        help="Number of attention heads (default: 4)",
    )
    train_parser.add_argument(
        "--n-seeds", type=int, default=4,
        help="Number of PMA seed vectors (default: 4)",
    )
    train_parser.add_argument(
        "--d-ff", type=int, default=1088,
        help="Feed-forward dimension in SAB layers (default: 1088)",
    )
    train_parser.add_argument(
        "--mlp-hidden", type=int, default=256,
        help="Hidden dimension of the scoring MLP head (default: 256)",
    )
    train_parser.add_argument(
        "--val-interval", type=int, default=1,
        help="Run validation every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--unfreeze-embeddings",
        action="store_true",
        help="Enable embedding fine-tuning (Phase B, default: off)",
    )
    train_parser.add_argument(
        "--embedding-lr", type=float, default=1e-5,
        help="Learning rate for embedding fine-tuning (default: 1e-5)",
    )

    # ── evaluate-scorer ──────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "evaluate-scorer",
        help="Evaluate the trained scorer against Forge's deck builder",
    )
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint to evaluate (e.g. best_l2_h4_s4_ff1088_mlp256.pt)",
    )
    _add_cards_path(eval_parser)
    eval_parser.add_argument(
        "--pools",
        type=int,
        default=12,
        help="Number of sealed pools to generate for evaluation (default: 12)",
    )
    eval_parser.add_argument(
        "--best-of",
        type=int,
        default=3,
        help="Number of games per match (default: 3)",
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Java worker processes (default: 4)",
    )
    eval_parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for match files (default: temp dir)",
    )

    # ── match-outcomes ────────────────────────────────────────────
    match_parser = subparsers.add_parser(
        "match-outcomes",
        help="Generate sealed match outcome training data using Forge AI",
    )
    match_parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel Java worker processes to spawn (default: 12)",
    )

    return parser


def run_encode_cards(args: argparse.Namespace) -> int:
    """Execute the encode-cards command."""
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer
    from price_predictor.infrastructure.transformer_store import load_model
    from sealed.application.encode_cards import EncodeCardsUseCase
    from sealed.domain.card_encoder import CardEncoder
    from sealed.infrastructure.embedding_store import EmbeddingStore

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

    if args.clean:
        npz_files = list(cards_path.rglob("*.npz"))
        print(f"Cleaning {len(npz_files)} existing .npz files...")
        for f in npz_files:
            f.unlink()

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

    print(
        f"Done: {result.processed} processed, "
        f"{result.skipped} skipped, {len(result.errors)} errors"
    )

    for err in result.errors:
        print(f"  Error: {err}", file=sys.stderr)

    return 1 if result.errors else 0


def run_generate_pools(args: argparse.Namespace) -> int:
    """Execute the generate-pools command."""
    from sealed.application.generate_pools import GeneratePoolsUseCase
    from sealed.infrastructure.pool_connector import PoolConnector

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


def run_train_scorer(args: argparse.Namespace) -> int:
    """Execute the train-scorer command."""
    from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase

    config = TrainScorerConfig(
        outcomes_path=Path(args.outcomes_path),
        cards_path=Path(args.cards_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        resume=Path(args.resume) if args.resume else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_seeds=args.n_seeds,
        d_ff=args.d_ff,
        mlp_hidden=args.mlp_hidden,
        val_interval=args.val_interval,
        unfreeze_embeddings=args.unfreeze_embeddings,
        embedding_lr=args.embedding_lr,
    )

    try:
        use_case = TrainScorerUseCase()
        use_case.execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def run_evaluate_scorer(args: argparse.Namespace) -> int:
    """Execute the evaluate-scorer command."""
    from sealed.application.evaluate_scorer import EvaluateScorerConfig, EvaluateScorerUseCase

    config = EvaluateScorerConfig(
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        pools=args.pools,
        best_of=args.best_of,
        workers=args.workers,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )

    try:
        use_case = EvaluateScorerUseCase()
        use_case.execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def run_match_outcomes(args: argparse.Namespace) -> int:
    """Execute the match-outcomes command."""
    from sealed.application.match_outcomes import MatchOutcomeSupervisor

    output_path = Path("output") / "sealed" / "match-outcomes.txt"

    supervisor = MatchOutcomeSupervisor(
        worker_count=args.workers,
        output_path=output_path,
    )

    try:
        supervisor.run()
    except FileNotFoundError as exc:
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
    elif args.command == "train-scorer":
        sys.exit(run_train_scorer(args))
    elif args.command == "evaluate-scorer":
        sys.exit(run_evaluate_scorer(args))
    elif args.command == "match-outcomes":
        sys.exit(run_match_outcomes(args))
    else:
        parser.print_help()
        sys.exit(1)
