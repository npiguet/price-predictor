"""CLI interface for the sealed module (encode-cards, generate-pools)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.cli_helpers import add_dataclass_arg
from sealed.application.encode_cards import EncodeCardsConfig
from sealed.application.evaluate_scorer import EvaluateScorerConfig
from sealed.application.train_scorer import TrainScorerConfig


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
    subparsers = parser.add_subparsers(help="Available commands")

    _build_encode_cards_parser(subparsers)
    _build_generate_pools_parser(subparsers)
    _build_build_decks_parser(subparsers)
    _build_train_scorer_parser(subparsers)
    _build_evaluate_scorer_parser(subparsers)
    _build_match_outcomes_parser(subparsers)

    return parser


def _build_encode_cards_parser(subparsers) -> None:
    encode_parser = subparsers.add_parser(
        "encode-cards",
        help="Encode card scripts to .npz embedding files",
    )
    encode_parser.set_defaults(func=run_encode_cards)
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
    add_dataclass_arg(
        encode_parser, EncodeCardsConfig, "clean",
        help_text="Delete all existing .npz files before encoding (forces full re-encode)",
    )


def _build_generate_pools_parser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        "generate-pools",
        help="Generate sealed pools using Forge's booster generation logic",
    )
    generate_parser.set_defaults(func=run_generate_pools)
    generate_parser.add_argument(
        "--set",
        default=None,
        dest="set_code",
        help=(
            "MTG set code to generate boosters from (e.g. RVR, MH3, BLB). "
            "When omitted, each pool uses a randomly selected sealed-legal set."
        ),
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
        help=(
            "Output directory; pools.txt is written here. "
            "Default: output/sealed/pools/{set}/ when --set is given, "
            "output/sealed/pools/ otherwise."
        ),
    )


def _build_build_decks_parser(subparsers) -> None:
    build_parser = subparsers.add_parser(
        "build-decks",
        help="Build scorer-guided 40-card decks from a sealed pools file",
    )
    build_parser.set_defaults(func=run_build_decks)
    build_parser.add_argument(
        "--pools-path",
        required=True,
        help="Input pools file (with SET_CODE; prefixes)",
    )
    build_parser.add_argument(
        "--checkpoint",
        default="models/sealed/scorer/latest.pt",
        help="Scorer model checkpoint (default: models/sealed/scorer/latest.pt)",
    )
    _add_cards_path(build_parser)
    build_parser.add_argument(
        "--output",
        default="output/sealed/generated-decks.txt",
        help="Output generated-decks file (default: output/sealed/generated-decks.txt)",
    )
    build_parser.add_argument(
        "--sa-temperature",
        type=float,
        default=0.0,
        help=(
            "Initial temperature for simulated annealing. 0 = pure greedy "
            "(default). Try 0.1-1.0 for SA; reasonable values depend on "
            "the magnitude of typical score differences for the model."
        ),
    )
    build_parser.add_argument(
        "--sa-cooling",
        type=float,
        default=0.95,
        help="Per-iteration temperature multiplier (default: 0.95). Ignored if --sa-temperature is 0.",
    )
    build_parser.add_argument(
        "--sa-max-iterations",
        type=int,
        default=200,
        help="Hard cap on swap iterations (default: 200). Pure greedy stops earlier on convergence.",
    )
    build_parser.add_argument(
        "--print-decks",
        action="store_true",
        help=(
            "After building, also print each deck to stdout in the human-"
            "readable format used by evaluate-scorer (=== Deck N  score=... === "
            "header + one card per line with mana cost)."
        ),
    )


def _build_train_scorer_parser(subparsers) -> None:
    train_parser = subparsers.add_parser(
        "train-scorer",
        help="Train the deck scorer model on match outcome data",
    )
    train_parser.set_defaults(func=run_train_scorer)
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
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "epochs", "Number of training epochs",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "batch_size", "Training batch size",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "lr", "Learning rate", type_override=float,
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "n_layers",
        "Number of Set Transformer SAB layers",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "n_heads", "Number of attention heads",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "n_seeds", "Number of PMA seed vectors",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "d_ff",
        "Feed-forward dimension in SAB layers",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "mlp_hidden",
        "Hidden dimension of the scoring MLP head",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "dropout",
        "Dropout rate applied in SAB attention/FF, PMA attention, and scoring MLP",
        type_override=float,
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "val_interval",
        "Run validation every N epochs",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "unfreeze_embeddings",
        help_text="Enable embedding fine-tuning (Phase B)",
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "embedding_lr",
        help_text="Learning rate for embedding fine-tuning", type_override=float,
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "val_fraction",
        help_text="Fraction of examples held out for validation", type_override=float,
    )
    add_dataclass_arg(
        train_parser, TrainScorerConfig, "random_seed",
        help_text="RNG seed for the train/val split",
    )


def _build_evaluate_scorer_parser(subparsers) -> None:
    eval_parser = subparsers.add_parser(
        "evaluate-scorer",
        help="Evaluate the trained scorer against Forge's deck builder",
    )
    eval_parser.set_defaults(func=run_evaluate_scorer)
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint to evaluate (e.g. best_l2_h4_s4_ff1088_mlp256.pt)",
    )
    _add_cards_path(eval_parser)
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "pools",
        help_text="Number of sealed pools to generate for evaluation",
    )
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "best_of",
        help_text="Number of games per match",
    )
    add_dataclass_arg(
        eval_parser, EvaluateScorerConfig, "workers",
        help_text="Number of parallel Java worker processes",
    )
    eval_parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for match files (default: temp dir)",
    )
    eval_parser.add_argument(
        "--set",
        default=None,
        dest="set_code",
        help=(
            "MTG set code to evaluate on (e.g. RVR, MH3, BLB). When omitted,"
            " a random sealed-legal set is selected."
        ),
    )


def _build_match_outcomes_parser(subparsers) -> None:
    match_parser = subparsers.add_parser(
        "match-outcomes",
        help="Generate sealed match outcome training data using Forge AI",
    )
    match_parser.set_defaults(func=run_match_outcomes)
    match_parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel Java worker processes to spawn (default: 12)",
    )
    match_parser.add_argument(
        "--generated-decks-path",
        default=None,
        help=(
            "Optional path to a generated-decks.txt file. When given, workers run"
            " in self-play mode: deck A is sampled from this file each match and"
            " deck B is built by one of 5 weighted methods (4:3:2:1:4) with"
            " same-set pairing. When omitted, Phase 0 random-pool behavior is"
            " unchanged."
        ),
    )
    match_parser.add_argument(
        "--self-play-label",
        default=None,
        help=(
            "Label recorded as the method tag for scorer-built decks in self-play"
            " mode (e.g. 'gen-2'). Required iff --generated-decks-path is given;"
            " forbidden otherwise. Enables distinguishing self-play matches from"
            " different scorer generations in the combined training corpus."
        ),
    )
    match_parser.add_argument(
        "--best-of",
        type=int,
        default=7,
        help=(
            "Number of games per match (best-of-N). Must be a positive odd integer."
            " Default: 7."
        ),
    )


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

    print(f"Encoding cards in {cards_path}")

    model, config = load_model(encoder_path)
    model.eval()
    tokenizer = load_tokenizer(vocab_path)

    encoder = CardEncoder(model, tokenizer, max_seq_len=config.max_seq_len)
    store = EmbeddingStore()
    use_case = EncodeCardsUseCase()

    def report(processed: int, skipped: int) -> None:
        print(
            f"\rProgress: {processed} encoded ({skipped} skipped)",
            end="",
            flush=True,
        )

    result = use_case.execute(
        EncodeCardsConfig(cards_path=cards_path, clean=args.clean),
        encoder,
        store,
        progress=report,
    )
    if result.processed > 0 or result.skipped > 0:
        print()

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
        if set_code is None:
            pools_path = Path("output") / "sealed" / "pools"
        else:
            pools_path = Path("output") / "sealed" / "pools" / set_code
    else:
        raw = args.pools_path.replace("{set}", set_code) if set_code else args.pools_path
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


def run_build_decks(args: argparse.Namespace) -> int:
    """Execute the build-decks command."""
    from sealed.application.build_decks import BuildDecksConfig, BuildDecksUseCase

    config = BuildDecksConfig(
        pools_path=Path(args.pools_path),
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        output=Path(args.output),
        sa_temperature=args.sa_temperature,
        sa_cooling=args.sa_cooling,
        sa_max_iterations=args.sa_max_iterations,
        print_decks=args.print_decks,
    )

    try:
        use_case = BuildDecksUseCase()
        written = use_case.execute(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {written} decks to {config.output}")
    return 0


def run_train_scorer(args: argparse.Namespace) -> int:
    """Execute the train-scorer command."""
    from sealed.application.train_scorer import TrainScorerUseCase

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
        dropout=args.dropout,
        val_interval=args.val_interval,
        unfreeze_embeddings=args.unfreeze_embeddings,
        embedding_lr=args.embedding_lr,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
    )

    try:
        use_case = TrainScorerUseCase()
        use_case.execute(config)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def run_evaluate_scorer(args: argparse.Namespace) -> int:
    """Execute the evaluate-scorer command."""
    from sealed.application.evaluate_scorer import EvaluateScorerUseCase

    config = EvaluateScorerConfig(
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        pools=args.pools,
        best_of=args.best_of,
        workers=args.workers,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        set_code=args.set_code,
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

    if args.best_of < 1 or args.best_of % 2 == 0:
        print(
            f"Error: --best-of must be a positive odd integer, got: {args.best_of}",
            file=sys.stderr,
        )
        return 2

    # XOR check: --self-play-label is required iff --generated-decks-path is given.
    if args.generated_decks_path and not args.self_play_label:
        print(
            "Error: --self-play-label is required when --generated-decks-path is given",
            file=sys.stderr,
        )
        return 2
    if not args.generated_decks_path and args.self_play_label:
        print(
            "Error: --self-play-label is only valid with --generated-decks-path",
            file=sys.stderr,
        )
        return 2

    output_path = Path("output") / "sealed" / "match-outcomes.txt"
    generated_decks_path = (
        Path(args.generated_decks_path) if args.generated_decks_path else None
    )

    supervisor = MatchOutcomeSupervisor(
        worker_count=args.workers,
        output_path=output_path,
        best_of=args.best_of,
        generated_decks_path=generated_decks_path,
        self_play_label=args.self_play_label,
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
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
