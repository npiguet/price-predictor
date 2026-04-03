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
        help="Path to the pretrained transformer model .pt file "
             "(default: models/price-predictor/transformer/latest.pt)",
    )
    encode_parser.add_argument(
        "--vocab-path",
        default="models/price-predictor/transformer/vocab.txt",
        help="Path to the tokenizer vocabulary file "
             "(default: models/price-predictor/transformer/vocab.txt)",
    )
    encode_parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory containing .txt card script files (searched recursively) "
             "(default: output/cardsfolder/)",
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
        help="MTG set code to generate boosters from (e.g. RVR, MH3, BLB) (default: RVR)",
    )
    generate_parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of sealed pools to generate (default: 10000)",
    )
    generate_parser.add_argument(
        "--pools-path",
        default=None,
        help="Output directory; pools.txt is written here. Default: output/sealed/pools/{set}/",
    )

    # ── train ─────────────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train",
        help="Train a sealed deck-picker model",
    )
    train_parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help="Training stage (1 or 2)",
    )
    train_parser.add_argument(
        "--set",
        default="RVR",
        dest="set_code",
        help="MTG set code (e.g. RVR, MH3) (default: RVR)",
    )
    train_parser.add_argument(
        "--pools-path",
        default=None,
        help="Directory containing pools.txt (default: output/sealed/pools/{set}/)",
    )
    train_parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory containing .npz card embedding files (default: output/cardsfolder/)",
    )
    train_parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to save/load the .pt model checkpoint. "
            "Default: models/sealed/stage1/{set}/latest.pt (stage 1) or "
            "models/sealed/stage2/{set}/latest.pt (stage 2)"
        ),
    )
    train_parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Path to Stage 1 checkpoint to initialise Stage 2 from. "
            "Default: models/sealed/stage1/{set}/latest.pt. "
            "Only used when --stage 2 and --model-path does not exist."
        ),
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of episodes per training batch (default: 32)",
    )

    # ── validate-embeddings ───────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate-embeddings",
        help="Validate card embeddings using linear probes",
    )
    validate_parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory containing .npz embedding and .txt card text files "
             "(default: output/cardsfolder/)",
    )
    validate_parser.add_argument(
        "--threshold-accuracy",
        type=float,
        default=0.95,
        help="Minimum accuracy for classification probes (default: 0.95)",
    )
    validate_parser.add_argument(
        "--threshold-r2",
        type=float,
        default=0.85,
        help="Minimum R² for regression probes (default: 0.85)",
    )

    # ── sample ────────────────────────────────────────────────────
    sample_parser = subparsers.add_parser(
        "sample",
        help="Sample picks from a trained sealed deck-picker model",
    )
    sample_parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Model stage to sample from (1 or 2) (default: 1)",
    )
    sample_parser.add_argument(
        "--set",
        default="RVR",
        dest="set_code",
        help="MTG set code (e.g. RVR, MH3) (default: RVR)",
    )
    sample_parser.add_argument(
        "--pools-path",
        default=None,
        help="Directory containing pools.txt (default: output/sealed/pools/{set}/)",
    )
    sample_parser.add_argument(
        "--cards-path",
        default="output/cardsfolder/",
        help="Directory containing .npz card embedding files (default: output/cardsfolder/)",
    )
    sample_parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to .pt model checkpoint. "
            "Default: models/sealed/stage1/{set}/latest.pt (stage 1) or "
            "models/sealed/stage2/{set}/latest.pt (stage 2)"
        ),
    )
    sample_parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of sample episodes to display (default: 10)",
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

    print(f"Done: {result.processed} processed, {result.skipped} skipped, {len(result.errors)} errors")

    for err in result.errors:
        print(f"  Error: {err}", file=sys.stderr)

    return 1 if result.errors else 0


_COL_FEATURE = 32
_COL_SCORE = 8
_COL_EXACT = 12
_COL_THRESHOLD = 10


def _print_probe_row(r) -> None:
    status = "PASS" if r.passed else "FAIL"
    threshold_str = f"≥ {r.threshold:.3f}"
    exact_str = f"{r.rounded_score:.3f}" if r.rounded_score is not None else "-"
    print(
        f"{r.feature_name:<{_COL_FEATURE}}  "
        f"{r.score:>{_COL_SCORE}.3f}  "
        f"{exact_str:>{_COL_EXACT}}  "
        f"{threshold_str:>{_COL_THRESHOLD}}  "
        f"{status}"
    )


def run_validate_embeddings(args: argparse.Namespace) -> int:
    """Execute the validate-embeddings command."""
    from sealed.application.validate_embeddings import ValidateEmbeddingsUseCase

    cards_path = Path(args.cards_path)

    if not cards_path.exists():
        print(f"Error: Cards path not found: {cards_path}", file=sys.stderr)
        return 2

    print(f"Loading embeddings from {cards_path} ...")

    use_case = ValidateEmbeddingsUseCase()

    # Print the table header before probes start so results stream in live.
    _print_probe_table_header()

    try:
        result = use_case.execute(
            cards_path=cards_path,
            threshold_accuracy=args.threshold_accuracy,
            threshold_r2=args.threshold_r2,
            on_result=_print_probe_row,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    n_passed = sum(1 for r in result.probe_results if r.passed)
    n_total = len(result.probe_results)
    overall = "PASS" if result.all_passed else "FAIL"
    print(f"\nResult: {overall} ({n_passed}/{n_total} probes passed, "
          f"{result.n_cards:,} cards, {result.n_lands:,} lands)")

    return 0 if result.all_passed else 1


def _print_probe_table_header() -> None:
    header = (
        f"{'Feature':<{_COL_FEATURE}}  {'Score':>{_COL_SCORE}}  "
        f"{'Exact Match':>{_COL_EXACT}}  "
        f"{'Threshold':>{_COL_THRESHOLD}}  Status"
    )
    sep = (
        f"{'─' * _COL_FEATURE}  {'─' * _COL_SCORE}  "
        f"{'─' * _COL_EXACT}  "
        f"{'─' * _COL_THRESHOLD}  {'─' * 6}"
    )
    print()
    print(header)
    print(sep)


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


def run_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    if args.stage not in (1, 2):
        print(f"Error: unknown training stage {args.stage}", file=sys.stderr)
        return 1

    set_code = args.set_code

    if args.pools_path is None:
        pools_path = Path("output") / "sealed" / "pools" / set_code
    else:
        pools_path = Path(args.pools_path.replace("{set}", set_code))

    cards_path = Path(args.cards_path)

    if args.stage == 1:
        from sealed.application.train_stage1 import TrainStage1UseCase

        if args.model_path is None:
            model_path = Path("models") / "sealed" / "stage1" / set_code / "latest.pt"
        else:
            model_path = Path(args.model_path.replace("{set}", set_code))

        try:
            use_case = TrainStage1UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                batch_size=args.batch_size,
                set_code=set_code,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    else:  # stage == 2
        from sealed.application.train_stage2 import TrainStage2UseCase

        if args.model_path is None:
            model_path = Path("models") / "sealed" / "stage2" / set_code / "latest.pt"
        else:
            model_path = Path(args.model_path.replace("{set}", set_code))

        init_from_raw = getattr(args, "init_from", None)
        if init_from_raw is None:
            init_from = Path("models") / "sealed" / "stage1" / set_code / "latest.pt"
        else:
            init_from = Path(init_from_raw.replace("{set}", set_code))

        try:
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=args.batch_size,
                set_code=set_code,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    return 0


def run_sample(args: argparse.Namespace) -> int:
    """Execute the sample command."""
    set_code = args.set_code
    stage = getattr(args, "stage", 1)

    if args.pools_path is None:
        pools_path = Path("output") / "sealed" / "pools" / set_code
    else:
        pools_path = Path(args.pools_path.replace("{set}", set_code))

    cards_path = Path(args.cards_path)

    if stage == 2:
        from sealed.application.sample_stage2 import SampleStage2UseCase

        if args.model_path is None:
            model_path = Path("models") / "sealed" / "stage2" / set_code / "latest.pt"
        else:
            model_path = Path(args.model_path.replace("{set}", set_code))

        try:
            use_case = SampleStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                n_samples=args.n_samples,
            )
        except FileNotFoundError as exc:
            print(f"Error: checkpoint not found: {exc}", file=sys.stderr)
            return 2

    else:
        from sealed.application.sample_stage1 import SampleStage1UseCase

        if args.model_path is None:
            model_path = Path("models") / "sealed" / "stage1" / set_code / "latest.pt"
        else:
            model_path = Path(args.model_path.replace("{set}", set_code))

        try:
            use_case = SampleStage1UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                n_samples=args.n_samples,
            )
        except FileNotFoundError as exc:
            print(f"Error: checkpoint not found: {exc}", file=sys.stderr)
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
    elif args.command == "train":
        sys.exit(run_train(args))
    elif args.command == "sample":
        sys.exit(run_sample(args))
    elif args.command == "validate-embeddings":
        sys.exit(run_validate_embeddings(args))
    else:
        parser.print_help()
        sys.exit(1)
