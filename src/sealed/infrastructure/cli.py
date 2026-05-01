"""CLI interface for the sealed module (encode-cards, generate-pools)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_predictor.infrastructure.cli_helpers import add_dataclass_arg
from sealed.application.encode_cards import EncodeCardsConfig
from sealed.application.evaluate_scorer import EvaluateScorerConfig
from sealed.application.train_scorer import TrainScorerConfig
from sealed.domain.greedy_deck_builder import COLOR_PAIRS_STRATEGY
from sealed.infrastructure.match_worker_connector import DEFAULT_SIDE_B_DECKS_WEIGHT


def _parse_restarts(value: str) -> int | str:
    """Parse --restarts: a positive integer, or the literal 'color-pairs'."""
    if value == COLOR_PAIRS_STRATEGY:
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--restarts must be a positive integer or "
            f"{COLOR_PAIRS_STRATEGY!r}, got: {value!r}"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--restarts must be >= 1, got: {n}"
        )
    return n


def _parse_label(value: str) -> str:
    """Parse --label: a non-empty string with no ';', '|', or whitespace."""
    if not value:
        raise argparse.ArgumentTypeError("--label must be a non-empty string")
    forbidden = {";", "|"}
    if any(ch in forbidden for ch in value):
        raise argparse.ArgumentTypeError(
            f"--label must not contain ';' or '|' (file-format separators), "
            f"got: {value!r}"
        )
    if any(ch.isspace() for ch in value):
        raise argparse.ArgumentTypeError(
            f"--label must not contain whitespace, got: {value!r}"
        )
    return value


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
        "--encoder-checkpoint",
        default=None,
        help=(
            "Encoder weights source (default: "
            "models/price-predictor/transformer/latest.pt). "
            "Mutually exclusive with --scorer-checkpoint when both are "
            "explicitly passed."
        ),
    )
    encode_parser.add_argument(
        "--scorer-checkpoint",
        default=None,
        help=(
            "Extract encoder weights from a Phase B scorer checkpoint "
            "(no default). Mutually exclusive with an explicit "
            "--encoder-checkpoint. Rejected if the supplied checkpoint is "
            "Phase A (no encoder_state_dict)."
        ),
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
        "--label",
        type=_parse_label,
        required=True,
        help=(
            "Generation-method tag written as the first column of every output "
            "line (e.g. 'gen-3'). Recorded by match-outcomes self-play as the "
            "method tag for any match where this deck is sampled. Must be a "
            "non-empty string without ';', '|', or whitespace."
        ),
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
        default=0.8,
        help=(
            "Initial temperature for simulated annealing (default: 0.8). "
            "Set to 0 for pure greedy. Try 0.1-1.0 for SA; reasonable values "
            "depend on the magnitude of typical score differences for the model."
        ),
    )
    build_parser.add_argument(
        "--sa-cooling",
        type=float,
        default=0.85,
        help=(
            "Per-iteration temperature multiplier (default: 0.85). "
            "Ignored if --sa-temperature is 0. Lower cooling = faster decay = "
            "shorter exploration window. Below ~0.85 SA may flee good basins "
            "without finding new ones, which can produce worse decks than greedy."
        ),
    )
    build_parser.add_argument(
        "--sa-max-iterations",
        type=int,
        default=200,
        help=(
            "Hard cap on swap iterations (default: 200). "
            "Pure greedy stops earlier on convergence."
        ),
    )
    build_parser.add_argument(
        "--restarts",
        type=_parse_restarts,
        default=1,
        help=(
            "Restart strategy. Either a positive integer N (run the greedy/SA "
            "search N times per pool from independent random inits; default: 1) "
            "or the literal 'color-pairs' (run one search per MTG two-color "
            "pair, each seeded with an initial 23-spell deck filtered to "
            "on-color cards; the search is unconstrained after init). The "
            "highest-scoring deck across runs is kept."
        ),
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
    build_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from a partial run. Counts complete lines already in "
            "--output, skips that many pools from the front of --pools-path, "
            "and appends remaining decks to the existing file. Without this "
            "flag the output file is truncated."
        ),
    )


_TRAIN_SCORER_ARCHITECTURE_FLAGS: tuple[str, ...] = (
    "n_layers", "n_heads", "n_seeds", "d_ff", "mlp_hidden", "dropout",
)


def _build_train_scorer_parser(subparsers) -> None:
    train_parser = subparsers.add_parser(
        "train-scorer",
        help="Train the deck scorer model on match outcome data",
    )
    train_parser.set_defaults(func=run_train_scorer)
    # Every resumable training flag registers with default=None (sentinel) so
    # `run_train_scorer` can apply resume-precedence resolution: explicit CLI
    # > resumed `train_config` > dataclass default (FR-010,
    # contracts/checkpoint-format.md §Resume Precedence).
    train_parser.add_argument(
        "--outcomes-path",
        default=None,
        help=(
            "Path to match outcomes file "
            "(default: output/sealed/match-outcomes.txt). "
            "Resumable: on --resume, falls back to the resumed checkpoint's "
            "value when omitted."
        ),
    )
    train_parser.add_argument(
        "--cards-path",
        default=None,
        help=(
            "Directory with .npz card embeddings (default: output/cardsfolder/). "
            "Resumable."
        ),
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory for saving checkpoints (default: models/sealed/scorer/). "
            "Resumable."
        ),
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Path to checkpoint to resume training from. Mutually exclusive with "
            "--scorer-checkpoint. Phase-locked: a Phase A checkpoint may only "
            "resume with --embedding-lr 0; a Phase B checkpoint requires a "
            "non-zero --embedding-lr."
        ),
    )
    train_parser.add_argument(
        "--scorer-checkpoint",
        default=None,
        help=(
            "Bootstrap scorer weights from a Phase A checkpoint to start a "
            "fresh Phase B run (no default). Mutually exclusive with --resume "
            "and with any architecture flag (architecture is inherited from "
            "the checkpoint's stored config)."
        ),
    )
    train_parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "Source of encoder weights when starting a fresh Phase B run via "
            "--scorer-checkpoint (default: "
            "models/price-predictor/transformer/latest.pt). "
            "No effect on Phase A. Forbidden when explicitly passed alongside "
            "a Phase B --resume (the resumed checkpoint already carries "
            "fine-tuned encoder weights)."
        ),
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 100). Resumable.",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size (default: 64). Resumable.",
    )
    train_parser.add_argument(
        "--lr", type=float, default=None,
        help="Scorer learning rate (default: 1e-5). Resumable.",
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=None,
        help=(
            "Number of Set Transformer SAB layers (default: 6). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=None,
        help=(
            "Number of attention heads (default: 4). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--n-seeds", type=int, default=None,
        help=(
            "Number of PMA seed vectors (default: 4). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--d-ff", type=int, default=None,
        help=(
            "Feed-forward dimension in SAB layers (default: 1088). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--mlp-hidden", type=int, default=None,
        help=(
            "Hidden dimension of the scoring MLP head (default: 256). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Dropout rate (default: 0.2). "
            "Forbidden alongside --resume or --scorer-checkpoint."
        ),
    )
    train_parser.add_argument(
        "--embedding-lr", type=float, default=None,
        help=(
            "Encoder parameter group's learning rate (default: 0.0). "
            "0 = Phase A (encoder frozen, consumes the .npz cache). "
            "Non-zero = Phase B (encoder fine-tuned alongside the scorer); "
            "Phase B requires --scorer-checkpoint xor --resume."
        ),
    )
    train_parser.add_argument(
        "--patience", type=int, default=None,
        help=(
            "Early-stop after this many consecutive epochs without a new "
            "peak val_acc (default: 5). Resumable."
        ),
    )
    train_parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="Fraction of examples held out for validation (default: 0.2). Resumable.",
    )
    train_parser.add_argument(
        "--random-seed", type=int, default=None,
        help="RNG seed for the deterministic train/val split (default: 42). Resumable.",
    )
    train_parser.add_argument(
        "--encoder-chunk-size", type=int, default=None,
        help=(
            "Phase B only: run the encoder forward in chunks of this many "
            "unique cards at a time, with gradient checkpointing per chunk "
            "so activation memory is bounded (default: 128). Lower this if "
            "you still hit CUDA OOM; raise it if you have headroom and want "
            "the extra throughput. Resumable."
        ),
    )
    train_parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help=(
            "Per-parameter-group L2 norm cap applied between backward and "
            "optimizer step (default: 100.0). Set high enough that clipping "
            "only fires on real spikes; too-low values silently throttle "
            "the effective learning rate. The per-epoch report prints the "
            "mean and max pre-clip norm per group so you can verify "
            "clipping is rare. Resumable."
        ),
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
        "--side-a-decks",
        default=None,
        help=(
            "Optional path to a generated-decks file (built by build-decks)."
            " When given, deck A in every match is sampled from this file"
            " and the 4 Forge methods are NOT used to produce side A. When"
            " omitted, side A is built by the 4 Forge methods (weights"
            " 4:3:2:1) on a freshly-generated pool from a random sealed-legal"
            " set, like Phase 0."
        ),
    )
    match_parser.add_argument(
        "--side-b-decks",
        default=None,
        help=(
            "Optional path to a generated-decks file. When given, deck B is"
            " produced by the 4 Forge methods (weights 4:3:2:1) PLUS sampling"
            " from this file (weight --side-b-decks-weight). When omitted,"
            " deck B is produced by the 4 Forge methods only. The Forge"
            " methods always run, regardless of this flag — same-set pairing"
            " on deck A's set code is preserved."
        ),
    )
    match_parser.add_argument(
        "--side-b-decks-weight",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Weight of the file-sampled side-B method relative to the 4 Forge"
            f" methods (which carry weights 4:3:2:1, total 10). Default:"
            f" {DEFAULT_SIDE_B_DECKS_WEIGHT}. Set higher to oversample"
            " scorer-built decks against Forge methods (e.g. 8 for a gen3 run"
            " that needs strong gen1+gen2 representation on side B). Only"
            " meaningful when --side-b-decks is given; supplying it without"
            " --side-b-decks is an error."
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


_ENCODE_CARDS_DEFAULT_ENCODER = "models/price-predictor/transformer/latest.pt"


def run_encode_cards(args: argparse.Namespace) -> int:
    """Execute the encode-cards command."""
    from price_predictor.domain.entities import TransformerConfig
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer
    from price_predictor.infrastructure.transformer_model import (
        CardPriceTransformerModel,
    )
    from price_predictor.infrastructure.transformer_store import load_model
    from sealed.application.encode_cards import EncodeCardsUseCase
    from sealed.domain.card_encoder import CardEncoder
    from sealed.infrastructure.embedding_store import EmbeddingStore
    from sealed.infrastructure.scorer_store import ScorerStore

    if args.encoder_checkpoint is not None and args.scorer_checkpoint is not None:
        print(
            "Error: --encoder-checkpoint and --scorer-checkpoint are mutually "
            "exclusive: choose one source for the encoder weights.",
            file=sys.stderr,
        )
        return 2

    if args.scorer_checkpoint is not None:
        scorer_path = Path(args.scorer_checkpoint)
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
        config = cfg
    else:
        encoder_path = Path(
            args.encoder_checkpoint or _ENCODE_CARDS_DEFAULT_ENCODER,
        )
        if not encoder_path.exists():
            print(f"Error: Encoder model not found: {encoder_path}", file=sys.stderr)
            return 2
        model, config = load_model(encoder_path)

    vocab_path = Path(args.vocab_path)
    cards_path = Path(args.cards_path)
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found: {vocab_path}", file=sys.stderr)
        return 2
    if not cards_path.exists():
        print(f"Error: Cards path not found: {cards_path}", file=sys.stderr)
        return 2

    print(f"Encoding cards in {cards_path}")

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
        label=args.label,
        checkpoint=Path(args.checkpoint),
        cards_path=Path(args.cards_path),
        output=Path(args.output),
        sa_temperature=args.sa_temperature,
        sa_cooling=args.sa_cooling,
        sa_max_iterations=args.sa_max_iterations,
        restarts=args.restarts,
        print_decks=args.print_decks,
        resume=args.resume,
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


_RESUMABLE_FLAG_NAMES: tuple[str, ...] = (
    "outcomes_path", "cards_path", "checkpoint_dir",
    "epochs", "batch_size", "lr",
    "n_layers", "n_heads", "n_seeds", "d_ff", "mlp_hidden", "dropout",
    "embedding_lr", "patience", "val_fraction", "random_seed",
    "encoder_chunk_size", "max_grad_norm",
)
_PATH_FIELDS: frozenset[str] = frozenset({
    "outcomes_path", "cards_path", "checkpoint_dir",
    "encoder_checkpoint", "scorer_checkpoint", "resume",
})
# Fallback CLI defaults for fields that lack a dataclass default (required fields).
# These mirror the help-text defaults registered in `_build_train_scorer_parser`.
_REQUIRED_FIELD_CLI_DEFAULTS: dict[str, str] = {
    "outcomes_path": "output/sealed/match-outcomes.txt",
    "cards_path": "output/cardsfolder/",
    "checkpoint_dir": "models/sealed/scorer/",
}


def _dataclass_default(field_name: str):
    field = TrainScorerConfig.__dataclass_fields__[field_name]
    import dataclasses
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    if field_name in _REQUIRED_FIELD_CLI_DEFAULTS:
        return _REQUIRED_FIELD_CLI_DEFAULTS[field_name]
    raise ValueError(f"No fallback default for required field {field_name!r}")


def _coerce_path_value(field_name: str, value):
    if value is None:
        return None
    return Path(value) if field_name in _PATH_FIELDS else value


def run_train_scorer(args: argparse.Namespace) -> int:
    """Execute the train-scorer command."""
    from sealed.application.train_scorer import TrainScorerUseCase
    from sealed.infrastructure.scorer_store import ScorerStore

    if args.resume is not None and args.scorer_checkpoint is not None:
        print(
            "Error: --resume and --scorer-checkpoint are mutually exclusive: "
            "--resume continues an existing run; --scorer-checkpoint bootstraps "
            "a fresh Phase B run from a Phase A checkpoint.",
            file=sys.stderr,
        )
        return 2

    if args.resume is not None or args.scorer_checkpoint is not None:
        for flag in _TRAIN_SCORER_ARCHITECTURE_FLAGS:
            if getattr(args, flag) is not None:
                print(
                    f"Error: architecture flag --{flag.replace('_', '-')} "
                    "conflicts with --scorer-checkpoint; architecture is "
                    "inherited from the checkpoint's stored config. "
                    "Omit the flag.",
                    file=sys.stderr,
                )
                return 2

    resumed_train_config: dict | None = None
    resumed_is_phase_b: bool | None = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        try:
            resumed = ScorerStore().load_checkpoint(resume_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        resumed_train_config = resumed.train_config
        resumed_is_phase_b = resumed.encoder_state_dict is not None

    if args.scorer_checkpoint is not None:
        scorer_path = Path(args.scorer_checkpoint)
        try:
            ScorerStore().load_checkpoint(scorer_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    resolved: dict[str, object | None] = {}
    for flag in _RESUMABLE_FLAG_NAMES:
        cli_value = getattr(args, flag, None)
        if cli_value is not None:
            resolved[flag] = cli_value
        elif resumed_train_config is not None and flag in resumed_train_config:
            resolved[flag] = resumed_train_config[flag]
        else:
            default = _dataclass_default(flag)
            resolved[flag] = default if not isinstance(default, Path) else str(default)

    embedding_lr = float(resolved["embedding_lr"])
    requested_phase_b = embedding_lr != 0
    if requested_phase_b and (
        args.scorer_checkpoint is None and args.resume is None
    ):
        print(
            f"Error: --embedding-lr {embedding_lr} requires either "
            "--scorer-checkpoint <phaseA>.pt (fresh Phase B kickoff) or "
            "--resume <phaseB>.pt (continuing Phase B). Phase B against a "
            "randomly-initialized scorer is not supported.",
            file=sys.stderr,
        )
        return 2

    if args.scorer_checkpoint is not None and not requested_phase_b:
        print(
            "Error: --scorer-checkpoint is only valid when starting a Phase B "
            "run; pass --embedding-lr <nonzero> alongside it. To continue a "
            "Phase A run, use --resume instead.",
            file=sys.stderr,
        )
        return 2

    if (
        args.encoder_checkpoint is not None
        and args.scorer_checkpoint is None
        and args.resume is None
        and not requested_phase_b
    ):
        print(
            "Error: --encoder-checkpoint has no effect on a Phase A run "
            "(encoder is not in the training graph). Drop the flag, or add "
            "--scorer-checkpoint <phaseA>.pt and --embedding-lr <nonzero> to "
            "kick off Phase B.",
            file=sys.stderr,
        )
        return 2

    if resumed_is_phase_b is not None and resumed_is_phase_b != requested_phase_b:
        ckpt_phase = "B" if resumed_is_phase_b else "A"
        run_phase = "B" if requested_phase_b else "A"
        eff_lr = embedding_lr if requested_phase_b else 0
        print(
            f"Error: --resume {args.resume} is a Phase {ckpt_phase} checkpoint "
            f"but --embedding-lr {eff_lr} requests Phase {run_phase}. Use "
            "--scorer-checkpoint to start a fresh Phase B run from a Phase A "
            "checkpoint.",
            file=sys.stderr,
        )
        return 2

    if (
        args.encoder_checkpoint is not None
        and args.resume is not None
        and resumed_is_phase_b
    ):
        print(
            "Error: --encoder-checkpoint conflicts with --resume on a Phase B "
            "checkpoint, which already carries fine-tuned encoder weights. "
            "Omit --encoder-checkpoint.",
            file=sys.stderr,
        )
        return 2

    if args.encoder_checkpoint is not None:
        encoder_checkpoint = Path(args.encoder_checkpoint)
    elif (
        resumed_train_config is not None
        and resumed_train_config.get("encoder_checkpoint") is not None
    ):
        encoder_checkpoint = Path(resumed_train_config["encoder_checkpoint"])
    else:
        encoder_checkpoint = _dataclass_default("encoder_checkpoint")

    config_kwargs = {
        flag: _coerce_path_value(flag, resolved[flag])
        for flag in _RESUMABLE_FLAG_NAMES
    }
    config = TrainScorerConfig(
        resume=Path(args.resume) if args.resume else None,
        scorer_checkpoint=(
            Path(args.scorer_checkpoint) if args.scorer_checkpoint else None
        ),
        encoder_checkpoint=encoder_checkpoint,
        **config_kwargs,
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

    # --side-b-decks-weight uses argparse.SUPPRESS so the attribute is absent
    # from `args` when the user doesn't pass the flag. This lets us reject
    # "weight without --side-b-decks" while still supplying the runtime
    # default (DEFAULT_SIDE_B_DECKS_WEIGHT) when only --side-b-decks is set.
    explicit_side_b_weight = getattr(args, "side_b_decks_weight", None)
    if args.side_b_decks is None and explicit_side_b_weight is not None:
        print(
            "Error: --side-b-decks-weight is only valid with --side-b-decks",
            file=sys.stderr,
        )
        return 2

    output_path = Path("output") / "sealed" / "match-outcomes.txt"
    side_a_decks = Path(args.side_a_decks) if args.side_a_decks else None
    side_b_decks = Path(args.side_b_decks) if args.side_b_decks else None
    side_b_decks_weight = (
        explicit_side_b_weight
        if explicit_side_b_weight is not None
        else DEFAULT_SIDE_B_DECKS_WEIGHT
    )

    supervisor = MatchOutcomeSupervisor(
        worker_count=args.workers,
        output_path=output_path,
        best_of=args.best_of,
        side_a_decks_path=side_a_decks,
        side_b_decks_path=side_b_decks,
        side_b_decks_weight=side_b_decks_weight,
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
