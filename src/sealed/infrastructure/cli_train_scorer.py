"""The ``train-scorer`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sealed.application.train_scorer import TrainScorerConfig
from sealed.infrastructure.cli_common import _add_embeddings_path, _missing_paths_error
from sealed.infrastructure.cli_resume import resolve_resumable_args

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
            "Path to match outcomes file. Required unless --resume. "
            "Resumable: on --resume, falls back to the resumed checkpoint's "
            "value when omitted."
        ),
    )
    train_parser.add_argument(
        "--cards-path",
        default=None,
        help=(
            "Directory with .npz card embeddings. Required unless --resume. "
            "Resumable."
        ),
    )
    _add_embeddings_path(train_parser, resumable=True)
    train_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory for saving checkpoints (the run directory). "
            "Required unless --resume. Resumable."
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
            "--scorer-checkpoint (required then). "
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
    train_parser.add_argument(
        "--margin-weighting",
        choices=["linear", "log"],
        default=None,
        help=(
            "Weight each pairwise (winner, loser) example's contribution to "
            "the loss by the match's |wins_A - wins_B| margin (Bo7: 1-4). "
            "'linear' uses the margin directly; 'log' uses log(1 + margin) "
            "for a dampened version if linear training proves unstable. "
            "Absent = unweighted (every match contributes equally, "
            "regardless of how decisive it was; current default). Best/latest "
            "checkpoint names gain an _mwlin / _mwlog suffix when set so "
            "weighted runs do not clobber unweighted ones at the same "
            "architecture. Resumable: on --resume, falls back to the resumed "
            "checkpoint's value when omitted."
        ),
    )


_RESUMABLE_FLAG_NAMES: tuple[str, ...] = (
    "outcomes_path", "cards_path", "embeddings_path", "checkpoint_dir",
    "epochs", "batch_size", "lr",
    "n_layers", "n_heads", "n_seeds", "d_ff", "mlp_hidden", "dropout",
    "embedding_lr", "patience", "val_fraction", "random_seed",
    "encoder_chunk_size", "max_grad_norm", "margin_weighting",
)
_PATH_FIELDS: frozenset[str] = frozenset({
    "outcomes_path", "cards_path", "embeddings_path", "checkpoint_dir",
    "encoder_checkpoint", "scorer_checkpoint", "resume",
})
#: The path flags a fresh run must give: a resumed run inherits them from its checkpoint.
_SCORER_REQUIRED_PATHS: tuple[str, ...] = ("outcomes_path", "cards_path", "checkpoint_dir")


def _validate_scorer_exclusive_flags(args: argparse.Namespace) -> int | None:
    """Reject incompatible flag combinations that need only ``args``.

    Returns an exit code to propagate, or ``None`` to continue. Covers the
    --resume xor --scorer-checkpoint guard and the architecture-flags-forbidden
    -on-resume guard.
    """
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
    return None


def _resolve_scorer_encoder_checkpoint(
    args: argparse.Namespace, resumed_train_config: dict | None,
) -> Path | None:
    """Resolve --encoder-checkpoint with resume precedence (explicit > resumed
    > dataclass default). No effect on Phase A; consumed only on Phase B."""
    if args.encoder_checkpoint is not None:
        return Path(args.encoder_checkpoint)
    if (
        resumed_train_config is not None
        and resumed_train_config.get("encoder_checkpoint") is not None
    ):
        return Path(resumed_train_config["encoder_checkpoint"])
    return None


def _validate_scorer_phase(
    args: argparse.Namespace,
    embedding_lr: float,
    requested_phase_b: bool,
    resumed_is_phase_b: bool | None,
    encoder_checkpoint: Path | None,
) -> int | None:
    """Reject Phase A/B flag combinations that are only knowable post-resolution.

    Returns an exit code to propagate, or ``None`` to continue.
    """
    if requested_phase_b and args.scorer_checkpoint is None and args.resume is None:
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

    # A fresh Phase B kickoff loads the encoder from `--encoder-checkpoint`;
    # Phase A skips the encoder load path entirely, and Phase B resumes pull
    # encoder weights from the resumed checkpoint.
    is_phase_b_fresh_kickoff = requested_phase_b and args.resume is None
    if is_phase_b_fresh_kickoff and encoder_checkpoint is None:
        print(
            "Error: --encoder-checkpoint is required to start a fresh Phase B run: "
            "it names the encoder weights to fine-tune.",
            file=sys.stderr,
        )
        return 2
    return None


def run_train_scorer(args: argparse.Namespace) -> int:
    """Execute the train-scorer command."""
    from sealed.application.train_scorer import TrainScorerUseCase
    from sealed.infrastructure.scorer_store import ScorerStore

    exclusive_err = _validate_scorer_exclusive_flags(args)
    if exclusive_err is not None:
        return exclusive_err

    resumed_train_config: dict | None = None
    resumed_is_phase_b: bool | None = None
    if args.resume is not None:
        try:
            resumed = ScorerStore().load_checkpoint(Path(args.resume))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        resumed_train_config = resumed.train_config
        resumed_is_phase_b = resumed.encoder_state_dict is not None

    if args.scorer_checkpoint is not None:
        try:
            ScorerStore().load_checkpoint(Path(args.scorer_checkpoint))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    missing_err = _missing_paths_error(args, resumed_train_config, _SCORER_REQUIRED_PATHS)
    if missing_err is not None:
        return missing_err

    resolved = resolve_resumable_args(
        args,
        flag_names=_RESUMABLE_FLAG_NAMES,
        config_cls=TrainScorerConfig,
        path_fields=_PATH_FIELDS,
        resumed_config=resumed_train_config,
    )

    embedding_lr = float(resolved["embedding_lr"])
    requested_phase_b = embedding_lr != 0
    encoder_checkpoint = _resolve_scorer_encoder_checkpoint(args, resumed_train_config)

    phase_err = _validate_scorer_phase(
        args, embedding_lr, requested_phase_b, resumed_is_phase_b, encoder_checkpoint,
    )
    if phase_err is not None:
        return phase_err

    config = TrainScorerConfig(
        resume=Path(args.resume) if args.resume else None,
        scorer_checkpoint=(
            Path(args.scorer_checkpoint) if args.scorer_checkpoint else None
        ),
        encoder_checkpoint=encoder_checkpoint,
        **resolved,
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
