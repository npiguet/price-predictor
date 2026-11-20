"""The ``train-picker`` command."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from sealed.infrastructure.cli_common import _add_embeddings_path, _missing_paths_error
from sealed.infrastructure.cli_resume import resolve_resumable_args


def _build_train_picker_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-picker",
        help="Train a one-shot deck picker via REINFORCE against a frozen scorer",
    )
    parser.set_defaults(func=run_train_picker)
    # Resumable flags register with default=None so resume-precedence resolution
    # (explicit CLI > resumed train_config > dataclass default) works.
    _add_picker_path_flags(parser)
    _add_picker_start_flags(parser)
    _add_picker_architecture_flags(parser)
    _add_picker_objective_flags(parser)
    _add_picker_sampling_flags(parser)
    _add_picker_schedule_flags(parser)


def _add_picker_path_flags(parser: argparse.ArgumentParser) -> None:
    """The pools, the scorers, the embeddings and the run directory."""
    parser.add_argument(
        "--pools-path", default=None,
        help="Pre-generated pools file (SET_CODE;Card1|...). Required. Resumable.",
    )
    parser.add_argument(
        "--scorer-checkpoint", default=None,
        help=(
            "Frozen scorer used as the reward function. Must exist. Required "
            "unless --resume. Resumable."
        ),
    )
    parser.add_argument(
        "--auditor-scorer-checkpoint", default=None,
        help=(
            "Optional second scorer; when set, enables the per-epoch cross-scorer "
            "rank-correlation audit on the validation decks. Resumable."
        ),
    )
    parser.add_argument(
        "--cards-path", default=None,
        help="Directory of .npz card embeddings. Required unless --resume. Resumable.",
    )
    _add_embeddings_path(parser, resumable=True)
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help=(
            "Output dir for latest.pt + best_{timestamp}.pt (the run directory). "
            "Required unless --resume. Resumable."
        ),
    )


def _add_picker_start_flags(parser: argparse.ArgumentParser) -> None:
    """What a run starts from: a resumed run or another picker's weights."""
    parser.add_argument(
        "--resume", default=None,
        help=(
            "Continue a stopped run from this checkpoint (weights, optimizer, "
            "epoch, best_val_reward). Architecture flags forbidden. Mutually "
            "exclusive with --picker-checkpoint."
        ),
    )
    parser.add_argument(
        "--picker-checkpoint", default=None,
        help=(
            "Bootstrap a fresh run from this checkpoint's weights only. "
            "Architecture flags forbidden. Required when --kl-coef is non-zero. "
            "Mutually exclusive with --resume."
        ),
    )


#: The architecture flags: (flag, dest, type, help text before the shared
#: "Forbidden" sentence).
_PICKER_ARCHITECTURE_FLAG_SPECS: tuple[tuple[str, str, type, str], ...] = (
    ("--d-model", "d_model", int, (
        "Picker internal width (default: derived = embedding width). When set "
        "to a value other than the embedding width, a single input projection "
        "is inserted. "
    )),
    ("--n-layers", "n_layers", int, "Number of SAB layers (default: 4). "),
    ("--n-heads", "n_heads", int, "Attention heads per SAB (default: 8). "),
    ("--ff-dim", "d_ff", int, "Feed-forward dim (default: 4*d_model). "),
    ("--dropout", "dropout", float, "Dropout in SAB layers (default: 0.0). "),
)


def _add_picker_architecture_flags(parser: argparse.ArgumentParser) -> None:
    """The picker's shape, which a resumed or bootstrapped run inherits."""
    for flag, dest, kind, text in _PICKER_ARCHITECTURE_FLAG_SPECS:
        parser.add_argument(
            flag, type=kind, default=None, dest=dest,
            help=text + "Forbidden alongside --resume / --picker-checkpoint.",
        )

def _add_picker_objective_flags(parser: argparse.ArgumentParser) -> None:
    """The loss: the aux weight, advantage normalization and the objective."""
    parser.add_argument(
        "--aux-weight", type=float, default=None, dest="aux_weight",
        help="Coefficient on the aux pool-quality MSE loss (default: 0.1). Resumable.",
    )
    parser.add_argument(
        "--normalize-advantage", default=None,
        action=argparse.BooleanOptionalAction, dest="normalize_advantage",
        help=(
            "GRPO-style per-pool advantage normalization: divide the centered "
            "reward by the per-pool reward std before weighting the policy "
            "gradient. Sharpens the (otherwise tiny) signal when within-pool "
            "reward variance is small (default: off). When enabled you may want "
            "a larger --entropy-coef, since the rescaled policy term dominates "
            "the fixed-scale entropy/aux terms. Resumable."
        ),
    )
    parser.add_argument(
        "--objective", default=None, choices=["reinforce", "topk"],
        dest="objective",
        help=(
            "Training objective (default: reinforce). 'reinforce' is the "
            "advantage-weighted policy gradient with a per-pool baseline; "
            "'topk' is reward-ranked / best-of-N (keep the --topk highest-"
            "reward sampled decks per pool and maximize their log-prob, no "
            "baseline). Resumable and not architecture-locked, so a reinforce "
            "checkpoint can be continued or warm-started under topk."
        ),
    )
    parser.add_argument(
        "--topk", type=int, default=None, dest="topk",
        help=(
            "Decks kept per pool under --objective topk (default: 16). Must be "
            "between 1 and --n-samples; smaller = greedier. Resumable."
        ),
    )


def _add_picker_sampling_flags(parser: argparse.ArgumentParser) -> None:
    """How decks are sampled and how the optimizer steps."""
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="Pools per gradient step (default: 16). Resumable.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, dest="n_samples",
        help="Sampled decks per pool per step (default: 64). Resumable.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, dest="temperature",
        help="Softmax temperature for sampling (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=None, dest="entropy_coef",
        help="Initial entropy coefficient (default: 0.01). Resumable.",
    )
    parser.add_argument(
        "--entropy-decay-after", type=int, default=None, dest="entropy_decay_after",
        help="Consecutive improving-val epochs before entropy decays (default: 5). Resumable.",
    )
    parser.add_argument(
        "--lr", type=float, default=None, dest="lr",
        help="AdamW learning rate (default: 3e-4). Resumable.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=None, dest="max_grad_norm",
        help="Per-parameter-group L2 norm cap (default: 1.0). Resumable.",
    )


def _add_picker_schedule_flags(parser: argparse.ArgumentParser) -> None:
    """How long it trains, how it validates and the KL penalty."""
    parser.add_argument(
        "--epochs", type=int, default=None, dest="epochs",
        help="Maximum epochs (default: 100). Resumable.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None, dest="val_fraction",
        help="Front fraction of the pools file held out for validation (default: 0.2). Resumable.",
    )
    parser.add_argument(
        "--evals-per-epoch", type=int, default=None, dest="evals_per_epoch",
        help=(
            "How many times to validate per epoch (default: 1 = once at epoch "
            "end). Each eval runs the full validation set and updates latest.pt "
            "and best_*.pt; --patience is counted in these evals (mini-epochs). "
            "Resumable."
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=None, dest="patience",
        help=(
            "Early-stop after this many evals (mini-epochs) without val-reward "
            "improvement (default: 10). Equals epochs when --evals-per-epoch=1. "
            "Resumable."
        ),
    )
    parser.add_argument(
        "--kl-coef", type=float, default=None, dest="kl_coef",
        help=(
            "KL penalty coefficient against --picker-checkpoint's reference "
            "distribution (default: 0.0). Non-zero requires --picker-checkpoint. Resumable."
        ),
    )


_TRAIN_PICKER_ARCHITECTURE_FLAGS: tuple[str, ...] = (
    "d_model", "n_layers", "n_heads", "d_ff", "dropout",
)
_RESUMABLE_PICKER_FLAG_NAMES: tuple[str, ...] = (
    "pools_path", "scorer_checkpoint", "auditor_scorer_checkpoint",
    "cards_path", "embeddings_path", "checkpoint_dir",
    "d_model", "n_layers", "n_heads", "d_ff", "dropout",
    "aux_weight", "normalize_advantage", "objective", "topk",
    "batch_size", "n_samples", "temperature",
    "entropy_coef", "entropy_decay_after", "lr", "max_grad_norm",
    "epochs", "val_fraction", "patience", "evals_per_epoch", "kl_coef",
)
_PICKER_PATH_FIELDS: frozenset[str] = frozenset({
    "pools_path", "scorer_checkpoint", "auditor_scorer_checkpoint",
    "cards_path", "embeddings_path", "checkpoint_dir",
})


#: The path flags a fresh run must give: a resumed run inherits them from its checkpoint.
_PICKER_REQUIRED_PATHS: tuple[str, ...] = (
    "scorer_checkpoint", "cards_path", "checkpoint_dir",
)


def _resume_with_picker_checkpoint(args: argparse.Namespace) -> str | None:
    if args.resume is not None and args.picker_checkpoint is not None:
        return (
            "Error: --resume and --picker-checkpoint are mutually exclusive: "
            "--resume continues an existing run; --picker-checkpoint bootstraps "
            "a fresh run from another picker's weights."
        )
    return None


def _architecture_flag_given(args: argparse.Namespace) -> str | None:
    """The first architecture flag given, or None."""
    return next(
        (flag for flag in _TRAIN_PICKER_ARCHITECTURE_FLAGS if getattr(args, flag) is not None),
        None,
    )


def _architecture_on_inherited(args: argparse.Namespace) -> str | None:
    if args.resume is not None or args.picker_checkpoint is not None:
        flag = _architecture_flag_given(args)
        if flag is not None:
            return (
                f"Error: architecture flag --{flag.replace('_', '-')} "
                "conflicts with --resume / --picker-checkpoint; architecture "
                "is inherited from the checkpoint's stored config. Omit the flag."
            )
    return None


def _kl_without_reference(args: argparse.Namespace) -> str | None:
    kl_coef = args.kl_coef if args.kl_coef is not None else 0.0
    if kl_coef != 0 and args.picker_checkpoint is None:
        return (
            "Error: --kl-coef is non-zero but --picker-checkpoint is not set. "
            "The KL penalty needs a reference distribution; pass "
            "--picker-checkpoint <path>."
        )
    return None


#: The checks on the flags as given, in the order they refuse a run: each
#: returns its refusal, or None.
_FLAG_CHECKS = (
    _resume_with_picker_checkpoint,
    _architecture_on_inherited,
    _kl_without_reference,
)


def _no_pools(resolved: dict) -> str | None:
    if resolved["pools_path"] is None:
        return "Error: --pools-path is required."
    return None


def _topk_without_selection(resolved: dict) -> str | None:
    # Top-k objective needs real selection pressure: 1 <= topk < n_samples
    # (FR-039). topk == n_samples keeps every sampled deck = no selection.
    if resolved["objective"] == "topk":
        topk = int(resolved["topk"])  # type: ignore[arg-type]
        n_samples = int(resolved["n_samples"])  # type: ignore[arg-type]
        if not 1 <= topk < n_samples:
            return (
                f"Error: --topk must be between 1 and --n-samples-1 "
                f"({n_samples - 1}) for the topk objective, got {topk} with "
                f"--n-samples {n_samples}. topk == n_samples keeps every "
                "sampled deck, which removes all selection pressure."
            )
    return None


def _missing_scorer(resolved: dict) -> str | None:
    # Scorer must exist (FR-036).
    scorer_path = resolved["scorer_checkpoint"]
    if not scorer_path.exists():
        return (
            f"Error: scorer checkpoint not found at {scorer_path}. Train a "
            "scorer first (python -m sealed train-scorer), or pass an existing "
            "--scorer-checkpoint <path>."
        )
    return None


#: The checks on the resolved values, in the order they refuse a run.
_RESOLVED_CHECKS = (_no_pools, _topk_without_selection, _missing_scorer)


def _first_refusal(checks, value) -> int | None:
    """Exit 2 printing the first of ``checks``' refusals of ``value``, or None."""
    for check in checks:
        refusal = check(value)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 2
    return None


@dataclass(frozen=True)
class _ResumedPicker:
    """``--resume``'s recorded train config (None without it)."""

    train_config: dict | None = None


def _load_resumed_picker(args: argparse.Namespace) -> _ResumedPicker | int:
    """Load ``--resume``'s checkpoint; exit 2 when it is missing."""
    from sealed.infrastructure.picker_store import PickerStore

    if args.resume is None:
        return _ResumedPicker()
    try:
        resumed = PickerStore().load_checkpoint(Path(args.resume))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return _ResumedPicker(resumed.train_config)


def _resolve_picker_args(args: argparse.Namespace, resumed: _ResumedPicker) -> dict:
    from sealed.application.train_picker import TrainPickerConfig

    return resolve_resumable_args(
        args,
        flag_names=_RESUMABLE_PICKER_FLAG_NAMES,
        config_cls=TrainPickerConfig,
        path_fields=_PICKER_PATH_FIELDS,
        resumed_config=resumed.train_config,
    )


def _execute_train_picker(args: argparse.Namespace, resolved: dict) -> int:
    from sealed.application.train_picker import TrainPickerConfig, TrainPickerUseCase
    from sealed.domain.picker_model import PickerArchitectureError

    config = TrainPickerConfig(
        resume=Path(args.resume) if args.resume else None,
        picker_checkpoint=(
            Path(args.picker_checkpoint) if args.picker_checkpoint else None
        ),
        **resolved,
    )

    try:
        TrainPickerUseCase().execute(config)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except PickerArchitectureError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


def run_train_picker(args: argparse.Namespace) -> int:
    """Execute the train-picker command."""
    flags_err = _first_refusal(_FLAG_CHECKS, args)
    if flags_err is not None:
        return flags_err

    resumed = _load_resumed_picker(args)
    if isinstance(resumed, int):
        return resumed

    missing_err = _missing_paths_error(args, resumed.train_config, _PICKER_REQUIRED_PATHS)
    if missing_err is not None:
        return missing_err

    resolved = _resolve_picker_args(args, resumed)
    resolved_err = _first_refusal(_RESOLVED_CHECKS, resolved)
    if resolved_err is not None:
        return resolved_err

    return _execute_train_picker(args, resolved)
