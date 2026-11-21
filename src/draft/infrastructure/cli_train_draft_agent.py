"""The ``train-draft-agent`` command (gen-1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft.infrastructure.cli_common import (
    _add_checkpoint_dir,
    _add_embeddings_path,
    _require_trainer_paths,
)

# Architecture flags forbidden alongside --resume / --checkpoint (SC-006); they
# register with default=None so "was it supplied?" is detectable.
_TRAIN_AGENT_ARCHITECTURE_FLAGS: tuple[str, ...] = (
    "d_model", "n_layers", "n_heads", "ff_dim", "dropout",
)


# Non-architecture training flags resolved with resume precedence (explicit CLI
# > resumed train_config > dataclass default). imitation_agents is resolved
# separately (CLI string -> tuple); architecture flags are inherited from the
# checkpoint's stored config, not train_config.
_RESUMABLE_TRAIN_FLAGS: tuple[str, ...] = (
    "drafts_path", "cards_path", "embeddings_path", "checkpoint_dir",
    "imitation_weight", "critic_weight",
    "lr", "warmup_frac", "batch_size", "max_grad_norm", "epochs",
    "val_fraction", "evals_per_epoch", "patience",
    "lr_decay_patience", "lr_decay_factor", "min_lr",
)


_TRAIN_PATH_FIELDS = frozenset({
    "drafts_path", "cards_path", "embeddings_path", "checkpoint_dir",
})


def _resolve_train_agent_config(
    args: argparse.Namespace, resumed_train_config: dict | None,
):
    """Build TrainDraftAgentConfig with resume precedence (CLI > resumed > default)."""
    from draft.application.train_draft_agent import TrainDraftAgentConfig
    from sealed.infrastructure.cli_resume import resolve_resumable_args

    resolved = resolve_resumable_args(
        args,
        flag_names=_RESUMABLE_TRAIN_FLAGS,
        config_cls=TrainDraftAgentConfig,
        path_fields=_TRAIN_PATH_FIELDS,
        resumed_config=resumed_train_config,
    )

    # imitation_agents: CLI is a comma string; config wants a tuple.
    if args.imitation_agents is not None:
        agents = tuple(
            a.strip() for a in args.imitation_agents.split(",") if a.strip()
        )
    elif resumed_train_config and resumed_train_config.get("imitation_agents"):
        agents = tuple(resumed_train_config["imitation_agents"])
    else:
        agents = ("forge-full",)

    return TrainDraftAgentConfig(
        # Architecture is inherited from the checkpoint on resume/checkpoint;
        # these only matter for a fresh run and are forbidden when loading.
        d_model=args.d_model,
        n_layers=args.n_layers if args.n_layers is not None else 4,
        n_heads=args.n_heads if args.n_heads is not None else 8,
        ff_dim=args.ff_dim,
        dropout=args.dropout if args.dropout is not None else 0.0,
        imitation_agents=agents,
        resume=Path(args.resume) if args.resume else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        **resolved,
    )


def _build_train_draft_agent_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent",
        help="Train the imitation policy + MC-regression critic jointly on a recorded corpus",
    )
    parser.set_defaults(func=run_train_draft_agent)
    # Resumable flags register with default=None so a value resolves as:
    # explicit CLI > resumed checkpoint's train_config > dataclass default.
    parser.add_argument(
        "--drafts-path", default=None,
        help="Recorded corpus. Required on a fresh run; Resumable.",
    )
    parser.add_argument(
        "--cards-path", default=None,
        help=".npz cache. Required on a fresh run; Resumable.",
    )
    _add_embeddings_path(parser, resumable=True)
    _add_checkpoint_dir(parser, help_text=(
        "Directory of latest.pt and <timestamp>.pt, and the run's run.json. "
        "Required on a fresh run; Resumable."
    ))
    # Architecture flags: default=None so they can be rejected with --resume/--checkpoint.
    parser.add_argument(
        "--d-model", type=int, default=None, dest="d_model",
        help="Internal width (default: derived); a non-default value inserts a Linear.",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None, dest="n_layers",
        help="SAB layers (default: 4).",
    )
    parser.add_argument(
        "--n-heads", type=int, default=None, dest="n_heads",
        help="Attention heads (default: 8); d_model %% n_heads == 0 (fail fast).",
    )
    parser.add_argument(
        "--ff-dim", type=int, default=None, dest="ff_dim",
        help="Feed-forward width (default: 4 * d_model).",
    )
    parser.add_argument(
        "--dropout", type=float, default=None, dest="dropout",
        help="Transformer dropout (default: 0.0).",
    )
    parser.add_argument(
        "--imitation-weight", type=float, default=None,
        help="CE coefficient; 0 => critic-only (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--critic-weight", type=float, default=None,
        help="MSE coefficient; 0 => imitation-only (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--imitation-agents", default=None,
        help="Comma-separated whitelist of imitation-target agents (default: "
             "forge-full). Resumable.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="AdamW LR (default: 3e-4). Resumable (also re-applied to a resumed "
             "optimizer, so --lr changes the LR on resume).",
    )
    parser.add_argument(
        "--warmup-frac", type=float, default=None,
        help="LR linear-warmup fraction (default: 0.05). Resumable.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="States per gradient step (default: 32). Resumable.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="Per-group gradient-norm cap (default: 1.0). Resumable.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs (default: 100). Resumable.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="Draft-disjoint validation fraction (default: 0.0025 — a small "
             "held-out monitor; the huge train set makes overfit unlikely). Resumable.",
    )
    parser.add_argument(
        "--evals-per-epoch", type=int, default=None,
        help="Validate + checkpoint this many times per epoch (mini-epochs; "
             "default: 100). Resumable.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop after this many mini-epochs without val improvement "
             "(default: 30). Resumable.",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=None, dest="lr_decay_patience",
        help="Enable plateau LR annealing: after this many mini-epochs without a "
             "new best val_loss, LR *= --lr-decay-factor (down to --min-lr). Must "
             "be < --patience. Default: disabled (constant LR). Resumable.",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=None, dest="lr_decay_factor",
        help="LR multiplier applied on each plateau (default: 0.1). Resumable.",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, dest="min_lr",
        help="Annealing floor; no decay below this (default: lr * 1e-3). Resumable.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Continue a run: restore weights+optimizer+epoch+best-val and "
             "inherit its training settings; CLI flags override them; arch flags "
             "forbidden; xor --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Bootstrap fresh run from weights only; architecture flags forbidden; xor --resume.",
    )


def run_train_draft_agent(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent command."""
    from draft.application.train_draft_agent import TrainDraftAgentUseCase
    from draft.domain.draft_agent_model import DraftAgentArchitectureError
    from draft.infrastructure.draft_agent_store import DraftAgentStore

    if args.resume is not None and args.checkpoint is not None:
        print(
            "Error: --resume and --checkpoint are mutually exclusive: --resume "
            "continues an existing run; --checkpoint bootstraps a fresh run from "
            "another agent's weights.",
            file=sys.stderr,
        )
        return 2

    loading = args.resume is not None or args.checkpoint is not None
    if loading:
        for flag in _TRAIN_AGENT_ARCHITECTURE_FLAGS:
            if getattr(args, flag) is not None:
                print(
                    f"Error: architecture flag --{flag.replace('_', '-')} "
                    "conflicts with --resume / --checkpoint; architecture is "
                    "inherited from the checkpoint's stored config. Omit the flag.",
                    file=sys.stderr,
                )
                return 2

    # On --resume, inherit the run's settings from its checkpoint; CLI flags
    # override. (--checkpoint is a fresh run from weights only, so no inherit.)
    resumed_train_config: dict | None = None
    if args.resume is not None:
        try:
            resumed = DraftAgentStore().load_checkpoint(Path(args.resume))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        resumed_train_config = resumed.train_config

    _require_trainer_paths(args, resumed_train_config)
    config = _resolve_train_agent_config(args, resumed_train_config)

    try:
        TrainDraftAgentUseCase().execute(config)
    except DraftAgentArchitectureError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
