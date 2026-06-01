"""CLI interface for the draft module (generate-draft-data, train-draft-agent).

Mirrors ``sealed/infrastructure/cli.py``: ``build_parser`` adds one subparser
per command with ``set_defaults(func=…)`` dispatch, and ``main`` parses and
calls the resolved ``func``. Application imports are lazy (inside each ``run_*``)
so ``python -m draft --help`` and unit tests don't pay the torch import cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Architecture flags forbidden alongside --resume / --checkpoint (SC-006); they
# register with default=None so "was it supplied?" is detectable.
_TRAIN_AGENT_ARCHITECTURE_FLAGS: tuple[str, ...] = (
    "d_model", "n_layers", "n_heads", "ff_dim", "dropout",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draft",
        description="Draft agent data generation and training tools",
    )
    subparsers = parser.add_subparsers(help="Available commands")
    _build_generate_draft_data_parser(subparsers)
    _build_train_draft_agent_parser(subparsers)
    return parser


def _build_generate_draft_data_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate-draft-data",
        help=(
            "Drive Forge's draft AI for all pod seats, build+score a deck per "
            "seat, and append one self-contained JSON record per draft"
        ),
    )
    parser.set_defaults(func=run_generate_draft_data)
    parser.add_argument(
        "--n-drafts", type=int, required=True,
        help="Number of complete drafts to generate (counts pre-existing under --resume).",
    )
    parser.add_argument(
        "--set", default=None, dest="set_code",
        help=(
            "Restrict all drafts to one set code; else each draft independently "
            "picks a random sealed-legal set (FR-009)."
        ),
    )
    parser.add_argument(
        "--agent-mix", default="forge-full:6,forge-r30:1,forge-r100:1",
        help=(
            "Categorical agent weights, 'label:weight,...'; each of pod_size "
            "seats is sampled independently (FR-006). "
            "Default: forge-full:6,forge-r30:1,forge-r100:1"
        ),
    )
    parser.add_argument(
        "--scorer-checkpoint", default="models/sealed/scorer/latest.pt",
        help="Frozen scorer labeling decks (default: models/sealed/scorer/latest.pt).",
    )
    parser.add_argument(
        "--build-method", default="picker", choices=["picker", "greedy"],
        help="Deck builder for labeling (default: picker).",
    )
    parser.add_argument(
        "--picker-checkpoint", default="models/sealed/picker/latest.pt",
        help=(
            "Picker used when --build-method picker; ignored otherwise "
            "(default: models/sealed/picker/latest.pt)."
        ),
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz embedding cache (default: output/cardsfolder/).",
    )
    parser.add_argument(
        "--output-path", default="output/draft/drafts.jsonl",
        help="Destination JSONL; appended, created if missing.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append + count pre-existing drafts toward --n-drafts (FR-012).",
    )


def _build_train_draft_agent_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent",
        help="Train the imitation policy + MC-regression critic jointly on a recorded corpus",
    )
    parser.set_defaults(func=run_train_draft_agent)
    parser.add_argument(
        "--drafts-path", default="output/draft/drafts.jsonl",
        help="Recorded corpus (default: output/draft/drafts.jsonl).",
    )
    parser.add_argument(
        "--cards-path", default="output/cardsfolder/",
        help=".npz cache (default: output/cardsfolder/).",
    )
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
        "--imitation-weight", type=float, default=1.0,
        help="CE coefficient; 0 => critic-only (default: 1.0).",
    )
    parser.add_argument(
        "--critic-weight", type=float, default=1.0,
        help="MSE coefficient; 0 => imitation-only (default: 1.0).",
    )
    parser.add_argument(
        "--imitation-agents", default="forge-full",
        help="Comma-separated whitelist of agents whose picks are imitation targets.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="AdamW LR (default: 3e-4).",
    )
    parser.add_argument(
        "--warmup-frac", type=float, default=0.05,
        help="LR linear-warmup fraction (default: 0.05).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="States per gradient step (default: 32).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Per-group gradient-norm cap (default: 1.0).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max epochs (default: 100).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Draft-disjoint validation fraction (default: 0.2).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stop epochs without val improvement (default: 10).",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Restore weights+optimizer+epoch+best-val; arch flags forbidden; xor --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Bootstrap fresh run from weights only; architecture flags forbidden; xor --resume.",
    )


def run_generate_draft_data(args: argparse.Namespace) -> int:
    """Execute the generate-draft-data command."""
    from draft.application.agent_mix import AgentMixError, parse_agent_mix
    from draft.application.generate_draft_data import (
        GenerateDraftDataConfig,
        GenerateDraftDataSupervisor,
    )

    if args.n_drafts < 1:
        print("Error: --n-drafts must be >= 1.", file=sys.stderr)
        return 2

    try:
        agent_mix = parse_agent_mix(args.agent_mix)
    except AgentMixError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    config = GenerateDraftDataConfig(
        n_drafts=args.n_drafts,
        set_code=args.set_code,
        agent_mix=agent_mix,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        build_method=args.build_method,
        picker_checkpoint=Path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        output_path=Path(args.output_path),
        resume=args.resume,
    )

    try:
        GenerateDraftDataSupervisor(config).run()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


def run_train_draft_agent(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent command."""
    from draft.application.train_draft_agent import (
        TrainDraftAgentConfig,
        TrainDraftAgentUseCase,
    )
    from draft.domain.draft_agent_model import DraftAgentArchitectureError

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

    config = TrainDraftAgentConfig(
        drafts_path=Path(args.drafts_path),
        cards_path=Path(args.cards_path),
        d_model=args.d_model,
        n_layers=args.n_layers if args.n_layers is not None else 4,
        n_heads=args.n_heads if args.n_heads is not None else 8,
        ff_dim=args.ff_dim,
        dropout=args.dropout if args.dropout is not None else 0.0,
        imitation_weight=args.imitation_weight,
        critic_weight=args.critic_weight,
        imitation_agents=tuple(
            a.strip() for a in args.imitation_agents.split(",") if a.strip()
        ),
        lr=args.lr,
        warmup_frac=args.warmup_frac,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        patience=args.patience,
        resume=Path(args.resume) if args.resume else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
