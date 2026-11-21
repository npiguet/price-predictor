"""The ``generate-draft-data`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft.infrastructure.cli_common import (
    _add_embeddings_path,
    _optional_path,
    _picker_checkpoint_missing,
)


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
        "--scorer-checkpoint", required=True,
        help="Frozen scorer labeling decks.",
    )
    parser.add_argument(
        "--build-method", default="picker", choices=["picker", "greedy"],
        help="Deck builder for labeling (default: picker).",
    )
    parser.add_argument(
        "--picker-checkpoint", default=None,
        help="Picker; required with --build-method picker, ignored otherwise.",
    )
    parser.add_argument(
        "--cards-path", required=True,
        help=".npz embedding cache.",
    )
    _add_embeddings_path(parser)
    parser.add_argument(
        "--output-path", required=True,
        help="Destination JSONL; appended, created if missing.",
    )
    parser.add_argument(
        "--logs-dir", required=True,
        help=(
            "Directory of the live-seat worker's stderr log, worker-<run id>.log "
            "(under the layout, logs/ in the run directory)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append + count pre-existing drafts toward --n-drafts (FR-012).",
    )
    # Live model-pilot flags (opt-in; absent ⇒ byte-for-byte gen-1, SC-004).
    parser.add_argument(
        "--agent-checkpoint", action="append", default=[], dest="agent_checkpoints",
        metavar="LABEL=PATH",
        help=(
            "Bind a mix label to a trained agent checkpoint (repeatable). A bare "
            "PATH is shorthand for label 'draft-agent'. A mix label that is "
            "neither a Forge built-in nor bound here fails fast (FR-011)."
        ),
    )
    parser.add_argument(
        "--pick-mode", default="argmax", choices=["argmax", "sample"],
        help=(
            "argmax = the policy's strongest legal card (default); sample = "
            "temperature-scaled softmax over PACK logits (rollout diversity)."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Softmax temperature for --pick-mode sample; ignored for argmax (default 1.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed the pick-sampling RNG for reproducible sampled rollouts (SC-007).",
    )
    parser.add_argument(
        "--max-consecutive-faults", type=int, default=5,
        help=(
            "Abort the run with a nonzero exit after this many consecutive "
            "pick-fault-abandoned drafts (FR-016; default 5)."
        ),
    )


def _generate_flag_error(args: argparse.Namespace) -> bool:
    """Print the error and return True when generate-draft-data's flags disagree."""
    if args.pick_mode == "sample" and args.temperature <= 0:
        print("Error: --temperature must be > 0 with --pick-mode sample.", file=sys.stderr)
        return True
    return _picker_checkpoint_missing(args)


def run_generate_draft_data(args: argparse.Namespace) -> int:
    """Execute the generate-draft-data command."""
    from draft.application.agent_mix import AgentMixError, parse_agent_mix
    from draft.application.agent_registry import (
        AgentRegistryError,
        parse_agent_checkpoints,
    )
    from draft.application.generate_draft_data import (
        GenerateDraftDataConfig,
        GenerateDraftDataSupervisor,
        MaxConsecutiveFaultsError,
    )

    if args.n_drafts < 1:
        print("Error: --n-drafts must be >= 1.", file=sys.stderr)
        return 2

    try:
        agent_mix = parse_agent_mix(args.agent_mix)
    except AgentMixError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        agent_checkpoints = parse_agent_checkpoints(args.agent_checkpoints)
    except AgentRegistryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if _generate_flag_error(args):
        return 2

    config = GenerateDraftDataConfig(
        n_drafts=args.n_drafts,
        set_code=args.set_code,
        agent_mix=agent_mix,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        build_method=args.build_method,
        picker_checkpoint=_optional_path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_optional_path(args.embeddings_path),
        output_path=Path(args.output_path),
        resume=args.resume,
        agent_checkpoints=agent_checkpoints,
        pick_mode=args.pick_mode,
        temperature=args.temperature,
        seed=args.seed,
        max_consecutive_faults=args.max_consecutive_faults,
        worker_log_dir=Path(args.logs_dir),
    )

    try:
        GenerateDraftDataSupervisor(config).run()
    except AgentRegistryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except MaxConsecutiveFaultsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
