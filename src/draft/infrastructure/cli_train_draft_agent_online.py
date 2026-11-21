"""The ``train-draft-agent-online`` command (gen-3)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from draft.infrastructure.cli_common import (
    _add_embeddings_path,
    _optional_path,
    _picker_checkpoint_missing,
)

_ONLINE_DEFAULT_MIX = "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1"


def _build_train_draft_agent_online_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-draft-agent-online",
        help=(
            "Online self-play GRPO fine-tuning (gen-3): generate fresh drafts "
            "from the current policy, take one critic-free "
            "-A*logpi_T(a|s) pass over them, discard, regenerate"
        ),
    )
    parser.set_defaults(func=run_train_draft_agent_online)
    _add_online_agent_flags(parser)
    _add_online_reward_flags(parser)
    _add_online_rollout_flags(parser)
    _add_online_schedule_flags(parser)
    _add_online_corpus_flags(parser)
    _add_online_step_flags(parser)


def _add_online_agent_flags(parser: argparse.ArgumentParser) -> None:
    """Agent wiring (FR-003)."""
    parser.add_argument(
        "--learner", action="append", default=None, metavar="LABEL=PATH",
        help="The agent under training (required, exactly one). LABEL is its mix "
             "label; PATH warm-starts the policy at round 0. A bare PATH is not "
             "accepted — the label is load-bearing.",
    )
    parser.add_argument(
        "--frozen", action="append", default=None, metavar="LABEL=PATH",
        help="An untrained reference bound to a checkpoint (repeatable), e.g. the "
             "gen-1 anchor. Never enters the gradient.",
    )
    parser.add_argument(
        "--anchor", default=None, metavar="LABEL",
        help="Which --frozen label is the anchor-margin baseline. Required only "
             "when more than one --frozen is given.",
    )
    parser.add_argument(
        "--mix", default=_ONLINE_DEFAULT_MIX, metavar="LABEL:WEIGHT,...",
        help=f"Per-seat categorical over learner + frozen + Forge built-ins "
             f"(default: {_ONLINE_DEFAULT_MIX}).",
    )


def _add_online_reward_flags(parser: argparse.ArgumentParser) -> None:
    """Deck building & reward (same meaning as generate-draft-data), and the run directory."""
    parser.add_argument(
        "--scorer-checkpoint", required=True,
        help="Frozen scorer producing each seat's deck_score (the reward). Must exist.",
    )
    parser.add_argument(
        "--build-method", choices=("greedy", "picker"), default="greedy",
        help="Pool -> deck before scoring (default: greedy).",
    )
    parser.add_argument(
        "--picker-checkpoint", default=None,
        help="Required with --build-method picker, and must exist then; unused otherwise.",
    )
    parser.add_argument(
        "--cards-path", required=True,
        help=".npz embedding cache; fixes the model/scorer input width.",
    )
    _add_embeddings_path(parser)
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="The run directory: latest.pt, the snapshots and best_*.pt, run.json, and "
             "the worker's stderr log under logs/.",
    )


def _add_online_rollout_flags(parser: argparse.ArgumentParser) -> None:
    """Rollout & optimisation: sampling, the learning rate, rounds, the margin window."""
    parser.add_argument(
        "--agent-temp", default=None, dest="agent_temp", metavar="LABEL=T,...",
        help="Per-label sampling temperature; an omitted label plays argmax. "
             "Must name the --learner with a value > 0 -- that value is also "
             "what EVERY policy distribution (logpi, entropy, KL) is evaluated "
             "at. Naming a Forge built-in or an unbound label is an error. "
             "Required, no default.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--drafts-per-round", type=int, default=10, dest="drafts_per_round",
        help="Fresh drafts generated and trained on (one pass) per round (default: 10).",
    )
    parser.add_argument(
        "--anchor-window", type=int, default=100, dest="anchor_window",
        help="Sliding window (drafts) backing the anchor margin (default: 100).",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=25, dest="snapshot_every",
        help="Rounds between timestamped snapshots; latest.pt every round and "
             "best_*.pt on each new best margin (default: 25).",
    )


def _add_online_schedule_flags(parser: argparse.ArgumentParser) -> None:
    """Rollout & optimisation: early stop, LR annealing and the round budget."""
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Stop after N consecutive rounds with no new best anchor margin. "
             "Off by default. Must exceed --anchor-window / --drafts-per-round.",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=None, dest="lr_decay_patience",
        help="Anneal the LR after N rounds without a new best margin, resetting "
             "the stall counter so a decay pre-empts --patience. Off by default. "
             "Must sit between the window in rounds and --patience.",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.1, dest="lr_decay_factor",
        help="LR multiplier per plateau decay (default: 0.1).",
    )
    parser.add_argument(
        "--min-lr", type=float, default=None, dest="min_lr",
        help="Annealing floor (default: lr * 1e-3); an armed --patience only "
             "fires here.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None, dest="max_rounds",
        help="Optional round budget; omitted => runs until Ctrl-C.",
    )


def _add_online_corpus_flags(parser: argparse.ArgumentParser) -> None:
    """Rollout & optimisation: the drafts' sets, the corpus they append to, the seed."""
    parser.add_argument(
        "--set", dest="set_code", default=None,
        help="Restrict every draft to one set; else a random sealed-legal set per draft.",
    )
    parser.add_argument(
        "--output-path", required=True, dest="output_path",
        help="Corpus appended with every generated draft (shared file).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Torch/numpy init, batch shuffling, pick-sampling RNG. Forge-side "
             "randomness is NOT seeded (default: 42).",
    )


def _add_online_step_flags(parser: argparse.ArgumentParser) -> None:
    """Rollout & optimisation: the gradient step and the fault budget."""
    parser.add_argument(
        "--batch-size", type=int, default=32, dest="batch_size",
        help="States per gradient step (default: 32).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, dest="max_grad_norm",
        help="Per-parameter-group gradient-norm cap (default: 1.0).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200, dest="warmup_steps",
        help="Linear LR ramp over the first N optimizer steps of the run, then "
             "constant (default: 200).",
    )
    parser.add_argument(
        "--max-consecutive-faults", type=int, default=5,
        dest="max_consecutive_faults",
        help="Abort after this many consecutive pick-fault-abandoned drafts "
             "(default: 5).",
    )


@dataclass(frozen=True)
class _OnlineAgents:
    """The parsed agent wiring: the learner's binding, the frozen references, the
    seat mix and the per-label sampling temperatures."""

    learner_label: str
    learner_checkpoint: Path
    frozen: dict[str, Path]
    mix: list[tuple[str, int]]
    agent_temperatures: dict[str, float]


def _parse_online_agents(args: argparse.Namespace) -> _OnlineAgents | None:
    """The agent wiring, or None after printing the first parse error."""
    from draft.application.agent_mix import AgentMixError, parse_agent_mix
    from draft.application.agent_registry import AgentRegistryError
    from draft.application.online_agent_specs import (
        parse_agent_temperatures,
        parse_frozen_specs,
        parse_learner_spec,
    )
    from draft.application.online_config_validation import OnlineConfigError

    try:
        learner_label, learner_checkpoint = parse_learner_spec(args.learner)
        frozen = parse_frozen_specs(args.frozen)
        mix = parse_agent_mix(args.mix)
        agent_temperatures = parse_agent_temperatures(args.agent_temp)
    except (OnlineConfigError, AgentMixError, AgentRegistryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    return _OnlineAgents(
        learner_label=learner_label,
        learner_checkpoint=learner_checkpoint,
        frozen=frozen,
        mix=mix,
        agent_temperatures=agent_temperatures,
    )


def run_train_draft_agent_online(args: argparse.Namespace) -> int:
    """Execute the train-draft-agent-online command (spec 021)."""
    if _picker_checkpoint_missing(args):
        return 2
    agents = _parse_online_agents(args)
    if agents is None:
        return 2
    config = _online_config(args, agents)
    if _online_config_refusal(config):
        return 2
    return _execute_train_draft_agent_online(config)


def _online_config(args: argparse.Namespace, agents: _OnlineAgents):
    """The TrainDraftAgentOnlineConfig the flags and the parsed agent wiring give."""
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineConfig

    return TrainDraftAgentOnlineConfig(
        learner_label=agents.learner_label,
        learner_checkpoint=agents.learner_checkpoint,
        frozen=agents.frozen,
        anchor=args.anchor,
        mix=agents.mix,
        scorer_checkpoint=Path(args.scorer_checkpoint),
        build_method=args.build_method,
        picker_checkpoint=_optional_path(args.picker_checkpoint),
        cards_path=Path(args.cards_path),
        embeddings_path=_optional_path(args.embeddings_path),
        agent_temperatures=agents.agent_temperatures,
        lr=args.lr,
        drafts_per_round=args.drafts_per_round,
        anchor_window=args.anchor_window,
        snapshot_every=args.snapshot_every,
        patience=args.patience,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        min_lr=args.min_lr,
        max_rounds=args.max_rounds,
        set_code=args.set_code,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_path=Path(args.output_path),
        max_consecutive_faults=args.max_consecutive_faults,
        checkpoint_dir=Path(args.checkpoint_dir),
        worker_log_dir=Path(args.checkpoint_dir) / "logs",
    )


def _online_config_refusal(config) -> bool:
    """Print the error and return True when the config breaks a startup rule.

    Every rule runs before the Forge worker launches and before any update
    (FR-024, SC-006); execute() re-runs it so the use case is safe standalone.
    """
    from draft.application.online_config_validation import (
        OnlineConfigError,
        validate_config,
    )

    try:
        validate_config(config)
    except OnlineConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return True
    return False


def _execute_train_draft_agent_online(config) -> int:
    """Run the use case; map its refusals to exit codes."""
    from draft.application.agent_mix import AgentMixError
    from draft.application.agent_registry import AgentRegistryError
    from draft.application.generate_draft_data import MaxConsecutiveFaultsError
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineUseCase
    from draft.domain.draft_agent_model import DraftAgentArchitectureError

    try:
        TrainDraftAgentOnlineUseCase().execute(config)
    except DraftAgentArchitectureError as exc:  # a ValueError subclass — catch first
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    except MaxConsecutiveFaultsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # The loop installs SIGINT handlers and shuts down cleanly, printing the
        # final summary; reaching here means the interrupt landed during startup.
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (AgentRegistryError, AgentMixError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
