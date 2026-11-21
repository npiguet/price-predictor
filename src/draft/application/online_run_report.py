"""The gen-3 online run's startup echo and closing summary (FR-019)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from draft.application.online_anchor_window import AnchorWindow
from draft.application.online_config_validation import resolve_min_lr, window_rounds
from draft.application.online_log import _fmt_dur
from draft.application.online_round_report import _fmt_margin

if TYPE_CHECKING:
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineConfig


def _format_run_control(config: TrainDraftAgentOnlineConfig) -> str:
    """The opt-in stop/anneal settings, or a plain statement that neither is on."""
    parts: list[str] = []
    if config.patience is not None:
        parts.append(f"patience {config.patience} rounds")
    if config.lr_decay_patience is not None:
        floor = resolve_min_lr(config.lr, config.min_lr)
        parts.append(
            f"lr decay x{config.lr_decay_factor} after "
            f"{config.lr_decay_patience} rounds, floor {floor:.1e}"
        )
    return " | ".join(parts) if parts else "no automatic stop, no LR annealing"


def _format_agent_temperatures(config: TrainDraftAgentOnlineConfig) -> str:
    """Every model label with its temperature, or ``argmax`` where unnamed.

    Printed in full rather than as the raw ``--agent-temp`` string so a log alone
    states how each agent played — margins are only comparable within one field
    (data-model § 7.1), and the omitted labels are exactly the ones the raw flag
    would not mention.
    """
    parts = []
    for label in [config.learner_label, *sorted(config.frozen)]:
        temperature = config.temperature_for(label)
        parts.append(f"{label}={temperature}" if temperature > 0 else f"{label}=argmax")
    return " ".join(parts)


@dataclass(frozen=True)
class ResolvedStartup:
    """What startup resolved beyond the config: the supervisor's run id, the anchor,
    the base checkpoint's lineage counter, the device and the embedding width."""

    run_id: str
    anchor: str
    generation: int
    device: torch.device
    embedding_dim: int


def format_startup_echo(
    config: TrainDraftAgentOnlineConfig, *, startup: ResolvedStartup,
) -> list[str]:
    """The resolved run configuration, echoed once before the worker launches.

    ``generation`` is the **lineage counter** read from the base checkpoint, not
    the ``--learner`` mix label: a run labeled ``gen-3`` warm-started from a
    gen-1 base prints ``generation 1 -> 2``. The label names a kind of seat; the
    counter records how many training generations deep the weights are.
    """
    from draft.application.agent_mix import format_agent_mix

    run_id, anchor, generation = startup.run_id, startup.anchor, startup.generation
    rounds = window_rounds(config)
    lines = [
        f"Online GRPO run {run_id}: generation {generation} -> {generation + 1}",
        f"  learner   : {config.learner_label} <- {config.learner_checkpoint}",
    ]
    for label in sorted(config.frozen):
        suffix = "  (anchor)" if label == anchor else ""
        lines.append(f"  frozen    : {label} <- {config.frozen[label]}{suffix}")
    reward = f"  reward    : scorer {config.scorer_checkpoint} | build-method {config.build_method}"
    if config.build_method == "picker":
        reward += f" | picker {config.picker_checkpoint}"
    lines += [
        f"  mix       : {format_agent_mix(config.mix)}  (>=1 learner seat forced)",
        reward,
        (
            f"  rollout   : T {_format_agent_temperatures(config)} | "
            f"drafts/round={config.drafts_per_round} | "
            f"set={config.set_code or 'random'} | seed={config.seed} "
            "(Forge-side rollouts unseeded)"
        ),
        (
            f"  optimiser : lr={config.lr:.0e} batch={config.batch_size} "
            f"clip={config.max_grad_norm} warmup={config.warmup_steps} steps"
        ),
        f"  run ctrl  : {_format_run_control(config)}",
        (
            f"  runtime   : device {startup.device} | embedding width {startup.embedding_dim} | "
            f"anchor window {config.anchor_window} drafts "
            f"({rounds} rounds, ~{rounds / 2:.0f}-round lag)"
        ),
        (
            f"  outputs   : corpus {config.output_path} (append) | "
            f"checkpoints {config.checkpoint_dir}/ "
            f"(snapshot every {config.snapshot_every} rounds)"
        ),
    ]
    return lines


@dataclass(frozen=True)
class RunOutcome:
    """How a run ended: its rounds, drafts and learner picks, and the checkpoints it left."""

    rounds: int
    total_drafts: int
    learner_picks: int
    latest_path: Path
    snapshot_path: Path | None
    best_path: Path | None


def format_final_summary(
    outcome: RunOutcome, *, elapsed: float, window: AnchorWindow,
) -> list[str]:
    """The FR-019 closing block, printed on every termination path."""
    lines = [
        f"Done after {outcome.rounds} rounds | {outcome.total_drafts} drafts | "
        f"{outcome.learner_picks} learner picks | {_fmt_dur(elapsed)}",
        f"  latest checkpoint : {outcome.latest_path}",
    ]
    if outcome.snapshot_path is not None:
        lines.append(f"  final snapshot    : {outcome.snapshot_path}")
    if outcome.best_path is None:
        # Either the window never filled, or the margin was never defined. Say
        # so rather than naming a checkpoint chosen on partial evidence (FR-033).
        lines.append(
            "  best checkpoint   : none recorded "
            f"(anchor window never filled: {window.window_drafts} drafts)"
            if not window.is_full
            else "  best checkpoint   : none recorded (margin never defined)"
        )
        lines.append("  best anchor margin: n/a")
    else:
        lines.append(f"  best checkpoint   : {outcome.best_path}")
        lines.append(
            f"  best anchor margin: {window.best_margin:+.3f} at round "
            f"{window.best_round} (current {_fmt_margin(window.margin)})"
        )
    return lines
