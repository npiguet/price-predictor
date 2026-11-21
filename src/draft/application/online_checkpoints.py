"""The gen-3 checkpoint payload: gen-1 format, the critic carried through, and the
self-describing ``rl_metadata`` (data-model §8, FR-027/FR-028).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from draft.domain.draft_agent_model import DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

if TYPE_CHECKING:
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineConfig


# --------------------------------------------------------------------------- #
# Checkpoint payload (data-model §8, FR-027/FR-028)
# --------------------------------------------------------------------------- #

def _stringify(value: Any) -> Any:
    """Recursively render Paths as strings so the config survives torch.save."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    return value


def base_generation(rl_metadata: dict[str, Any] | None) -> int:
    """The base checkpoint's lineage counter (gen-1 checkpoints carry none)."""
    if not rl_metadata:
        return 1
    return int(rl_metadata.get("generation", 1))


def build_rl_metadata(
    config: TrainDraftAgentOnlineConfig, generation: int, lr_decay_count: int = 0,
) -> dict[str, Any]:
    """Gen-3's self-describing metadata — no critic/GAE/KL/entropy knobs exist.

    ``lr`` is the **base** rate, not the annealed value; ``lr_decay_count``
    records how far the run annealed (effective LR =
    ``lr × lr_decay_factor ** lr_decay_count``). The count is provenance only and
    is never read back — gen-3 has no ``--resume``, so restarting from a snapshot
    deliberately re-runs warmup and resets both the moments and the decay
    position (research D13).
    """
    return {
        "generation": generation + 1,
        "base_checkpoint": str(config.learner_checkpoint),
        "algorithm": "online-grpo",
        "lr": config.lr,
        # The learner's own, kept under its gen-2-era name for readers that
        # expect a single scalar; the map beside it records the whole field, so
        # a checkpoint states the conditions its margins were measured under.
        "rollout_temperature": config.rollout_temperature,
        "agent_temperatures": dict(config.agent_temperatures or {}),
        "drafts_per_round": config.drafts_per_round,
        "lr_decay_count": lr_decay_count,
    }


def _write_checkpoint(
    store: DraftAgentStore,
    model: DraftAgentModel,
    optimizer: torch.optim.Optimizer,
    config: TrainDraftAgentOnlineConfig,
    base: Any,
    generation: int,
    round_index: int,
    path: Path,
    lr_decay_count: int = 0,
) -> None:
    """Write one gen-1-format checkpoint carrying the critic head unchanged.

    ``best_val_loss`` is ``inf`` because no held-out metric exists (FR-026), and
    ``critic_mean``/``critic_std`` are copied verbatim from the base so the
    (untrained) critic still de-standardizes to raw scorer-score space.
    """
    store.save_checkpoint(
        model,
        optimizer,
        round_index,
        float("inf"),
        base.config,
        path,
        critic_mean=base.critic_mean,
        critic_std=base.critic_std,
        train_config=_stringify(asdict(config)),
        lr_decay_count=lr_decay_count,
        rl_metadata=build_rl_metadata(config, generation, lr_decay_count),
    )
