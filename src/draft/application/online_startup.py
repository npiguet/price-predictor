"""The gen-3 online run's startup steps: seeding, the learner and its two
diagnostic references, the optimizer with its LR schedule, and the pick services
(FR-002, FR-025, research D5, D17).

Each step runs once, in the order :class:`TrainDraftAgentOnlineUseCase` calls it;
the order is load-bearing, because building a model draws from the torch RNG
that :func:`seed_run` seeds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from draft.application.online_config_validation import (
    check_embedding_width,
    resolve_min_lr,
    validate_config,
)
from draft.application.online_diagnostics import SweepPolicies
from draft.application.online_update import Learner
from draft.domain.draft_agent_model import DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

if TYPE_CHECKING:
    from draft.application.train_draft_agent_online import TrainDraftAgentOnlineConfig


@dataclass(frozen=True)
class SeededRun:
    """A validated config's anchor and learner temperature, with the run's seeded RNG."""

    anchor: str
    temperature: float
    rng: random.Random


def seed_run(config: TrainDraftAgentOnlineConfig) -> SeededRun:
    """Validate the config, then seed torch, numpy and the run's own RNG."""
    anchor = validate_config(config)
    temperature = float(config.rollout_temperature)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)
    return SeededRun(anchor=anchor, temperature=temperature, rng=rng)


@dataclass(frozen=True)
class LoadedLearner:
    """The learner's base checkpoint and the store that read it, the embedding width
    it was checked against, the device, and the policies built from it."""

    store: DraftAgentStore
    base: Any
    embedding_dim: int
    device: torch.device
    policies: SweepPolicies


def load_learner(config: TrainDraftAgentOnlineConfig) -> LoadedLearner:
    """Load the learner: the model the loop optimises AND the model that drafts."""
    from draft.application.imitation_model_setup import _select_device

    store = DraftAgentStore()
    base = store.load_checkpoint(config.learner_checkpoint)
    embedding_dim = check_embedding_width(base.config, config)
    device = _select_device()
    return LoadedLearner(
        store=store, base=base, embedding_dim=embedding_dim, device=device,
        policies=_load_policies(base, device),
    )


def _load_policies(base: Any, device: torch.device) -> SweepPolicies:
    """The live model, πₖ and π₀, built in that order."""
    model = DraftAgentModel(base.config)
    model.load_state_dict(base.model_state_dict)
    model.to(device)
    model.eval()
    # πₖ for the movement diagnostics: resident, never in train mode.
    prev_model = _reference_model(base.config, device, state=None)
    # π₀ for the cumulative-KL diagnostic: the learner's own warm start,
    # frozen for the whole run. Not the anchor — that is a different
    # checkpoint whenever --learner and --frozen differ (research D17).
    init_model = _reference_model(base.config, device, state=base.model_state_dict)
    return SweepPolicies(prev_model=prev_model, model=model, init_model=init_model)


def _reference_model(
    model_config: Any, device: torch.device, *, state: dict[str, Any] | None,
) -> DraftAgentModel:
    """A model that is only ever forwarded: in ``eval()``, with no gradients."""
    reference = DraftAgentModel(model_config)
    if state is not None:
        reference.load_state_dict(state)
    reference.to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def build_optimisation(
    config: TrainDraftAgentOnlineConfig, model: DraftAgentModel,
) -> tuple[Learner, Any]:
    """The optimizer and LR schedule that step ``model``, and the plateau controller."""
    from draft.application.draft_lr_schedule import _make_scheduler, _PlateauLR

    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": config.lr, "name": "agent"}],
    )
    # One warmup at the start of the run, then constant — never rebuilt per
    # round, so the LR schedule and optimizer moments are continuous across
    # rounds (FR-025). An online run has no total step count, so the ramp is
    # expressed in optimizer steps rather than a fraction of a horizon (D15).
    # The plateau controller folds its multiplier into the same warmup
    # lambda, so a decay scales the post-warmup constant without re-running
    # the ramp (FR-035). Constructed even when annealing is disabled — with
    # no decay ever taken its multiplier stays 1.0.
    plateau = _PlateauLR(
        base_lr=config.lr,
        factor=config.lr_decay_factor,
        min_lr=resolve_min_lr(config.lr, config.min_lr),
    )
    scheduler = _make_scheduler(
        optimizer, total_steps=config.warmup_steps, warmup_frac=1.0,
        controller=plateau,
    )
    return Learner(model=model, optimizer=optimizer, scheduler=scheduler), plateau


def build_pick_services(
    config: TrainDraftAgentOnlineConfig, loaded: LoadedLearner, locator: Any,
    temperature: float,
) -> dict[str, Any]:
    """One pick service per frozen label, then the learner's, served by the live model.

    Per-label pick modes (research D5), read straight off --agent-temp: a
    label it names samples at that T, a label it omits plays argmax, its
    best. Argmax is the default because a draft allocates one fixed pool of
    cards, so a card a frozen agent wrongly declines does not vanish — it
    flows downstream to its podmates. A sampled field hands the learner
    cards a properly-playing agent would have kept, making the training
    field weak in exactly the dimension drafting is about, and unlike the
    evaluation field, where every agent plays argmax. Naming a frozen label
    anyway buys pod diversity at that price; the operator decides.
    """
    from draft.application.agent_pick_service import AgentPickService

    services = {}
    for index, (label, path) in enumerate(sorted(config.frozen.items())):
        frozen_temperature = config.temperature_for(label)
        services[label] = AgentPickService(
            path, locator, device=loaded.device,
            pick_mode="sample" if frozen_temperature > 0 else "argmax",
            temperature=frozen_temperature or 1.0,
            # A distinct stream per sampled agent: sharing the learner's seed
            # would couple the field's exploration to the learner's.
            seed=config.seed + 1 + index if frozen_temperature > 0 else None,
        )
    services[config.learner_label] = AgentPickService.from_model(
        loaded.policies.model, loaded.base.config, locator, device=loaded.device,
        pick_mode="sample", temperature=temperature, seed=config.seed,
    )
    return services
