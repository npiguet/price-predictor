"""Gen-2 RL self-play fine-tuning of the two-headed draft agent (spec 020).

On-policy actor-critic (REINFORCE + GAE baseline + KL anchor + entropy bonus)
over a self-play corpus the reference checkpoint generated. Per learner pick the
loss is

    L = − A_t·logπ_T(a_t|s_t)              policy gradient (learner picks only)
      + value_weight · MSE(V(s_t), R_std)  critic regression (all non-failed seats)
      − entropy_coef · H(π_T(·|s_t))       exploration bonus (learner picks)
      + kl_coef · KL(π_T ‖ π_ref,T)        anchor to the reference policy

where π_T is the PACK-masked softmax at the rollout temperature T (research D4),
A_t is the detached GAE(λ) advantage over the seat's ordered trajectory
(research D2/D3), and R is the pod-relative leave-one-out deck_score (terminal).

Mirrors ``train_draft_agent``'s scaffolding (warmup-then-constant LR, per-group
clip, mini-epoch eval + best/latest + early stop, LR-plateau annealing); the
per-pick state walk is the shared :mod:`draft.application.draft_pick_states`. The
REINFORCE helpers (entropy/KL) follow ``sealed.application.picker_training_step``. On-
policy pairing of corpus↔checkpoint is the operator's responsibility — the only
safeguard is the behaviour-anomaly warning (research D6); the corpus is unchanged
(no provenance field, FR-021).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from draft.application.imitation_model_setup import _select_device
from draft.application.rl_behaviour_check import (
    _behaviour_anomaly_summary,
    report_behaviour_check,
)
from draft.application.rl_examples import RANDOM_SEED
from draft.application.rl_log import _log
from draft.application.rl_training_loop import (
    RLCheckpointTargets,
    RLTrainingData,
    RLTrainingLoop,
    RLTrainingProgress,
)
from draft.application.rl_training_setup import (
    build_learner,
    checkpoint_targets,
    load_corpus,
    start_run,
    steps_per_epoch,
)
from draft.infrastructure.draft_agent_store import DraftAgentStore


@dataclass(kw_only=True)
class TrainDraftAgentRLConfig:
    """Inputs to the RL trainer (data-model §1).

    ``checkpoint``/``learner_agents``/``rollout_temperature`` are required on a
    fresh run (enforced by the CLI / startup validation); they are resumable so
    ``--resume`` inherits them. Architecture is inherited from the loaded
    checkpoint — there are no architecture flags.
    """

    checkpoint: Path | None = None
    resume: Path | None = None
    drafts_path: Path
    critic_corpus: tuple[Path, ...] = ()
    cards_path: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    #: Where the checkpoints (``latest.pt``, ``<timestamp>.pt``) go.
    checkpoint_dir: Path
    learner_agents: tuple[str, ...] = ()
    rollout_temperature: float | None = None
    gae_lambda: float = 0.95
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    entropy_decay_after: int = 5
    value_weight: float = 1.0
    lr: float = 3e-4
    warmup_frac: float = 0.05
    batch_size: int = 32
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.0025
    evals_per_epoch: int = 100
    patience: int = 30
    lr_decay_patience: int | None = None
    lr_decay_factor: float = 0.1
    min_lr: float | None = None


@dataclass
class TrainDraftAgentRLResult:
    best_val_loss: float
    best_epoch: int
    best_path: Path
    generation: int


class TrainDraftAgentRLUseCase:
    """On-policy actor-critic RL fine-tuning (spec 020 FR-001 … FR-021)."""

    def execute(self, config: TrainDraftAgentRLConfig) -> TrainDraftAgentRLResult:
        self._validate_config(config)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = DraftAgentStore()

        corpus = load_corpus(config)
        start = start_run(config, store, corpus, device)
        report_behaviour_check(_behaviour_anomaly_summary(
            start.models.reference, corpus.examples, start.source, config,
        ))
        n_steps = steps_per_epoch(config, start)
        learner = build_learner(config, start, n_steps)
        targets = checkpoint_targets(config, store, start.resume.generation)
        val = sorted(start.val, key=lambda ex: ex.n_tokens)
        data = RLTrainingData(start.train, val, corpus.trajectories, start.source, n_steps)

        resume = start.resume
        progress = RLTrainingProgress(
            start_epoch=resume.start_epoch,
            best_val_loss=resume.best_val_loss,
            best_at=(resume.start_epoch, 0),
        )
        RLTrainingLoop(config, learner, data, targets).run(progress)
        return _finish(progress, targets)

    @staticmethod
    def _validate_config(config: TrainDraftAgentRLConfig) -> None:
        """Startup validation (FR-017, SC-003): the first rule broken is refused."""
        _check_one_start(config)
        _check_learner_agents(config)
        _check_rollout_temperature(config)
        _check_lr_decay(config)


def _check_one_start(config: TrainDraftAgentRLConfig) -> None:
    if (config.checkpoint is None) == (config.resume is None):
        raise ValueError("exactly one of --checkpoint or --resume is required")


def _check_learner_agents(config: TrainDraftAgentRLConfig) -> None:
    if not config.learner_agents:
        raise ValueError("--learner-agents must be a non-empty whitelist")


def _check_rollout_temperature(config: TrainDraftAgentRLConfig) -> None:
    if config.rollout_temperature is None or config.rollout_temperature <= 0:
        raise ValueError("--rollout-temperature must be supplied and positive")


def _check_lr_decay(config: TrainDraftAgentRLConfig) -> None:
    """The plateau-decay flags, checked only when ``--lr-decay-patience`` is set."""
    patience = config.lr_decay_patience
    if patience is None:
        return
    if patience < 1:
        raise ValueError("--lr-decay-patience must be >= 1")
    if not 0.0 < config.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be in (0, 1)")
    if patience >= config.patience:
        raise ValueError("--lr-decay-patience must be < --patience")


def _finish(
    progress: RLTrainingProgress, targets: RLCheckpointTargets,
) -> TrainDraftAgentRLResult:
    best_epoch, best_step = progress.best_at
    _log(
        f"Done. Best val_loss={progress.best_val_loss:.4f} at epoch {best_epoch} "
        f"step {best_step}; best checkpoint: {targets.best_path} (generation "
        f"{targets.generation})"
    )
    return TrainDraftAgentRLResult(
        progress.best_val_loss, best_epoch, targets.best_path, targets.generation,
    )
