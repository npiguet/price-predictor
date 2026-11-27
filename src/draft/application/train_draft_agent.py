"""Train the two-headed draft agent (policy + critic) on a recorded corpus.

Each ``(draft, seat, pack, pick)`` is one typed-token example (FR-030). The
imitation policy is trained with cross-entropy over the whitelisted seats'
``PACK`` tokens (FR-033); the Monte-Carlo critic is trained with MSE over the
standardized leave-one-out pod-relative reward of every non-failed seat
(FR-032). Loss = ``imitation_weight·CE + critic_weight·MSE``.

Follows the ``train_picker`` / ``train_scorer`` / ``train_encoder`` scaffolding
(warmup-then-constant LR, per-group max-norm clip, resume/bootstrap guard,
best-by-val + ``latest.pt``, early stop) without extracting a shared trainer —
see research §Third-instance check.

This module holds the config and the use case; the run's start is in
``imitation_training_setup`` and its epochs in ``imitation_training_loop``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from draft.application.imitation_examples import RANDOM_SEED
from draft.application.imitation_log import _log
from draft.application.imitation_model_setup import _select_device
from draft.application.imitation_training_loop import (
    CheckpointTargets,
    ImitationTrainingLoop,
    TrainingData,
    TrainingProgress,
)
from draft.application.imitation_training_setup import (
    build_learner,
    checkpoint_targets,
    load_corpus,
    start_model,
    steps_per_epoch,
)
from draft.infrastructure.draft_agent_store import DraftAgentStore


@dataclass(kw_only=True)
class TrainDraftAgentConfig:
    drafts_path: Path
    cards_path: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    #: Where the checkpoints (``latest.pt``, ``<timestamp>.pt``) go.
    checkpoint_dir: Path
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.0
    imitation_weight: float = 1.0
    critic_weight: float = 1.0
    imitation_agents: tuple[str, ...] = ("forge-full",)
    lr: float = 3e-4
    warmup_frac: float = 0.05
    batch_size: int = 32
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.0025
    evals_per_epoch: int = 100  # "mini-epochs": validate + checkpoint this often
    patience: int = 30  # counted in mini-epochs (evals), not full epochs
    # Optional ReduceLROnPlateau-style annealing: after lr_decay_patience
    # mini-epochs without a new best val_loss, LR *= lr_decay_factor, down to
    # min_lr. None disables (constant LR after warmup, the default).
    lr_decay_patience: int | None = None
    lr_decay_factor: float = 0.1
    min_lr: float | None = None  # decay floor; None => lr * 1e-3
    resume: Path | None = None
    checkpoint: Path | None = None


def _validate_lr_decay(config: TrainDraftAgentConfig) -> None:
    """Fail fast on inconsistent LR-annealing flags (no-op when disabled)."""
    if config.lr_decay_patience is None:
        return
    if config.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be >= 1")
    if not 0.0 < config.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be in (0, 1)")
    if config.lr_decay_patience >= config.patience:
        raise ValueError(
            f"--lr-decay-patience ({config.lr_decay_patience}) must be < "
            f"--patience ({config.patience}) so a decay pre-empts early stop"
        )


@dataclass
class TrainDraftAgentResult:
    best_val_loss: float
    best_epoch: int
    best_path: Path


class TrainDraftAgentUseCase:
    """Joint policy + critic training (FR-030 … FR-039)."""

    def execute(self, config: TrainDraftAgentConfig) -> TrainDraftAgentResult:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = DraftAgentStore()
        _validate_lr_decay(config)

        corpus = load_corpus(config)
        resume, source = start_model(config, store, corpus, device)
        n_steps = steps_per_epoch(config, corpus)
        learner = build_learner(config, resume, n_steps, device)
        targets = checkpoint_targets(config, store)
        # Sort val once by length so eval batches are low-padding (cheap eval).
        val = sorted(corpus.val, key=lambda ex: ex.n_tokens)
        data = TrainingData(corpus.train, val, source, n_steps)

        progress = TrainingProgress(
            start_epoch=resume.start_epoch,
            best_val_loss=resume.best_val_loss,
            best_at=(resume.start_epoch, 0),
        )
        ImitationTrainingLoop(config, learner, data, targets).run(progress)
        return _finish(progress, targets)


def _finish(progress: TrainingProgress, targets: CheckpointTargets) -> TrainDraftAgentResult:
    best_epoch, best_step = progress.best_at
    _log(
        f"Done. Best val_loss={progress.best_val_loss:.4f} at epoch {best_epoch} "
        f"step {best_step}; best checkpoint: {targets.best_path}"
    )
    return TrainDraftAgentResult(progress.best_val_loss, best_epoch, targets.best_path)
