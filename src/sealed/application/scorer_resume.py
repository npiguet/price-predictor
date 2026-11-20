"""What a ``train-scorer`` run starts from: a ``--resume`` checkpoint, a
``--scorer-checkpoint`` bootstrap or a fresh scorer, each checked against
the width of the ``.npz`` cache, and the AdamW optimizer over the scorer's
(and in Phase B the encoder's) parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore

if TYPE_CHECKING:
    from sealed.application.train_scorer import TrainScorerConfig


@dataclass
class ResumeState:
    model: SetTransformerScorer
    start_epoch: int
    best_val_accuracy: float
    optimizer_state: dict | None
    encoder_state_dict: dict | None = None
    encoder_config: dict | None = None
    phase: Literal["A", "B"] = "A"


def _check_scorer_width(
    checkpoint_d_model: int, embedding_dim: int, source: str, cards_path: Path,
) -> None:
    """Raise if a loaded scorer expects a different card-embedding width than
    the ``.npz`` cache currently provides — almost always a sign the cache
    was re-encoded with a different encoder than the scorer was trained on."""
    if checkpoint_d_model != embedding_dim:
        raise ValueError(
            f"{source} expects {checkpoint_d_model}-wide card embeddings, but "
            f"the .npz cache under {cards_path} is {embedding_dim}-wide. "
            "Re-run `sealed encode-cards` with the encoder this scorer was "
            "trained on, or start a fresh scorer (drop --scorer-checkpoint)."
        )


def _resume_or_build_model(
    config: TrainScorerConfig, store: ScorerStore, embedding_dim: int,
) -> ResumeState:
    """Build the scorer model and decide what to seed it (and the encoder)
    from. ``embedding_dim`` is the width of the cached card embeddings (read
    off the ``.npz`` cache); a fresh scorer is sized to it, and a loaded
    scorer is validated against it. Cross-phase resume MUST raise per FR-004
    (the CLI catches that earlier; this is the second layer of defense)."""
    requested_phase = config.phase

    if config.resume is not None:
        checkpoint = store.load_checkpoint(config.resume)
        ckpt_phase: Literal["A", "B"] = (
            "B" if checkpoint.encoder_state_dict is not None else "A"
        )
        if ckpt_phase != requested_phase:
            raise ValueError(
                f"--resume {config.resume} is a Phase {ckpt_phase} checkpoint "
                f"but --embedding-lr {config.embedding_lr} requests Phase "
                f"{requested_phase}. Use --scorer-checkpoint to start a fresh "
                "Phase B run from a Phase A checkpoint."
            )
        _check_scorer_width(
            checkpoint.config.d_model, embedding_dim,
            f"--resume {config.resume}", config.embeddings_path or config.cards_path,
        )
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        return ResumeState(
            model=model,
            start_epoch=checkpoint.epoch + 1,
            best_val_accuracy=checkpoint.best_val_accuracy,
            optimizer_state=checkpoint.optimizer_state_dict or None,
            encoder_state_dict=checkpoint.encoder_state_dict,
            encoder_config=checkpoint.encoder_config,
            phase=ckpt_phase,
        )

    if config.scorer_checkpoint is not None:
        checkpoint = store.load_checkpoint(config.scorer_checkpoint)
        _check_scorer_width(
            checkpoint.config.d_model, embedding_dim,
            f"--scorer-checkpoint {config.scorer_checkpoint}",
            config.embeddings_path or config.cards_path,
        )
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        return ResumeState(
            model=model,
            start_epoch=0,
            best_val_accuracy=-1.0,
            optimizer_state=None,  # scorer-checkpoint bootstrap resets training-loop state
            encoder_state_dict=None,  # encoder weights come from --encoder-checkpoint
            encoder_config=None,
            phase=requested_phase,
        )

    return ResumeState(
        model=SetTransformerScorer(config.scorer_config(d_model=embedding_dim)),
        start_epoch=0,
        best_val_accuracy=-1.0,
        optimizer_state=None,
        phase=requested_phase,
    )


def _build_optimizer(
    model: SetTransformerScorer,
    config: TrainScorerConfig,
    encoder: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer.

    Phase A: a single ``"scorer"`` parameter group at ``config.lr``.
    Phase B (``encoder`` supplied): two groups — scorer at ``config.lr`` and
    encoder at ``config.embedding_lr`` — for per-group clipping (FR-005a, FR-008).
    The ``"name"`` key on each group is ignored by AdamW but read by
    ``_train_one_epoch`` to label the per-group gradient norm.
    """
    groups: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": config.lr, "name": "scorer"},
    ]
    if encoder is not None:
        groups.append(
            {"params": list(encoder.parameters()), "lr": config.embedding_lr,
             "name": "encoder"},
        )
    return torch.optim.AdamW(groups)


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    """Move Adam momentum/variance buffers to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
