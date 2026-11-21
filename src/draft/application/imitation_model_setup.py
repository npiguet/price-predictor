"""The gen-1 model's start: the device, and a fresh model, a ``--resume`` or a
``--checkpoint`` warm start, each checked against the cache's embedding width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore, LoadedDraftAgentCheckpoint

if TYPE_CHECKING:
    from draft.application.train_draft_agent import TrainDraftAgentConfig


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _Resume:
    model: DraftAgentModel
    start_epoch: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None
    lr_decay_count: int = 0


@dataclass(frozen=True)
class CorpusShape:
    """What a fresh model is sized from, and a loaded one checked against: the
    cache's embedding width and the corpus's packs and pack size."""

    embedding_dim: int
    packs: int
    pack_size: int


def _resume_or_build(
    config: TrainDraftAgentConfig, store: DraftAgentStore, shape: CorpusShape,
) -> tuple[_Resume, float, float]:
    """Return (resume bundle, critic_mean, critic_std) — stats from checkpoint when loading."""
    if config.resume is not None:
        return _resume(config, store, shape.embedding_dim)
    if config.checkpoint is not None:
        return _warm_start(config, store, shape.embedding_dim)
    return _Resume(_fresh_model(config, shape), 0, float("inf"), None), 0.0, 1.0


def _load_checked(
    path: Path, config: TrainDraftAgentConfig, store: DraftAgentStore, embedding_dim: int,
) -> tuple[LoadedDraftAgentCheckpoint, DraftAgentModel]:
    """The checkpoint at ``path``, width-checked, and the model it holds."""
    ckpt = store.load_checkpoint(path)
    _check_dims(ckpt.config, embedding_dim, config.embeddings_path or config.cards_path)
    model = DraftAgentModel(ckpt.config)
    model.load_state_dict(ckpt.model_state_dict)
    return ckpt, model


def _resume(
    config: TrainDraftAgentConfig, store: DraftAgentStore, embedding_dim: int,
) -> tuple[_Resume, float, float]:
    """``--resume``: the run continues from the epoch after the checkpoint's."""
    ckpt, model = _load_checked(config.resume, config, store, embedding_dim)
    # Inherit the annealing position unless --lr was explicitly overridden
    # (a different base LR restarts annealing from decay_count 0). --lr is a
    # resumable flag, so an *absent* --lr resolves equal to the stored lr.
    resumed_lr = (ckpt.train_config or {}).get("lr")
    decay_count = ckpt.lr_decay_count
    if resumed_lr is not None and resumed_lr != config.lr:
        decay_count = 0
    return (
        _Resume(model, ckpt.epoch + 1, ckpt.best_val_loss,
                ckpt.optimizer_state_dict or None, decay_count),
        ckpt.critic_mean, ckpt.critic_std,
    )


def _warm_start(
    config: TrainDraftAgentConfig, store: DraftAgentStore, embedding_dim: int,
) -> tuple[_Resume, float, float]:
    """``--checkpoint``: the checkpoint's weights, a fresh optimizer from epoch 0."""
    ckpt, model = _load_checked(config.checkpoint, config, store, embedding_dim)
    return (
        _Resume(model, 0, float("inf"), None),
        ckpt.critic_mean, ckpt.critic_std,
    )


def _fresh_model(config: TrainDraftAgentConfig, shape: CorpusShape) -> DraftAgentModel:
    agent_config = DraftAgentConfig(
        embedding_dim=shape.embedding_dim,
        packs=shape.packs,
        P=shape.pack_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    )
    return DraftAgentModel(agent_config)


def _check_dims(
    ckpt_config: DraftAgentConfig, embedding_dim: int, cards_path: Path,
) -> None:
    if ckpt_config.embedding_dim != embedding_dim:
        raise ValueError(
            f"Checkpoint expects {ckpt_config.embedding_dim}-wide card "
            f"embeddings, but the .npz cache under {cards_path} is "
            f"{embedding_dim}-wide. Re-run `sealed encode-cards` with the "
            "encoder this agent was trained on."
        )
