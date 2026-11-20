"""The ``train-picker`` start: a fresh picker, a resumed run or a bootstrap
from another picker's checkpoint, and its optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sealed.domain.picker_model import PickerModel
from sealed.infrastructure.picker_store import PickerStore

if TYPE_CHECKING:
    from sealed.application.train_picker import TrainPickerConfig

def _check_picker_width(
    checkpoint_dim: int, embedding_dim: int, source: str, cards_path: Path,
) -> None:
    if checkpoint_dim != embedding_dim:
        raise ValueError(
            f"{source} expects {checkpoint_dim}-wide card embeddings, but the "
            f".npz cache under {cards_path} is {embedding_dim}-wide. Re-run "
            "`sealed encode-cards` with the encoder this picker was trained on, "
            "or start a fresh picker (drop --resume)."
        )


@dataclass
class _PickerResume:
    model: PickerModel
    start_epoch: int
    best_val_reward: float
    optimizer_state: dict[str, Any] | None
    reference_state: dict[str, Any] | None  # for KL penalty (bootstrap only)


def _resume_or_build_picker(
    config: TrainPickerConfig, store: PickerStore, embedding_dim: int,
) -> _PickerResume:
    if config.resume is not None:
        ckpt = store.load_checkpoint(config.resume)
        _check_picker_width(
            ckpt.config.embedding_dim, embedding_dim,
            f"--resume {config.resume}", config.embeddings_path or config.cards_path,
        )
        model = PickerModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        return _PickerResume(
            model=model,
            start_epoch=ckpt.epoch + 1,
            best_val_reward=ckpt.best_val_reward,
            optimizer_state=ckpt.optimizer_state_dict or None,
            reference_state=None,
        )

    if config.picker_checkpoint is not None:
        ckpt = store.load_checkpoint(config.picker_checkpoint)
        _check_picker_width(
            ckpt.config.embedding_dim, embedding_dim,
            f"--picker-checkpoint {config.picker_checkpoint}",
            config.embeddings_path or config.cards_path,
        )
        model = PickerModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        return _PickerResume(
            model=model,
            start_epoch=0,
            best_val_reward=float("-inf"),
            optimizer_state=None,
            reference_state=ckpt.model_state_dict if config.kl_coef != 0 else None,
        )

    model = PickerModel(config.picker_config(embedding_dim))
    return _PickerResume(
        model=model,
        start_epoch=0,
        best_val_reward=float("-inf"),
        optimizer_state=None,
        reference_state=None,
    )


def _build_optimizer(
    model: PickerModel, config: TrainPickerConfig,
) -> torch.optim.Optimizer:
    groups = [{"params": list(model.parameters()), "lr": config.lr, "name": "picker"}]
    return torch.optim.AdamW(groups)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _restored_optimizer(
    picker: PickerModel, config: TrainPickerConfig, resume: _PickerResume,
    device: torch.device,
) -> torch.optim.Optimizer:
    """The picker's optimizer, with a resumed run's state moved to ``device``."""
    optimizer = _build_optimizer(picker, config)
    if resume.optimizer_state:
        optimizer.load_state_dict(resume.optimizer_state)
        _move_optimizer_state(optimizer, device)
    return optimizer


def _reference_picker(
    config: TrainPickerConfig, resume: _PickerResume, picker: PickerModel,
    device: torch.device,
) -> PickerModel | None:
    """The frozen bootstrap picker the KL penalty pulls toward, or None."""
    reference: PickerModel | None = None
    if config.kl_coef != 0 and resume.reference_state is not None:
        reference = PickerModel(picker.config)
        reference.load_state_dict(resume.reference_state)
        reference.eval()
        reference.to(device)
    return reference
