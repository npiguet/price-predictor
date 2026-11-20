"""What a ``train-scorer`` checkpoint holds: the rolling ``latest`` saved
every epoch, the ``best`` saved on a new peak validation accuracy, the
Phase B encoder beside the scorer, and the flattened run config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sealed.application.scorer_training_context import _TrainingContext
from sealed.infrastructure.scorer_store import ScorerStore

if TYPE_CHECKING:
    from sealed.application.train_scorer import TrainScorerConfig


@dataclass(frozen=True)
class EpochAccuracy:
    """One epoch's validation accuracy beside the best one before it: what
    decides whether the epoch's checkpoint is also the new best."""

    epoch: int
    val_accuracy: float
    best_so_far: float


def _persist_checkpoint(
    ctx: _TrainingContext,
    store: ScorerStore,
    accuracy: EpochAccuracy,
) -> tuple[bool, float]:
    epoch, current_best = accuracy.epoch, accuracy.best_so_far
    encoder_state_dict = (
        ctx.encoder.state_dict() if ctx.encoder is not None else None
    )
    encoder_config = (
        asdict(ctx.encoder.config) if ctx.encoder is not None else None
    )
    store.save_checkpoint(
        ctx.model, ctx.optimizer, epoch, current_best,
        ctx.model.config, ctx.latest_path,
        train_config=ctx.train_config,
        encoder_state_dict=encoder_state_dict,
        encoder_config=encoder_config,
    )
    new_best = accuracy.val_accuracy > current_best
    if new_best:
        current_best = accuracy.val_accuracy
        store.save_checkpoint(
            ctx.model, ctx.optimizer, epoch, current_best,
            ctx.model.config, ctx.best_path,
            train_config=ctx.train_config,
            encoder_state_dict=encoder_state_dict,
            encoder_config=encoder_config,
        )
    return new_best, current_best


def _build_train_config(config: TrainScorerConfig) -> dict[str, Any]:
    """Flatten ``TrainScorerConfig`` to a JSON-friendly dict for checkpoint persistence.

    ``Path``-typed values become strings; ``None`` is preserved (FR-009,
    Decision §1, contracts/checkpoint-format.md §train_config Schema).
    """
    raw = asdict(config)
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}
