"""``resume.pt``: everything an interrupted training run needs to continue.

Not a checkpoint: it keeps what a checkpoint drops. Every draw is seeded by
``(seed, epoch, position, step)``, so no other state needs saving, and a resume
under different settings is refused: the run's settings, the dict every
checkpoint records too, compared but for the flags whose field declares that a
resume may change them (``run_settings``).
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import torch

from effects.application.run_settings import compared_settings
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.application.training_schedule import StopperStateError
from effects.domain.loss_denominators import LossDenominators
from effects.domain.settings_shape import SettingsShapeError, require_settings_shape
from effects.infrastructure.effect_model_store import (
    RESUME_BEST_NAME,
    RESUME_NAME,
    ResumeMismatchError,
    load_resume_state,
    resume_mismatches,
    save_resume_state,
)
from price_predictor.infrastructure.torch_checkpoint import copy_atomically

if TYPE_CHECKING:
    from effects.application.run_context import RunContext

logger = logging.getLogger(__name__)

def run_seed(config: TrainEffectModelConfig) -> int:
    """The seed every draw of the run uses: ``--seed``; else, on a resume, the
    one ``resume.pt`` records (a resume compares it, so a drawn one could never
    match); else one drawn from the OS."""
    if config.seed is not None:
        return config.seed
    if config.resume:
        state = load_resume_state(config.resolved_model_output())
        return state["settings"]["seed"]
    return random.SystemRandom().getrandbits(32)


def settings_mismatches(saved: dict, current: dict) -> list[str]:
    """Every compared setting whose value differs between a resume file's
    settings and this run's: every one recorded but the flags whose field
    declares that a resume may change them."""
    return resume_mismatches(
        compared_settings(saved, TrainEffectModelConfig),
        compared_settings(current, TrainEffectModelConfig),
    )


def _require_same_settings(saved: dict, run: RunContext) -> None:
    """Refuse a resume file recording its settings in the old shape, by name,
    or under other settings than this run's."""
    try:
        require_settings_shape(saved, holder="resume file")
    except SettingsShapeError as error:
        raise ResumeMismatchError(str(error)) from error
    mismatches = settings_mismatches(saved, run.training_settings)
    if mismatches:
        raise ResumeMismatchError(
            "The resume file was written under other settings: " + "; ".join(mismatches)
        )


def save_resume(run: RunContext, epoch: int, *, best: bool = False) -> None:
    """Everything the next epoch needs, written once the epoch is scored,
    training-only heads included."""
    directory = run.spec.config.resolved_model_output()
    save_resume_state(directory, {
        "epoch": epoch,
        "step": run.step,
        "warmup": run.warmup,
        "settings": dict(run.training_settings),
        "encoder": run.encoder.state_dict(),
        "model": run.model.state_dict(),
        "identity_table": (
            run.identity_table.state_dict()
            if run.identity_table is not None else None
        ),
        "optimizer": run.optimizer.state_dict(),
        "stopper": run.stopper.state_dict(),
        "noise": run.noise.state_dict(),
        "loss_denominators": (
            None if run.loss_denominators is None else run.loss_denominators.as_dict()
        ),
        # ``state`` None before the average's start: the resumed run starts
        # it there, as this one would have.
        "weight_average": run.average.state_dict(),
        # The stopper's own best, recorded under its data-model name too.
        "best_control_score": run.best_control_score,
        "rng": {
            "cpu": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        },
    })
    if best:
        # The best epoch's own resume state, so a later run can continue
        # from the best epoch rather than the last one (FR-063f).
        copy_atomically(directory / RESUME_NAME, directory / RESUME_BEST_NAME)


def require_optimizer_groups(saved: dict, optimizer: torch.optim.Optimizer) -> None:
    """Refuse a resume file whose optimizer holds other groups than this run's
    (one written before the weight decay was split off the biases, gains and
    tables), by name instead of torch's group-count error."""
    def shape(groups) -> list[tuple]:
        return [(g.get("name"), g.get("weight_decay"), len(g["params"])) for g in groups]

    if shape(saved["param_groups"]) != shape(optimizer.param_groups):
        raise ResumeMismatchError(
            "The resume file's optimizer groups (name, weight decay, parameters) "
            f"{shape(saved['param_groups'])} differ from this run's "
            f"{shape(optimizer.param_groups)}: it was written before weight "
            "decay was limited to the weight matrices; train from scratch."
        )


def restore_resume(run: RunContext) -> int | None:
    """The first epoch to train, with ``run`` where it left off; None when
    nothing is left. Raises ``ResumeMismatchError`` for a resume file written
    under other settings."""
    state = load_resume_state(run.spec.config.resolved_model_output())
    _require_same_settings(state["settings"], run)
    run.encoder.load_state_dict(state["encoder"])
    run.model.load_state_dict(state["model"])
    if run.identity_table is not None and state["identity_table"] is not None:
        run.identity_table.load_state_dict(state["identity_table"])
    require_optimizer_groups(state["optimizer"], run.optimizer)
    run.optimizer.load_state_dict(state["optimizer"])
    try:
        run.stopper.load_state_dict(state["stopper"])
    except StopperStateError as error:
        raise ResumeMismatchError(f"The resume file cannot continue: {error}") from error
    run.noise.load_state_dict(state["noise"], run.device)
    # Reused, never recomputed: the run divides by one set of numbers (FR-062).
    denominators = state.get("loss_denominators")
    run.loss_denominators = (
        None if denominators is None else LossDenominators.from_dict(denominators)
    )
    run.average.load_state_dict(state.get("weight_average"))
    torch.set_rng_state(state["rng"]["cpu"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    epoch, run.step = state["epoch"], state["step"]
    # The run that started fixed its warmup from its own --epochs, which a
    # resume may change; the average's start follows from it and flags a
    # resume compares, as the re-warm does from the step.
    run.fix_warmup(state["warmup"])
    # Recomputed under this run's patience rather than read from the file, so
    # a run that stopped early continues when given more patience, or none
    # (FR-063f).
    if run.stopper.should_stop or epoch >= run.spec.config.epochs:
        logger.info(
            "The resume file records a finished run (epoch %d); nothing to "
            "resume.", epoch,
        )
        return None
    logger.info(
        "Resuming after epoch %d, step %d: best control score %.4f, %d "
        "epoch(s) since.", epoch, run.step, run.stopper.best,
        run.stopper.since_best,
    )
    return epoch + 1
