"""``resume.pt``: everything an interrupted training run needs to continue.

Not a checkpoint: it keeps what a checkpoint drops. Every draw is seeded by
``(seed, epoch, position, step)``, so no other state needs saving, and a resume
under different settings is refused.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import fields
from typing import TYPE_CHECKING

import torch

from effects.application.train_effect_model import SWEEP_FLAGS, RunSpec, TrainEffectModelConfig
from effects.infrastructure.effect_model_store import (
    RESUME_BEST_NAME,
    RESUME_NAME,
    ResumeMismatchError,
    load_resume_state,
    resume_mismatches,
    save_resume_state,
    tokenizer_hashes,
)

if TYPE_CHECKING:
    from effects.application.run_context import RunContext

logger = logging.getLogger(__name__)

#: Flags a resumed run may set differently from the run it continues. The
#: corpus is compared by its digest instead, the vocabulary and keyword
#: definitions by their bytes, and the holdout unit is the corpus's own.
RESUME_MAY_CHANGE: frozenset[str] = frozenset({
    "epochs", "patience", "lr_scale", "save_every_epoch", "model_output",
    "prefetch_workers", "resume", "corpus", "cards_folders", "variant_scripts",
    "holdout_unit", "vocab_path", "keyword_definitions",
})
#: Flags the checkpoint's training settings record, and the resume file
#: compares there as recorded: ``seed`` as the value the run drew, and
#: ``lethality_copies`` only when set.
RECORDED_IN_TRAINING_SETTINGS: frozenset[str] = frozenset({
    "seed", "lethality_copies", *SWEEP_FLAGS,
})


def compared_flags() -> tuple[str, ...]:
    """The config fields a resume compares directly, in declaration order."""
    return tuple(
        spec.name for spec in fields(TrainEffectModelConfig)
        if spec.name not in RESUME_MAY_CHANGE | RECORDED_IN_TRAINING_SETTINGS
    )


def resume_settings(spec: RunSpec, *, training_settings: dict) -> dict:
    """Every setting a resumed epoch has to share with the interrupted run:
    every flag but :data:`RESUME_MAY_CHANGE`, so a flag added later is checked
    by default, the corpus by its digest, and the training settings as
    recorded."""
    config = spec.config
    return {
        "seed": spec.seed,
        "corpus_digest": spec.facts.digest,
        **tokenizer_hashes(config.tokenizer_files),
        **{name: getattr(config, name) for name in compared_flags()},
        **training_settings,
    }


def _settings_of(run: RunContext) -> dict:
    return resume_settings(run.spec, training_settings=run.training_settings)


def save_resume(run: RunContext, epoch: int, *, best: bool = False) -> None:
    """Everything the next epoch needs, written once the epoch is scored,
    training-only heads included."""
    directory = run.spec.config.resolved_model_output()
    save_resume_state(directory, {
        "epoch": epoch,
        "step": run.step,
        "warmup": run.warmup,
        "settings": _settings_of(run),
        "encoder": run.encoder.state_dict(),
        "model": run.model.state_dict(),
        "identity_table": (
            run.identity_table.state_dict()
            if run.identity_table is not None else None
        ),
        "optimizer": run.optimizer.state_dict(),
        "stopper": run.stopper.state_dict(),
        "noise": run.noise.state_dict(),
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
        shutil.copyfile(directory / RESUME_NAME, directory / RESUME_BEST_NAME)


def restore_resume(run: RunContext) -> int | None:
    """The first epoch to train, with ``run`` where it left off; None when
    nothing is left. Raises ``ResumeMismatchError`` for a resume file written
    under other settings."""
    state = load_resume_state(run.spec.config.resolved_model_output())
    mismatches = resume_mismatches(state["settings"], _settings_of(run))
    if mismatches:
        raise ResumeMismatchError(
            "The resume file was written under other settings: "
            + "; ".join(mismatches)
        )
    run.encoder.load_state_dict(state["encoder"])
    run.model.load_state_dict(state["model"])
    if run.identity_table is not None and state["identity_table"] is not None:
        run.identity_table.load_state_dict(state["identity_table"])
    run.optimizer.load_state_dict(state["optimizer"])
    run.stopper.load_state_dict(state["stopper"])
    run.noise.load_state_dict(state["noise"], run.device)
    torch.set_rng_state(state["rng"]["cpu"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    epoch, run.step = state["epoch"], state["step"]
    # The run that started fixed its warmup from its own --epochs, which a
    # resume may change.
    run.warmup = state["warmup"]
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
        "Resuming after epoch %d, step %d: best card-disjoint %.4f, %d "
        "epoch(s) since.", epoch, run.step, run.stopper.best,
        run.stopper.since_best,
    )
    return epoch + 1
