"""``resume.pt``: everything an interrupted training run needs to continue.

Not a checkpoint. It carries the full model with the training-only heads, the
optimizer, the step, the early stopper, the noise on ``e`` and the CPU/CUDA
generator states — everything a checkpoint drops. No other state needs saving,
because every draw a run takes is seeded by ``(seed, epoch, position, step)``;
for the same reason a resume under a different seed, corpus, size or batching is
refused, while the epoch count and the patience may change.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable

import torch

from effects.application.train_effect_model import EarlyStopper, TrainEffectModelConfig
from effects.infrastructure.effect_model_store import (
    RESUME_BEST_NAME,
    RESUME_NAME,
    ResumeMismatchError,
    load_resume_state,
    resume_mismatches,
    save_resume_state,
)

logger = logging.getLogger(__name__)


class ResumeState:
    """Writes a run's resume file after each epoch and restores a run from it."""

    def __init__(
        self, config: TrainEffectModelConfig, *, seed: int, corpus_digest: str,
        training_settings: Callable[[], dict], noise, identity_table,
        device: torch.device,
    ) -> None:
        self.config = config
        self.seed = seed
        self.corpus_digest = corpus_digest
        #: The checkpoint's training settings, read when a file is written or
        #: compared, after the run has fixed its script-API vocabularies.
        self.training_settings = training_settings
        self.noise = noise
        self.identity_table = identity_table
        self.device = device

    def settings(self) -> dict:
        """Every setting a resumed epoch has to share with the interrupted run.

        The draws need no state of their own: every shard draw, shuffle and
        per-step draw is seeded by the seed, the epoch and the step's place in
        it, so the seed and the shapes that place a step are what must match.
        The epoch count and the patience are left out, so an interrupted run
        can be resumed with more epochs or more patience.
        """
        config = self.config
        return {
            "seed": self.seed,
            "corpus_digest": self.corpus_digest,
            "variant": config.variant,
            "e_dim": config.e_dim,
            "encoder_layers": config.encoder_layers,
            "encoder_d_model": config.encoder_d_model,
            "steps_per_epoch": config.steps_per_epoch,
            "shards_per_epoch": config.shards_per_epoch,
            "batch_size": config.batch_size,
            "grad_accum": config.grad_accum,
            "curriculum_epoch": config.curriculum_epoch,
            "withhold_keyword": config.withhold_keyword,
            "keyword_expand_p": config.keyword_expand_p,
            "context_dropout": config.context_dropout,
            **self.training_settings(),
        }

    def save(
        self, epoch: int, run, stopper: EarlyStopper, *, finished: bool,
        best: bool = False,
    ) -> None:
        """Everything the next epoch needs, written once the epoch is scored.

        The model's whole state, training-only heads included — the saved
        checkpoint drops those, and a resumed run would have to relearn them —
        with the optimizer's moments, the noise's running covariance and the
        random generators, which dropout and the noise draw read on the device.
        """
        directory = self.config.resolved_model_output()
        save_resume_state(directory, {
            "epoch": epoch,
            "step": run.step,
            "finished": finished,
            "settings": self.settings(),
            "encoder": run.encoder.state_dict(),
            "model": run.model.state_dict(),
            "identity_table": (
                self.identity_table.state_dict()
                if self.identity_table is not None else None
            ),
            "optimizer": run.optimizer.state_dict(),
            "stopper": stopper.state_dict(),
            "noise": self.noise.state_dict(),
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

    def restore(self, run, stopper: EarlyStopper) -> int | None:
        """The first epoch to train, with ``run`` where it left off; None when
        nothing is left.

        Refuses a resume file written under other settings: resumed from a
        different seed or corpus, the run would be neither the interrupted one
        nor a fresh one.
        """
        state = load_resume_state(self.config.resolved_model_output())
        mismatches = resume_mismatches(state["settings"], self.settings())
        if mismatches:
            raise ResumeMismatchError(
                "The resume file was written under other settings: "
                + "; ".join(mismatches)
            )
        run.encoder.load_state_dict(state["encoder"])
        run.model.load_state_dict(state["model"])
        if self.identity_table is not None and state["identity_table"] is not None:
            self.identity_table.load_state_dict(state["identity_table"])
        run.optimizer.load_state_dict(state["optimizer"])
        stopper.load_state_dict(state["stopper"])
        self.noise.load_state_dict(state["noise"], self.device)
        torch.set_rng_state(state["rng"]["cpu"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        epoch, run.step = state["epoch"], state["step"]
        # Recomputed under this run's patience rather than read from the file,
        # so a run that stopped early continues when given more patience, or
        # none (FR-063f).
        if stopper.should_stop or epoch >= self.config.epochs:
            logger.info(
                "The resume file records a finished run (epoch %d); nothing "
                "to resume.", epoch,
            )
            return None
        logger.info(
            "Resuming after epoch %d, step %d: best card-disjoint %.4f, %d "
            "epoch(s) since.", epoch, run.step, stopper.best, stopper.since_best,
        )
        return epoch + 1
