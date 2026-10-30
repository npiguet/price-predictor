"""What a training run says between its steps, and the tallies it says it from.

One home for every line's wording, and for the counts gathered from the worker
processes that those lines report.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from effects.application.step_loss import StepLoss, read_back
from effects.application.step_preparation import (
    NO_TEXT_BUCKET,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.train_effect_model import (
    CorpusFacts,
    EpochResult,
    TrainEffectModelConfig,
    check_holdout,
)
from effects.application.training_log import (
    format_norms,
    format_parts,
    format_shares,
    format_terms,
)
from effects.application.validation_pass import ValidationSamples
from effects.domain.ability_encoder import TruncationLog, log_truncation
from effects.domain.gate_metrics import GateOneMetrics
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.sampling import SAMPLING_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class EpochTally:
    """One epoch's training numbers, summed on the device and read once."""

    running: torch.Tensor
    taken: int = 0
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Trained records by rarity bucket and by rule family (FR-055).
    buckets: Counter[str] = field(default_factory=Counter)
    families: Counter[str] = field(default_factory=Counter)

    def add_shares(self, ready: StepReady) -> None:
        self.buckets.update(ready.buckets)
        self.families.update(ready.families)

    def add_step(self, loss: StepLoss) -> None:
        self.running += loss.shipped.detach()
        self.taken += 1
        for term, value in loss.terms.items():
            held = self.terms.get(term)
            self.terms[term] = value.detach() if held is None else held + value.detach()

    @property
    def train_loss(self) -> float:
        return float(self.running) / max(self.taken, 1)

    def mean_terms(self) -> dict[str, float]:
        steps = max(self.taken, 1)
        return {name: value / steps for name, value in read_back(self.terms).items()}


@dataclass
class ShardStats:
    """One shard's training numbers, for the line it logs when it ends."""

    loss: torch.Tensor
    waited: float = 0.0
    batch_waited: float = 0.0
    steps: int = 0
    learning_rate: float = 0.0
    #: The last reported step's loss by field, gradient norms and head terms.
    parts: dict[str, float] = field(default_factory=dict)
    norms: dict[str, float] = field(default_factory=dict)
    heads: dict[str, float] = field(default_factory=dict)
    #: The last optimizer step's norms, still on the device until reported.
    unread_norms: dict[str, torch.Tensor] = field(default_factory=dict)

    def add_step(self, loss: StepLoss, *, reported: bool) -> None:
        self.loss += loss.shipped.detach()
        self.steps += 1
        if reported:
            self.heads = read_back(loss.terms)
        if loss.parts:
            self.parts = loss.parts.as_dict()


class RunReporter:
    """Every line a training run logs between steps, and the tallies those
    lines gather from every process."""

    def __init__(self, config: TrainEffectModelConfig, facts: CorpusFacts) -> None:
        self.config = config
        self.facts = facts
        #: One per run, so a truncated text is reported once (FR-006).
        self.truncations = TruncationLog(
            log_truncation(logger, "train-effect-model"),
        )
        self._rarity_reported = False
        #: Unconverted-script lookups counted by the workers' own sidecar caches.
        self._unresolved_elsewhere: Counter[str] = Counter()

    def missing_classes(self, present: frozenset[str]) -> None:
        """Say which sampling classes the validation samples do not name: the
        fields only those classes supervise silently drop for the whole run."""
        missing = [name for name in SAMPLING_CLASSES if name not in present]
        if not missing:
            return
        logger.warning(
            "The corpus's validation samples name %d of the %d sampling "
            "classes; %s are missing, so the fields only those classes "
            "supervise carry no loss this run.",
            len(present), len(SAMPLING_CLASSES), ", ".join(missing),
        )

    def no_probe(self) -> None:
        logger.error(
            "The corpus's validation samples hold no records, so there is "
            "nothing to measure feature widths from.",
        )

    def run_start(self, seed: int, samples: ValidationSamples, *, shards: int) -> None:
        logger.info(
            "Seed %d (%s), %d of %d training shards per epoch, drawn across "
            "the corpus.",
            seed,
            "given" if self.config.seed is not None else "drawn; pass "
            f"--seed {seed} to repeat this run",
            min(self.config.shards_per_epoch, shards), shards,
        )
        logger.info("Classes present: %s", ", ".join(sorted(samples.present)))

    def thin_holdout(self, samples: ValidationSamples) -> None:
        """Warn on a thin holdout, raise on an empty one (FR-088b); the gate-1
        slice is the build's count, not this run's."""
        warning = check_holdout(
            card_disjoint_records=len(samples.card_disjoint),
            unique_text_records=self.facts.gate_one_records,
        )
        if warning:
            logger.warning("%s", warning)

    def workers(self, count: int, depth: int) -> None:
        logger.info(
            "Preparing steps in %d worker process(es), up to %d step(s) ahead "
            "each.", count, depth,
        )

    def curriculum(self, epoch: int, fields: int) -> None:
        logger.info(
            "epoch %d enables the sparse field group (%d fields now); "
            "the early stopper starts over", epoch, fields,
        )

    def shard_opened(self, opened: ShardOpened) -> None:
        """Say, once per run, how much of a shard the rarity table covers: a
        table matching nothing otherwise looks exactly like a healthy one."""
        if (
            opened.rarity_coverage is None or not opened.trainable
            or self._rarity_reported
        ):
            return
        self._rarity_reported = True
        found, distinct = opened.rarity_coverage
        share = 100.0 * found / distinct if distinct else 0.0
        report = logger.info if found else logger.warning
        report(
            "Rarity table names %d of this shard's %d distinct ability "
            "text(s) (%.1f%%); the rest weigh by this shard's own game count.",
            found, distinct, share,
        )

    def shard_closed(self, closed: ShardClosed) -> None:
        """What a worker's own caches counted, into this process's reports."""
        self._unresolved_elsewhere.update(closed.unresolved)
        for text, keys in closed.truncated:
            self.truncations.note(text, keys)

    def shard(
        self, task: ShardTask, of: int, opened: ShardOpened | None,
        stats: ShardStats, *, trained: float,
    ) -> None:
        name = Path(task.shard).name
        trainable = opened.trainable if opened is not None else 0
        held_back = opened.held_back if opened is not None else 0
        if not trainable:
            logger.info(
                "epoch %d | shard %d/%d %s | nothing trainable, all %d records "
                "held back | skipped",
                task.epoch, task.position, of, name, held_back,
            )
            return
        logger.info(
            "epoch %d | shard %d/%d %s | %d records trainable, %d held back | "
            "%d steps | loss %.4f | lr %.2e | %.1f steps/s | "
            "wait %.1fs, batch wait %.1fs, train %.1fs%s%s%s",
            task.epoch, task.position, of, name, trainable, held_back, task.budget,
            float(stats.loss) / stats.steps if stats.steps else float("nan"),
            stats.learning_rate, task.budget / trained if trained > 0 else float("nan"),
            stats.waited, stats.batch_waited, trained, format_norms(stats.norms),
            format_parts(stats.parts), format_terms(stats.heads),
        )

    def epoch(
        self, result: EpochResult, metrics: GateOneMetrics,
        card_parts: dict[str, float], floor: dict[str, float], tally: EpochTally,
    ) -> None:
        logger.info(
            "epoch %d | train %.4f | card-disjoint %.4f | "
            "game-disjoint %.4f | gate F1 %.3f | zone acc %.3f | "
            "deviance %.3f%s",
            result.epoch, result.train_loss, result.card_disjoint_loss,
            result.game_disjoint_loss, metrics.affected_gate_f1,
            metrics.zone_outcome_accuracy, metrics.mean_poisson_deviance,
            format_parts(card_parts, floor),
        )
        logger.info(
            "epoch %d | trained records by rarity bucket: %s%s\n"
            "  by rule family: %s",
            result.epoch,
            format_shares(tally.buckets, (*RARITY_BUCKETS, NO_TEXT_BUCKET)),
            format_terms(tally.mean_terms()),
            format_shares(tally.families),
        )

    def unresolved(self, preparer: StepPreparer) -> None:
        """Scripts the converted corpus does not hold, asked for by any process."""
        unresolved = Counter(preparer.sidecars.unresolved) + self._unresolved_elsewhere
        if not unresolved:
            return
        logger.info(
            "%d ability scripts the converted corpus does not hold, "
            "%d lookups so far; those abilities reach the model as no "
            "text. Most asked: %s. A `_token` stem under cardsfolder/ "
            "is a token keyed to the wrong tree — it is the token's own "
            "cast-spell key, which maps to no text either way, while "
            "its ability lines are keyed under tokenscripts/ and do "
            "resolve; a variant-scripts/ key is a variant the tree this "
            "run read has no sidecar for, so the tree is not the one the "
            "corpus was built against.",
            len(unresolved),
            sum(unresolved.values()),
            ", ".join(f"{name} ({hits})" for name, hits in unresolved.most_common(3)),
        )

    def early_stop(self) -> None:
        logger.info(
            "Early stop: %d epochs without a new card-disjoint best",
            self.config.patience,
        )
