"""Validation: both strata of the corpus's fixed samples, scored every epoch.

The samples are read once and never redrawn (FR-089), so an epoch's number
moves only with the model.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from effects.application.step_loss import StepLossComputer, TrainedModules
from effects.application.step_preparation import DeviceStep
from effects.application.train_effect_model import TrainEffectModelConfig, load_shard
from effects.application.training_log import format_floor
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.gate_metrics import GateOneMetrics, MarginAccumulator
from effects.domain.loss_floor import PartsMean, constant_predictor_floor
from effects.domain.sampling import BatchPlan, sampling_class
from effects.domain.strata import VALIDATION_STRATA, Stratum

if TYPE_CHECKING:
    from effects.application.run_context import RunContext

logger = logging.getLogger(__name__)

#: Records kept aside to measure feature widths from, before any shard loads.
PROBE_RECORDS = 64


@dataclass(frozen=True)
class ValidationSamples:
    """The corpus's fixed validation samples, one per stratum, read once and
    never redrawn (FR-089)."""

    card_disjoint: list
    game_disjoint: list
    #: Records to measure feature widths from, before any shard loads.
    probe: list
    #: The sampling classes the two samples hold; every epoch's field set is
    #: built from these.
    present: frozenset[str]

    @classmethod
    def load(cls, paths: Mapping[str, Path]) -> ValidationSamples:
        logger.info(
            "Reading the corpus's validation samples: %s.",
            ", ".join(str(path) for path in paths.values()),
        )
        card_disjoint = load_shard(paths[Stratum.CARD_DISJOINT])
        game_disjoint = load_shard(paths[Stratum.GAME_DISJOINT])
        for stratum, records in zip(VALIDATION_STRATA, (card_disjoint, game_disjoint)):
            logger.info(
                "%s validation: %d records over %d games, %d classes",
                stratum, len(records), len({r.game_id for r in records}),
                len({sampling_class(r) for r in records}),
            )
        return cls(
            card_disjoint=card_disjoint,
            game_disjoint=game_disjoint,
            probe=card_disjoint[:PROBE_RECORDS] or game_disjoint[:PROBE_RECORDS],
            present=frozenset(
                sampling_class(record) for record in (*card_disjoint, *game_disjoint)
            ),
        )


@contextmanager
def evaluating(modules: TrainedModules) -> Iterator[None]:
    """Both modules in evaluation mode for the block, back in training mode
    however it ends: every training step assumes training mode."""
    modules.encoder.eval()
    modules.model.eval()
    try:
        yield
    finally:
        modules.encoder.train()
        modules.model.train()


@dataclass(frozen=True, slots=True)
class _HostCopy:
    """One validation batch's gathered outputs on their way to the host,
    queued without waiting and read a batch later so the device never idles."""

    gathered: torch.Tensor
    done: torch.cuda.Event | None
    prepared: DeviceStep

    @classmethod
    def queue(cls, gathered: torch.Tensor, prepared: DeviceStep) -> _HostCopy:
        host = gathered.float().to("cpu", non_blocking=True)
        done = None
        if gathered.device.type == "cuda":
            done = torch.cuda.Event()
            done.record()
        return cls(host, done, prepared)

    def fold_into(self, margins: MarginAccumulator) -> None:
        if self.done is not None:
            self.done.synchronize()
        margins.add(self.gathered, self.prepared.targets)


class FloorCache:
    """The constant-predictor floor, keyed by the active field set.

    The validation targets never move (FR-089), so the floor changes only when
    the field set does, at the curriculum epoch.
    """

    def __init__(self) -> None:
        self._fields: tuple[FieldSpec, ...] | None = None
        self.floor: dict[str, float] = {}

    def stale(self, fields: tuple[FieldSpec, ...]) -> bool:
        return self._fields != fields

    def update(
        self, fields: tuple[FieldSpec, ...], floor: Mapping[str, float],
    ) -> None:
        self._fields = fields
        self.floor = dict(floor)


@dataclass(frozen=True)
class ValidationPass:
    """One stratum's numbers from one pass over its sample."""

    #: The mean shipped loss over the pass's batches; NaN for an empty sample.
    loss: float
    #: The loss by field, averaged over batches; empty unless asked for.
    parts: dict[str, float]
    #: Gate 1's margins over the pass's outputs; None unless asked for.
    margins: MarginAccumulator | None
    #: Each batch's host targets, for the floor; None unless asked for.
    targets: list[EntityTargetBatch] | None

    def progress(self) -> GateOneMetrics:
        """Gate 1's three numbers on the whole card-disjoint sample, every
        epoch.

        A progress reading, not the gate: the evaluator scores only the
        unique-text stratum, and in float32 where this pass runs under bf16
        autocast, so a logit near zero can land on the other side here.
        """
        assert self.margins is not None
        return self.margins.metrics()


class Validator:
    """Scores the validation samples with the run's scoring device half, and
    keeps the constant-predictor floor the epoch line compares against."""

    def __init__(
        self, config: TrainEffectModelConfig, device: torch.device,
        losses: StepLossComputer,
    ) -> None:
        self.batch_size = config.batch_size
        self.device = device
        #: The device half validation shares with training, so both score
        #: one objective.
        self.losses = losses
        self.floor = FloorCache()

    def epoch(
        self, run: RunContext, samples: ValidationSamples,
        fields: tuple[FieldSpec, ...],
    ) -> tuple[ValidationPass, ValidationPass]:
        """Both strata's passes: the card-disjoint one by field, with gate 1's
        progress margins, and the floor recomputed when the field set moved."""
        card = self.score(
            samples.card_disjoint, run, fields, parts=True, margins=True,
            # Collected only on an epoch with a floor to compute: a copy of
            # the whole sample's targets.
            targets=self.floor.stale(fields),
        )
        game = self.score(samples.game_disjoint, run, fields)
        if card.targets is not None:
            self.floor.update(
                fields, constant_predictor_floor(card.targets, fields=fields),
            )
            logger.info(
                "Constant-predictor floor (card-disjoint): %s",
                format_floor(self.floor.floor),
            )
        return card, game

    def score(
        self, records, run: RunContext, fields: tuple[FieldSpec, ...], *,
        parts: bool = False, margins: bool = False, targets: bool = False,
    ) -> ValidationPass:
        """Mean loss over the stratum, in fixed-size batches, not one per game:
        one game's outlier would move a per-game mean many times over.

        ``fields`` is the epoch's set, so validation and training score one
        objective. ``parts`` keeps the loss by field; ``targets`` each batch's
        host targets for the floor and ``margins`` its outputs for gate 1's
        progress margins, so neither needs a second pass. Records are grouped
        by game within a batch, so one encode of a text serves the whole game.
        """
        kept: list[EntityTargetBatch] | None = [] if targets else None
        accumulator = MarginAccumulator(fields) if margins else None
        if not records:
            return ValidationPass(float("nan"), {}, accumulator, kept)
        size = self.batch_size
        total = torch.zeros((), device=self.device)
        batches = 0
        means = PartsMean()
        pending: _HostCopy | None = None
        with evaluating(run), torch.no_grad():
            for start in range(0, len(records), size):
                grouped: dict[str, list] = defaultdict(list)
                for record in records[start:start + size]:
                    grouped[record.game_id].append(record)
                prepared = run.preparer.prepare_scoring_step(
                    BatchPlan(dict(grouped)), fields,
                )
                if kept is not None:
                    kept.append(prepared.targets)
                computed = self.losses.device_loss(
                    prepared, run, run.scoring, report_parts=parts,
                )
                total += computed.shipped
                batches += 1
                means.add(computed.parts)
                if accumulator is not None:
                    copy = _HostCopy.queue(computed.gathered, prepared)
                    if pending is not None:
                        pending.fold_into(accumulator)
                    pending = copy
            if pending is not None:
                pending.fold_into(accumulator)
        return ValidationPass(
            float(total) / batches if batches else float("nan"),
            means.means() if parts else {}, accumulator, kept,
        )
