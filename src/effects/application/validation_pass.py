"""Validation: both strata of the corpus's fixed samples, scored every epoch,
and the control score the run selects its checkpoint by.

The samples are read once and never redrawn (FR-089), so an epoch's number
moves only with the model. Their losses, the gate-1 progress margins and the
constant-predictor floor are reported; the stratum scores and the control
score that choose the best epoch (FR-068, FR-069) are the control score's
(``control_score``), whose cache :meth:`Validator.prepare` builds once at
startup from the corpus's frozen selection set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

from effects.application.record_forward import RecordForward
from effects.application.step_loss import StepLoss, StepLossComputer
from effects.application.step_preparation import DeviceStep
from effects.application.train_effect_model import (
    MissingTrainingInputError,
    TrainEffectModelConfig,
)
from effects.application.training_log import format_floor
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.gate_metrics import GateOneMetrics, MarginAccumulator
from effects.domain.loss_floor import PartsMean, TargetSample, constant_predictor_floor
from effects.domain.sampling import sampling_class
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.infrastructure.shard_io import load_shard

if TYPE_CHECKING:
    from effects.application.control_score import ControlScore, ControlScorer, FrozenSelection
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


class PassCollector(Protocol):
    """What one validation pass gathers beside its loss, batch by batch."""

    #: Whether the pass has to keep each field's term for this collector.
    wants_parts: bool

    def add(self, prepared: DeviceStep, computed: StepLoss) -> None:
        """Fold in one batch, as prepared and as scored."""

    def finish(self) -> None:
        """The pass is over: fold in anything still in flight."""


class FieldParts:
    """The loss by field, per record over the pass's records."""

    wants_parts = True

    def __init__(self) -> None:
        self._means = PartsMean()

    def add(self, prepared: DeviceStep, computed: StepLoss) -> None:
        self._means.add(computed.parts, rows=int(prepared.targets.gate.shape[0]))

    def finish(self) -> None:
        pass

    def means(self) -> dict[str, float]:
        return self._means.means()


class KeptTargets:
    """The sample's supervised targets, for the constant-predictor floor:
    kept only on an epoch with a floor to compute."""

    wants_parts = False

    def __init__(self) -> None:
        self.sample = TargetSample()

    def add(self, prepared: DeviceStep, computed: StepLoss) -> None:
        self.sample.add(prepared.targets, prepared.heads)

    def finish(self) -> None:
        pass


class ProgressMargins:
    """Gate 1's margins over the pass's outputs. Each batch's copy to the host
    is folded in a batch later, so the device never waits on it."""

    wants_parts = False

    def __init__(self, fields: tuple[FieldSpec, ...], log_dispersion: torch.Tensor) -> None:
        self.margins = MarginAccumulator(fields, log_dispersion)
        self._pending: _HostCopy | None = None

    def add(self, prepared: DeviceStep, computed: StepLoss) -> None:
        copy = _HostCopy.queue(computed.gathered, prepared)
        self.finish()
        self._pending = copy

    def finish(self) -> None:
        if self._pending is not None:
            self._pending.fold_into(self.margins)
            self._pending = None


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
    #: The loss by field, averaged over batches; empty unless collected.
    parts: dict[str, float]
    #: Gate 1's margins over the pass's outputs; None unless collected.
    margins: MarginAccumulator | None

    def progress(self) -> GateOneMetrics:
        """Gate 1's three numbers on the whole card-disjoint sample, every
        epoch.

        A progress reading, not the gate: the evaluator scores only the
        unique-text stratum, where this pass reads the whole sample.
        """
        assert self.margins is not None
        return self.margins.metrics()


class EpochScores(NamedTuple):
    """One epoch's validation: both strata's reported passes, and the control
    score the epoch is selected by."""

    card: ValidationPass
    game: ValidationPass
    control: ControlScore
    #: The control score's wall-clock seconds this epoch (research E2).
    control_seconds: float = 0.0


class Validator:
    """Scores the validation samples with the run's scoring device half, keeps
    the constant-predictor floor the epoch line compares against, and hands
    each stratum's score to the control score (FR-068)."""

    def __init__(
        self, config: TrainEffectModelConfig, device: torch.device,
        losses: StepLossComputer,
    ) -> None:
        self.config = config
        self.batch_size = config.batch_size
        self.device = device
        #: The device half validation shares with training, so both score
        #: one objective.
        self.losses = losses
        self.floor = FloorCache()
        #: The control score's cache; :meth:`prepare` builds it.
        self.scorer: ControlScorer | None = None

    def prepare(
        self, run: RunContext, selection: FrozenSelection | None, *,
        rule_outputs: Mapping[str, tuple[str, ...]],
    ) -> None:
        """Build the control score's cache from the corpus's frozen selection
        set, once, before the first epoch (research E2): every field set the
        run's epochs score, the run's rule mask, the grids edited on the run's
        surface. Raises ``ControlScoreError`` on a component with no item."""
        from effects.application.control_score import ControlScorer
        from effects.domain.grid_edits import GridEdits

        if selection is None:
            raise MissingTrainingInputError(
                "No selection set: train-effect-model selects its checkpoint by "
                "the control score of the corpus's frozen selection set (FR-069).")
        self.scorer = ControlScorer.prepare(
            selection.selection, selection.strata, preparer=run.preparer,
            fields=list(run.fields.values()),
            rule_outputs=rule_outputs, mask=str(self.config.rule_mask),
            stratum_weight=self.config.stratum_weight,
            deviance_floor=self.config.deviance_floor,
            editor=GridEdits(run.preparer.scoring_batcher()),
        )

    def control(self, run: RunContext, fields: tuple[FieldSpec, ...]) -> ControlScore:
        """The epoch's control score on the weights the modules hold, through
        the one chunked forward."""
        if self.scorer is None:
            raise RuntimeError("the control score was never prepared")
        return self.scorer.epoch(
            RecordForward(run.encoder, run.model, run.scoring), self.losses, fields)

    def epoch(
        self, run: RunContext, samples: ValidationSamples,
        fields: tuple[FieldSpec, ...],
    ) -> EpochScores:
        """Both strata's passes, the card-disjoint one by field, with gate 1's
        progress margins, and the floor recomputed when the field set moved;
        then the control score."""
        parts = FieldParts()
        margins = ProgressMargins(fields, run.model.per_entity_dispersion().detach())
        kept = KeptTargets() if self.floor.stale(fields) else None
        collectors = (parts, margins) if kept is None else (parts, margins, kept)
        card = ValidationPass(
            self.score(samples.card_disjoint, run, fields, collectors),
            parts.means(), margins.margins,
        )
        game = ValidationPass(self.score(samples.game_disjoint, run, fields), {}, None)
        if kept is not None:
            self.floor.update(
                fields, constant_predictor_floor(kept.sample, fields=fields),
            )
            logger.info(
                "Constant-predictor floor (card-disjoint): %s",
                format_floor(self.floor.floor),
            )
        started = time.perf_counter()
        control = self.control(run, fields)
        return EpochScores(card, game, control, time.perf_counter() - started)

    def score(
        self, records, run: RunContext, fields: tuple[FieldSpec, ...],
        collectors: tuple[PassCollector, ...] = (),
    ) -> float:
        """Mean loss over the stratum, in fixed-size batches, not one per game:
        one game's outlier would move a per-game mean many times over; NaN for
        an empty stratum.

        ``fields`` is the epoch's set, so validation and training score one
        objective. Each of ``collectors`` gathers its own number off the same
        batches, so none needs a second pass. A batch keeps the sample's
        order: its texts are encoded once whichever records share them
        (``SurfaceBatcher.batch_texts``), so grouping by game would save
        nothing.
        """
        if not records:
            return float("nan")
        size = self.batch_size
        prepared = (
            run.preparer.prepare_scoring_step(records[start:start + size], fields)
            for start in range(0, len(records), size)
        )
        return score_steps(
            prepared, RecordForward(run.encoder, run.model, run.scoring), self.losses,
            collectors, device=self.device,
        )


def score_steps(
    steps: Iterable[DeviceStep], forward: RecordForward, losses: StepLossComputer,
    collectors: tuple[PassCollector, ...] = (), *, device: torch.device,
) -> float:
    """The mean shipped loss over ``steps`` through the one chunked forward
    (:mod:`effects.application.record_forward`), each of ``collectors`` folding
    in every step as prepared and scored; NaN for no step.

    The steps may come prepared one by one, as validation's do, or from a
    cache prepared once, as the control score's deviance sample does.
    """
    report_parts = any(collector.wants_parts for collector in collectors)
    total = torch.zeros((), device=device)
    batches = 0
    for prepared, computed in forward.steps(steps, losses, report_parts=report_parts):
        total += computed.shipped
        batches += 1
        for collector in collectors:
            collector.add(prepared, computed)
    for collector in collectors:
        collector.finish()
    return float(total) / batches if batches else float("nan")
