"""One chunked forward over records: the device half every scorer runs
(research E2).

Validation's loss, gate 1's margins, the evaluation's breakdowns, the knowledge
probes and the control score all run the same few steps over a list of
records — the host half of a chunk (``SurfaceBatcher.prepare``, or a scoring
step a cache prepared once), its texts encoded and scattered
(``Materializer.materialize``), the trunk, and one head or another — with both
modules in evaluation mode, no gradient, and the bf16 autocast validation
runs under on CUDA. Running them here, one way, keeps the number a
checkpoint was selected on the very number it is reported on: selection and
reporting read their halves through one forward at one precision, where a
float32 report against a bf16 selection would make the halves incomparable.

The autocast is ``step_loss.autocast_enabled``'s, the one validation's loss
runs under, so it is off on the CPU and on a card without bf16.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING

import torch

from effects.application.step_loss import autocast_enabled
from effects.domain.surface_collation import entity_index, per_entity_targets

if TYPE_CHECKING:
    from effects.application.step_loss import StepLoss, StepLossComputer
    from effects.application.step_preparation import DeviceStep
    from effects.application.surface_batching import (
        EncodedBatch,
        Materializer,
        PreparedBatch,
        SurfaceBatcher,
    )
    from effects.domain.effect_head_layout import FieldSpec
    from effects.domain.entity_field_targets import EntityTargetBatch

#: Records per forward pass wherever a caller names no size: gate 1, the
#: breakdowns and the probes.
BATCH_RECORDS = 32


@dataclass(frozen=True)
class ForwardOutputs:
    """One chunk's trunk output, and its heads read on demand, in float32.

    Read inside the forward's :meth:`RecordForward.running` block, so each
    head runs under the same autocast as the trunk.
    """

    model: object
    hidden: torch.Tensor
    encoded: EncodedBatch

    def per_entity(self) -> torch.Tensor:
        """``(records, slots, width)``: the per-entity head at every slot."""
        return self.model.per_entity(self.hidden).float()

    def per_entity_at(self, index: torch.Tensor) -> torch.Tensor:
        """``(records, columns, width)``: the per-entity head at the slots
        ``index`` names, one row of slot indices per record."""
        return self.model.per_entity_at(self.hidden, index.to(self.hidden.device)).float()

    def verdict(self) -> torch.Tensor:
        """``(records, width)``: the verdict head at ``[ACT]``."""
        return self.model.verdict(self.hidden).float()

    def global_outputs(self) -> torch.Tensor:
        """``(records, width)``: the global head at ``[GLOBAL]``."""
        return self.model.global_outputs(self.hidden).float()

    def created_objects(self) -> torch.Tensor:
        """``(records, width)``: the created-objects head."""
        return self.model.created_objects(self.hidden).float()


@dataclass(frozen=True)
class RecordChunk:
    """One chunk of records, the surfaces built for them and their outputs."""

    records: list
    surfaces: list
    outputs: ForwardOutputs

    def at_entities(self) -> torch.Tensor:
        """The per-entity head at each surface's ``entity_slots``, in their
        order, on the host: ``(records, entities, width)``."""
        return self.outputs.per_entity_at(entity_index(self.surfaces)).cpu()


class RecordForward:
    """The two modules and the device half of a batch, run over chunks.

    ``materializer`` turns a chunk's staged texts into ``e`` on its device;
    a scoring materializer (no noise on ``e``) is the only kind a forward
    here should be handed.
    """

    def __init__(self, encoder, model, materializer: Materializer | None) -> None:
        self.encoder = encoder
        self.model = model
        self.materializer = materializer

    @property
    def autocast(self) -> bool:
        """Whether the forward runs under bf16 autocast: validation's rule."""
        return self.materializer is not None and autocast_enabled(self.materializer.device)

    def _autocast(self):
        if not self.autocast:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @contextmanager
    def running(self) -> Iterator[None]:
        """Both modules in evaluation mode, no gradient and the autocast for
        the block; a module that was training is put back in training mode
        however the block ends, since every training step assumes it."""
        modules = (self.encoder, self.model)
        training = [getattr(module, "training", True) for module in modules]
        for module in modules:
            module.eval()
        try:
            with torch.no_grad(), self._autocast():
                yield
        finally:
            for module, was in zip(modules, training):
                if was:
                    module.train()

    def forward(self, prepared: PreparedBatch) -> ForwardOutputs:
        """One prepared chunk through the encoder and the trunk; call inside
        :meth:`running`."""
        inputs, _surfaces, encoded = self.materializer.materialize(prepared, self.encoder)
        return ForwardOutputs(self.model, self.model(inputs), encoded)

    def records(
        self, records: Iterable, batcher: SurfaceBatcher, *, size: int = BATCH_RECORDS,
        strips: Sequence | None = None,
    ) -> Iterator[RecordChunk]:
        """``records`` ``size`` at a time, each chunk's host half prepared here
        by ``batcher`` (``strips``, one per record, are gate 2's perturbation)
        and run under :meth:`running`; read each chunk's outputs before the
        next is asked for."""
        offset = 0
        with self.running():
            for chunk in batched(records, size):
                chunk = list(chunk)
                strip = None if strips is None else list(strips[offset:offset + len(chunk)])
                offset += len(chunk)
                prepared = batcher.prepare(chunk, strip)
                yield RecordChunk(chunk, prepared.surfaces, self.forward(prepared))

    def prepared(self, batches: Iterable[PreparedBatch]) -> Iterator[ForwardOutputs]:
        """Batches a cache prepared ahead, each run under :meth:`running`."""
        with self.running():
            for prepared in batches:
                yield self.forward(prepared)

    def steps(
        self, steps: Iterable[DeviceStep], losses: StepLossComputer, *,
        report_parts: bool = False,
    ) -> Iterator[tuple[DeviceStep, StepLoss]]:
        """Each scoring step and its losses, as validation scores them: the
        loss computer's own device half (which carries the autocast) under
        evaluation mode and no gradient."""
        with self.running():
            for step in steps:
                yield step, losses.device_loss(
                    step, self, self.materializer, report_parts=report_parts,
                )


def entity_chunks(
    records, forward: RecordForward, batcher: SurfaceBatcher,
    fields: tuple[FieldSpec, ...], *, size: int = BATCH_RECORDS,
) -> Iterator[tuple[list, torch.Tensor, EntityTargetBatch]]:
    """``(chunk, gathered, targets)`` per chunk of ``records``, on the host,
    ``gathered`` aligned column for column with the unmasked targets: gate 1's
    margins and the evaluation's breakdowns read one pass."""
    for chunk in forward.records(records, batcher, size=size):
        yield (chunk.records, chunk.at_entities(),
               per_entity_targets(chunk.records, chunk.surfaces, fields))


__all__ = [
    "BATCH_RECORDS", "ForwardOutputs", "RecordChunk", "RecordForward", "entity_chunks",
]
