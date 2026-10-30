"""Token surfaces and their targets, padded into the trunk's tensors.

Built in numpy and handed to torch once per array: this is every step's host
half, and a per-element torch assignment costs a dispatch per element.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.effect_targets import derive_targets
from effects.domain.entity_field_targets import EntityFieldTargets, EntityTargetBatch


@dataclass(frozen=True, slots=True)
class TrunkInputs:
    """One padded batch of token surfaces, as the trunk reads it.

    Each slot kind's features sit in a ``(batch, slots, width)`` tensor of
    their own, zero at the slots of other kinds. ``e_rows`` names the row of
    the batch's ``e`` matrix each slot reads (``-1`` where its ``e`` is in
    ``e_vectors`` already); :meth:`with_e` resolves it on the device. Without
    ``option_kinds`` every row reads as an ordinary one.
    """

    slot_features: dict[SlotKind, torch.Tensor]
    slot_kinds: torch.Tensor
    positions: torch.Tensor
    e_vectors: torch.Tensor
    attention_mask: torch.Tensor
    option_kinds: torch.Tensor | None = None
    e_rows: torch.Tensor | None = None

    def _mapped(self, move) -> TrunkInputs:
        def moved(tensor):
            return None if tensor is None else move(tensor)

        return TrunkInputs(
            slot_features={kind: move(tensor) for kind, tensor in self.slot_features.items()},
            slot_kinds=move(self.slot_kinds),
            positions=move(self.positions),
            e_vectors=move(self.e_vectors),
            attention_mask=move(self.attention_mask),
            option_kinds=moved(self.option_kinds),
            e_rows=moved(self.e_rows),
        )

    def to(self, device, *, non_blocking: bool = False) -> TrunkInputs:
        return self._mapped(lambda tensor: tensor.to(device, non_blocking=non_blocking))

    def pin(self) -> TrunkInputs:
        """Every tensor copied to page-locked host memory, so a
        ``non_blocking`` copy does not hold the host."""
        return self._mapped(torch.Tensor.pin_memory)

    def with_e(self, matrix: torch.Tensor | None) -> TrunkInputs:
        """The slots that named a row of ``matrix`` filled with it
        (:func:`scatter_e_rows`), and ``e_rows`` spent; a batch with no
        matrix keeps the ``e_vectors`` it was collated with."""
        if matrix is None:
            return replace(self, e_rows=None)
        return replace(
            self, e_vectors=scatter_e_rows(self.e_vectors, self.e_rows, matrix), e_rows=None,
        )


def gather_entities(outputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``(batch, entities, width)``: each surface's ``[CARD]`` and ``[PLAYER]``
    columns of the trunk output, in one batched gather."""
    return outputs.gather(1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))


def entity_index(surfaces: list) -> torch.Tensor:
    """``(batch, entities)``: the column of each surface's ``entity_slots``, in
    their order, zero past a surface's last (a padding column no reader takes)."""
    width = max((len(s.entity_slots) for s in surfaces), default=1)
    index = np.zeros((len(surfaces), width), dtype=np.int64)
    for row, surface in enumerate(surfaces):
        index[row, : len(surface.entity_slots)] = surface.entity_slots
    return torch.from_numpy(index)


def collate_surfaces(
    surfaces: list, *, e_dim: int, widths: dict[SlotKind, int],
) -> TrunkInputs:
    """Pad a batch of token surfaces, to its own longest, into the trunk's
    inputs, each slot kind's features in a tensor of its own.

    A slot whose ``e`` is an ``int`` row lands in ``e_rows`` (``-1``
    elsewhere), for the caller to resolve on the device with
    :meth:`TrunkInputs.with_e`.
    """
    batch = len(surfaces)
    width = max((len(s.slots) for s in surfaces), default=1)
    features = {
        kind: np.zeros((batch, width, size), dtype=np.float32)
        for kind, size in widths.items()
    }
    slot_kinds = np.zeros((batch, width), dtype=np.int64)
    positions = np.zeros((batch, width), dtype=np.int64)
    e_vectors = np.zeros((batch, width, e_dim), dtype=np.float32)
    e_rows = np.full((batch, width), -1, dtype=np.int64)
    attention = np.zeros((batch, width), dtype=np.int64)
    option_kinds = np.zeros((batch, width), dtype=np.int64)

    for row, surface in enumerate(surfaces):
        slots = surface.slots
        filled = len(slots)
        slot_kinds[row, :filled] = [int(slot.kind) for slot in slots]
        positions[row, :filled] = [slot.position for slot in slots]
        attention[row, :filled] = 1
        option_kinds[row, :filled] = [int(slot.option) for slot in slots]
        for column, slot in enumerate(slots):
            if slot.features and slot.kind in features:
                values = slot.features
                features[slot.kind][row, column, : len(values)] = values
            if isinstance(slot.e, int):
                e_rows[row, column] = slot.e
            elif slot.e is not None:
                e_vectors[row, column, : len(slot.e)] = slot.e

    return TrunkInputs(
        slot_features={
            kind: torch.from_numpy(array) for kind, array in features.items()
        },
        slot_kinds=torch.from_numpy(slot_kinds),
        positions=torch.from_numpy(positions),
        e_vectors=torch.from_numpy(e_vectors),
        attention_mask=torch.from_numpy(attention),
        option_kinds=torch.from_numpy(option_kinds),
        e_rows=torch.from_numpy(e_rows),
    )


def scatter_e_rows(
    e_vectors: torch.Tensor, e_rows: torch.Tensor, matrix: torch.Tensor,
) -> torch.Tensor:
    """Fill the slots that named a row of ``matrix`` with that row, in one
    gather on the matrix's device.

    The effect head's loss reaches the encoder only through this gather; a
    host round trip per slot would sever the gradient and train the head
    against a constant.
    """
    valid = e_rows >= 0
    gathered = matrix[e_rows.clamp(min=0)]
    return torch.where(valid.unsqueeze(-1), gathered, e_vectors)


def entity_target_tensors(
    surfaces: list, targets_per_surface: list[dict], fields: tuple[FieldSpec, ...],
) -> EntityTargetBatch:
    """The batch's per-entity targets.

    ``index`` selects the ``[CARD]`` and ``[PLAYER]`` columns of the trunk
    output; an entity no slot represents is not gathered. A field's value
    where it is not present is a zero the loss never reads.
    """
    batch = len(surfaces)
    width = max((len(s.entity_slots) for s in surfaces), default=1)
    gate = np.zeros((batch, width), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=np.float32)
    collected: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}

    for spec in fields:
        shape = (batch, width, spec.arity) if spec.type.per_member else (batch, width)
        collected[spec.name] = np.zeros(shape, dtype=np.float32)
        present[spec.name] = np.zeros((batch, width), dtype=np.bool_)

    for row, (surface, targets) in enumerate(zip(surfaces, targets_per_surface)):
        for column, slot_index in enumerate(surface.entity_slots):
            mask[row, column] = 1
            entry = targets.get(surface.slots[slot_index].subject_id)
            if entry is None:
                continue
            gate[row, column] = 1.0 if entry.affected else 0.0
            for spec in fields:
                value = entry.supervised(spec)
                if value is None:
                    continue
                present[spec.name][row, column] = True
                if isinstance(value, list):
                    collected[spec.name][row, column, : len(value)] = value
                else:
                    collected[spec.name][row, column] = float(value)
    return EntityTargetBatch(
        torch.from_numpy(gate),
        EntityFieldTargets(
            {name: torch.from_numpy(values) for name, values in collected.items()},
            {name: torch.from_numpy(rows) for name, rows in present.items()},
        ),
        torch.from_numpy(mask),
        entity_index(surfaces),
    )


def per_entity_targets(
    records: list, surfaces: list, fields: tuple[FieldSpec, ...],
) -> EntityTargetBatch:
    """The per-entity targets of ``records``, built into ``surfaces``."""
    return entity_target_tensors(
        surfaces, [derive_targets(record) for record in records], fields,
    )
