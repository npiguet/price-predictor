"""Token surfaces and their targets, padded into the trunk's tensors.

Built in numpy and handed to torch once per array: this is every step's host
half, and a per-element torch assignment costs a dispatch per element.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from effects.domain.effect_head_layout import FieldSpec
from effects.domain.effect_targets import derive_targets
from effects.domain.entity_field_targets import EntityFieldTargets, EntityTargetBatch
from effects.domain.input_layout import SlotKind
from effects.domain.input_vocabularies import ID_PADDING
from effects.domain.rule_mask import CopyWeights


@dataclass(frozen=True, slots=True)
class TrunkInputs:
    """One padded batch of token surfaces, as the trunk reads it.

    Each slot kind's features sit in a ``(batch, slots, width)`` tensor of
    their own, zero at the slots of other kinds. ``e_rows`` names the row of
    the batch's ``e`` matrix each slot reads (``-1`` where its ``e`` is in
    ``e_vectors`` already); :meth:`with_e` resolves it on the device.
    ``option_kinds`` marks a charm mode's ``option`` rows.
    """

    slot_features: dict[SlotKind, torch.Tensor]
    slot_kinds: torch.Tensor
    positions: torch.Tensor
    e_vectors: torch.Tensor
    attention_mask: torch.Tensor
    option_kinds: torch.Tensor
    e_rows: torch.Tensor | None = None
    #: ``(N, 4)`` long ``[batch row, from slot, to slot, relation type]``; None
    #: or empty where the batch holds no relation (root spec § 10.1).
    relations: torch.Tensor | None = None
    #: ``(batch, slots, k)`` long rows of the subtype table, padded with
    #: ``ID_PADDING``: each ``[CARD]`` slot's subtypes, its chosen types and its
    #: real face's subtypes.
    subtype_ids: torch.Tensor | None = None
    chosen_type_ids: torch.Tensor | None = None
    #: The same, for a face-down ``[CARD]`` slot's real face's subtypes.
    real_face_subtype_ids: torch.Tensor | None = None
    #: ``(batch, slots, k)`` rows of the named-counter table, padded with
    #: ``ID_PADDING``, and the count beside each.
    counter_ids: torch.Tensor | None = None
    counter_counts: torch.Tensor | None = None

    def _mapped(self, move) -> TrunkInputs:
        def moved(tensor):
            return None if tensor is None else move(tensor)

        return TrunkInputs(
            slot_features={kind: move(tensor) for kind, tensor in self.slot_features.items()},
            slot_kinds=move(self.slot_kinds),
            positions=move(self.positions),
            e_vectors=move(self.e_vectors),
            attention_mask=move(self.attention_mask),
            option_kinds=move(self.option_kinds),
            e_rows=moved(self.e_rows),
            relations=moved(self.relations),
            subtype_ids=moved(self.subtype_ids),
            chosen_type_ids=moved(self.chosen_type_ids),
            real_face_subtype_ids=moved(self.real_face_subtype_ids),
            counter_ids=moved(self.counter_ids),
            counter_counts=moved(self.counter_counts),
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

    A slot's ``e_row`` lands in ``e_rows`` (``-1`` elsewhere), for the
    caller to resolve on the device with
    :meth:`TrunkInputs.with_e`.
    """
    width = max((len(s.slots) for s in surfaces), default=1)
    arrays = _SlotArrays.zeros((len(surfaces), width), e_dim=e_dim, widths=widths)
    relations = _relations(surfaces)
    id_lists = _id_lists(surfaces, width)
    for row, surface in enumerate(surfaces):
        arrays.fill(row, surface)
    return arrays.inputs(relations, id_lists)


@dataclass(frozen=True, slots=True)
class _SlotArrays:
    """One batch's per-slot arrays, ``(batch, width)`` or ``(batch, width,
    size)``, filled a surface at a time and handed to torch once each."""

    features: dict[SlotKind, np.ndarray]
    slot_kinds: np.ndarray
    positions: np.ndarray
    e_vectors: np.ndarray
    e_rows: np.ndarray
    attention: np.ndarray
    option_kinds: np.ndarray

    @classmethod
    def zeros(
        cls, shape: tuple[int, int], *, e_dim: int, widths: dict[SlotKind, int],
    ) -> _SlotArrays:
        """Blank arrays of ``shape``: zero, and ``-1`` in ``e_rows``."""
        return cls(
            features={
                kind: np.zeros((*shape, size), dtype=np.float32)
                for kind, size in widths.items()
            },
            slot_kinds=np.zeros(shape, dtype=np.int64),
            positions=np.zeros(shape, dtype=np.int64),
            e_vectors=np.zeros((*shape, e_dim), dtype=np.float32),
            e_rows=np.full(shape, -1, dtype=np.int64),
            attention=np.zeros(shape, dtype=np.int64),
            option_kinds=np.zeros(shape, dtype=np.int64),
        )

    def fill(self, row: int, surface) -> None:
        """``surface``'s slots, written along ``row``."""
        slots = surface.slots
        filled = len(slots)
        self.slot_kinds[row, :filled] = [int(slot.kind) for slot in slots]
        self.positions[row, :filled] = [slot.position for slot in slots]
        self.attention[row, :filled] = 1
        self.option_kinds[row, :filled] = [int(slot.option) for slot in slots]
        self._fill_columns(row, slots)

    def _fill_columns(self, row: int, slots) -> None:
        """Each slot's features and its ``e_row`` or ``e_vector``."""
        features, e_rows, e_vectors = self.features, self.e_rows, self.e_vectors
        for column, slot in enumerate(slots):
            if slot.features and slot.kind in features:
                values = slot.features
                features[slot.kind][row, column, : len(values)] = values
            if slot.e_row is not None:
                e_rows[row, column] = slot.e_row
            elif slot.e_vector is not None:
                e_vectors[row, column, : len(slot.e_vector)] = slot.e_vector

    def inputs(self, relations: np.ndarray, id_lists: dict[str, np.ndarray]) -> TrunkInputs:
        """The arrays, the relations and the id lists as the trunk's inputs."""
        return TrunkInputs(
            slot_features={
                kind: torch.from_numpy(array) for kind, array in self.features.items()
            },
            slot_kinds=torch.from_numpy(self.slot_kinds),
            positions=torch.from_numpy(self.positions),
            e_vectors=torch.from_numpy(self.e_vectors),
            attention_mask=torch.from_numpy(self.attention),
            option_kinds=torch.from_numpy(self.option_kinds),
            e_rows=torch.from_numpy(self.e_rows),
            relations=torch.from_numpy(relations),
            **{name: torch.from_numpy(array) for name, array in id_lists.items()},
        )


def _relations(surfaces: list) -> np.ndarray:
    """``(N, 4)`` long ``[batch row, from slot, to slot, relation type]``."""
    return np.array(
        [(row, i, j, r) for row, surface in enumerate(surfaces)
         for i, j, r in surface.relations],
        dtype=np.int64).reshape(-1, 4)


#: The slot fields that are id lists, as ``(TrunkInputs field, Slot field)``.
_ID_LISTS = (
    ("subtype_ids", "subtypes"),
    ("chosen_type_ids", "chosen_types"),
    ("real_face_subtype_ids", "real_face_subtypes"),
)


def _id_lists(surfaces: list, width: int) -> dict[str, np.ndarray]:
    """Each ``[CARD]`` slot's id lists and its counters, padded to one depth:
    ``(batch, width, depth)`` arrays keyed by their :class:`TrunkInputs` field."""
    slots = [(row, column, slot) for row, surface in enumerate(surfaces)
             for column, slot in enumerate(surface.slots)]
    depth = max((len(getattr(slot, field, ())) for _, _, slot in slots
                 for field in (*(field for _, field in _ID_LISTS), "counters")), default=0)
    shape = (len(surfaces), width, max(depth, 1))
    arrays = {name: np.full(shape, ID_PADDING, dtype=np.int64) for name, _ in _ID_LISTS}
    arrays["counter_ids"] = np.full(shape, ID_PADDING, dtype=np.int64)
    arrays["counter_counts"] = np.zeros(shape, dtype=np.float32)
    for row, column, slot in slots:
        _fill_slot(arrays, row, column, slot)
    return arrays


def _fill_slot(arrays: dict[str, np.ndarray], row: int, column: int, slot) -> None:
    """One slot's id lists and counters, written at its ``(row, column)``."""
    for name, field in _ID_LISTS:
        ids = getattr(slot, field, ())
        arrays[name][row, column, : len(ids)] = ids
    for k, (counter, count) in enumerate(slot.counters):
        arrays["counter_ids"][row, column, k] = counter
        arrays["counter_counts"][row, column, k] = count


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
    weights: list[CopyWeights] | None = None,
) -> EntityTargetBatch:
    """The batch's per-entity targets.

    ``index`` selects the ``[CARD]`` and ``[PLAYER]`` columns of the trunk
    output; an entity no slot represents is not gathered. A field's value
    where it is not present is a zero the loss never reads. A record's
    targets carrying ``gated`` false (:class:`RecordTargets`) train no gate.
    ``weights`` is each record's rule mask (FR-061): a field it weighs zero
    is not present there, the gate trains only at the entities it weighs
    above zero, and a fractional weight rides beside the targets; None
    trains every record at full weight.
    """
    batch = len(surfaces)
    width = max((len(s.entity_slots) for s in surfaces), default=1)
    gated = _gated(targets_per_surface, batch)
    arrays = _TargetArrays.zeros((batch, width), fields)
    for row, (surface, targets) in enumerate(zip(surfaces, targets_per_surface)):
        arrays.fill(row, surface, targets)
    field_weights, gate_weight = arrays.rule_mask(surfaces, weights)
    return EntityTargetBatch(
        torch.from_numpy(arrays.gate),
        arrays.field_targets(field_weights),
        torch.from_numpy(arrays.mask),
        entity_index(surfaces),
        torch.from_numpy(gated),
        gate_weight,
    )


def _gated(targets_per_surface: list[dict], batch: int) -> np.ndarray:
    """``(batch,)`` bool: whether each record's targets train the gate."""
    return np.array(
        [getattr(targets, "gated", True) for targets in targets_per_surface] or [True],
        dtype=np.bool_,
    )[:batch]


@dataclass(frozen=True, slots=True)
class _TargetArrays:
    """One batch's per-entity target arrays, ``(batch, entities)`` (a
    per-member field's values with a trailing ``arity``), filled a surface at
    a time."""

    fields: tuple[FieldSpec, ...]
    gate: np.ndarray
    mask: np.ndarray
    collected: dict[str, np.ndarray]
    present: dict[str, np.ndarray]

    @classmethod
    def zeros(cls, shape: tuple[int, int], fields: tuple[FieldSpec, ...]) -> _TargetArrays:
        """Blank arrays of ``shape``: no entity real, affected or present."""
        gate = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        collected: dict[str, np.ndarray] = {}
        present: dict[str, np.ndarray] = {}
        for spec in fields:
            values = (*shape, spec.arity) if spec.type.per_member else shape
            collected[spec.name] = np.zeros(values, dtype=np.float32)
            present[spec.name] = np.zeros(shape, dtype=np.bool_)
        return cls(fields, gate, mask, collected, present)

    def fill(self, row: int, surface, targets) -> None:
        """``surface``'s entities' targets from ``targets``, along ``row``."""
        fields, gate, collected, present = self.fields, self.gate, self.collected, self.present
        slots = surface.slots
        self.mask[row, : len(surface.entity_slots)] = 1
        found = {
            column: entry for column, slot_index in enumerate(surface.entity_slots)
            if (entry := targets.get(slots[slot_index].subject_id)) is not None
        }
        for column, entry in found.items():
            for spec in fields:
                value = entry.supervised(spec)
                if value is None:
                    continue
                present[spec.name][row, column] = True
                if isinstance(value, list):
                    collected[spec.name][row, column, : len(value)] = value
                else:
                    collected[spec.name][row, column] = float(value)
            gate[row, column] = 1.0 if entry.affected else 0.0

    def rule_mask(
        self, surfaces: list, weights: list[CopyWeights] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Each record's rule-mask weights (FR-061) as the batch carries them:
        a field weighed zero cleared from ``present`` in place, the fractional
        field weights by field, and the gate's per-entity weight, None where
        every record is unmasked."""
        if weights is None or all(copy.unmasked for copy in weights):
            return {}, None
        field_weights = self._field_weights(weights)
        return field_weights, _gate_weights(surfaces, weights, self.mask.shape)

    def _field_weights(self, weights: list[CopyWeights]) -> dict[str, torch.Tensor]:
        """Each field's per-record weight, where any is fractional; a field
        a record weighs zero cleared from ``present`` on its row."""
        field_weights: dict[str, torch.Tensor] = {}
        for spec in self.fields:
            row_weights = np.array([copy.weight(spec.name) for copy in weights], dtype=np.float32)
            self.present[spec.name][row_weights == 0] = False
            if ((row_weights != 0) & (row_weights != 1)).any():
                field_weights[spec.name] = torch.from_numpy(row_weights)
        return field_weights

    def field_targets(self, field_weights: dict[str, torch.Tensor]) -> EntityFieldTargets:
        """The field arrays as tensors, beside ``field_weights``."""
        return EntityFieldTargets(
            {name: torch.from_numpy(values) for name, values in self.collected.items()},
            {name: torch.from_numpy(rows) for name, rows in self.present.items()},
            field_weights,
        )


def _gate_weights(
    surfaces: list, weights: list[CopyWeights], shape: tuple[int, ...],
) -> torch.Tensor:
    """``shape`` float: the gate's weight at each entity of a masked record,
    one at an unmasked record's and at padding."""
    gate_weight = np.ones(shape, dtype=np.float32)
    for row, (surface, copy) in enumerate(zip(surfaces, weights)):
        if copy.unmasked:
            continue
        for column, slot_index in enumerate(surface.entity_slots):
            gate_weight[row, column] = copy.gate_weight(surface.slots[slot_index].subject_id)
    return torch.from_numpy(gate_weight)


def per_entity_targets(
    records: list, surfaces: list, fields: tuple[FieldSpec, ...],
) -> EntityTargetBatch:
    """The per-entity targets of ``records``, built into ``surfaces``."""
    return entity_target_tensors(
        surfaces, [derive_targets(record) for record in records], fields,
    )
