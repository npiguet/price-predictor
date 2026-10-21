"""Token surfaces and their targets, padded into the trunk's tensors.

Built in numpy and handed to torch once per array: this is every step's host
half, and a per-element torch assignment costs a dispatch per element.
"""

from __future__ import annotations

import numpy as np
import torch

from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_head_layout import EntityFieldTargets, FieldSpec, FieldType


def gather_entities(outputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``(batch, entities, width)``: each surface's ``[CARD]`` and ``[PLAYER]``
    columns of the per-entity output, in one batched gather."""
    return outputs.gather(1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))


def collate_surfaces(
    surfaces: list, *, e_dim: int, widths: dict[SlotKind, int],
) -> dict[str, torch.Tensor]:
    """Pad a batch of token surfaces into the trunk's keyword arguments.

    Each slot kind's features go into its own zero-padded tensor, so the model
    can project all four with four matmuls rather than gathering per slot. The
    batch pads to its own longest surface: board sizes vary a great deal and
    padding to a fixed width would spend most of the compute on empty slots.

    A slot whose ``e`` is an ``int`` names a row of a matrix this function does
    not have — the training loop's live encoder output. Its row lands in
    ``e_rows`` (``-1`` everywhere else) and the caller resolves it on the device
    with :func:`scatter_e_rows`. Resolving it here would mean pulling the
    encoder's vectors back to the host once per ability slot, which both costs a
    synchronization per slot and severs the gradient.

    The padding is done in numpy and handed to torch once per array, because a
    batch of 32 surfaces holds about two thousand slots and the per-slot
    ``torch.tensor`` this used to build was the expensive part: each one takes
    the tensor-construction path for a handful of floats, and there were three
    hundred thousand of them over a four-shard run. numpy pads into a buffer
    that is already the right shape and ``torch.from_numpy`` wraps it without
    copying, so the same bytes reach the model.
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
        # Whole rows at a time: these three are one scalar per slot, and a
        # numpy element assignment costs about what the row assignment does.
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

    return {
        "slot_features": {
            kind: torch.from_numpy(array) for kind, array in features.items()
        },
        "slot_kinds": torch.from_numpy(slot_kinds),
        "positions": torch.from_numpy(positions),
        "e_vectors": torch.from_numpy(e_vectors),
        "e_rows": torch.from_numpy(e_rows),
        "attention_mask": torch.from_numpy(attention),
        "option_kinds": torch.from_numpy(option_kinds),
    }


def scatter_e_rows(
    e_vectors: torch.Tensor, e_rows: torch.Tensor, matrix: torch.Tensor,
) -> torch.Tensor:
    """Fill the slots that named a row of ``matrix`` with that row.

    One gather over the whole batch, on whatever device the matrix is on, and
    differentiable — which is the point. ``e`` is what the ability encoder
    produces, so the effect head's loss reaches the encoder only through this
    path; a per-slot host round trip would train the head against a constant.
    """
    valid = e_rows >= 0
    gathered = matrix[e_rows.clamp(min=0)]
    return torch.where(valid.unsqueeze(-1), gathered, e_vectors)


def entity_target_tensors(
    surfaces: list, targets_per_surface: list[dict], fields: tuple[FieldSpec, ...],
) -> tuple[torch.Tensor, EntityFieldTargets, torch.Tensor, torch.Tensor]:
    """Gather per-entity outputs and their targets into aligned tensors.

    Returns ``(gate_target, field_targets, entity_mask, entity_index)``, where
    ``entity_index`` selects the ``[CARD]`` and ``[PLAYER]`` columns of the
    trunk output. A record that mentions an entity no slot represents is simply
    not gathered — the surface is the authority on what exists. Each field is
    present only where its entity's targets supervise it
    (``EntityTargets.supervised``); everywhere else its value is a placeholder
    zero the loss never reads.
    """
    # Built in numpy and handed over whole: assigning element by element into
    # torch tensors costs a dispatch per element, on every step's host half.
    batch = len(surfaces)
    width = max((len(s.entity_slots) for s in surfaces), default=1)
    gate = np.zeros((batch, width), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=np.float32)
    index = np.zeros((batch, width), dtype=np.int64)
    collected: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}

    for spec in fields:
        # Only a multi-binary field's *target* is as wide as its prediction.
        # A categorical is just as wide to predict and one class index to
        # supervise, so its target must stay scalar per entity — cross-entropy
        # takes indices, not one-hots.
        shape = (
            (batch, width, spec.arity)
            if spec.type is FieldType.MULTI_BINARY
            else (batch, width)
        )
        collected[spec.name] = np.zeros(shape, dtype=np.float32)
        present[spec.name] = np.zeros((batch, width), dtype=np.bool_)

    for row, (surface, targets) in enumerate(zip(surfaces, targets_per_surface)):
        for column, slot_index in enumerate(surface.entity_slots):
            slot = surface.slots[slot_index]
            key = slot.entity_id or slot.player_id
            mask[row, column] = 1
            index[row, column] = slot_index
            entry = targets.get(key)
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
    return (
        torch.from_numpy(gate),
        EntityFieldTargets(
            {name: torch.from_numpy(values) for name, values in collected.items()},
            {name: torch.from_numpy(rows) for name, rows in present.items()},
        ),
        torch.from_numpy(mask),
        torch.from_numpy(index),
    )
