"""A batch's per-entity targets, as tensors; kept apart from the
torch-free ``effect_head_layout`` so curation never loads torch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property

import torch

from effects.domain.effect_head_layout import FIELDS_BY_NAME


@dataclass(frozen=True)
class EntityFieldTargets:
    """A batch's per-entity field targets, each beside where it is present.

    ``values[name]`` is ``(batch, entities)``, or ``(batch, entities, arity)``
    for a multi-binary field; ``present[name]`` is a ``(batch, entities)``
    bool mask that stays on the host whatever device the values move to.
    ``weights[name]``, where set, is a ``(batch,)`` float on the host: each
    record's weight on the field under the rule mask (FR-061); a field it
    does not hold weighs one wherever present.
    """

    values: Mapping[str, torch.Tensor]
    present: Mapping[str, torch.Tensor]
    weights: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def to(self, device, *, non_blocking: bool = False) -> EntityFieldTargets:
        return EntityFieldTargets(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.values.items()
            },
            self.present,
            self.weights,
        )

    def pinned(self) -> EntityFieldTargets:
        """The values in page-locked host memory, ready for a ``non_blocking``
        :meth:`to`; ``present`` stays where it is, never leaving the host."""
        return EntityFieldTargets(
            {name: value.pin_memory() for name, value in self.values.items()},
            self.present,
            self.weights,
        )

    def select(self, rows) -> EntityFieldTargets:
        return EntityFieldTargets(
            {name: value[rows] for name, value in self.values.items()},
            {name: mask[rows] for name, mask in self.present.items()},
            {name: weight[rows] for name, weight in self.weights.items()},
        )


@dataclass(frozen=True)
class EntityTargetBatch:
    """One batch's per-entity targets, aligned column for column.

    ``gate`` is 1 where an entity was affected, ``mask`` 1 for a real entity,
    and ``index`` the trunk-output column each entity is gathered from; all
    three are ``(batch, entities)``. ``gated`` is ``(batch,)``, 1 for a record
    that trains the gate (a playability record does not); None trains it on
    every record. ``gate_weight``, ``(batch, entities)`` float, is the rule
    mask's weight on the gate at each entity (FR-061); None weighs one
    everywhere. Kept whole, so the loss, the margins and the
    constant-predictor floor all read exactly the entities the head did,
    selected by :attr:`real`, :attr:`gate_scored`, :attr:`affected` and
    :meth:`scored`. No slots: the masks are cached, since the loss reads
    ``affected`` once per field.
    """

    gate: torch.Tensor
    fields: EntityFieldTargets
    mask: torch.Tensor
    index: torch.Tensor
    gated: torch.Tensor | None = None
    gate_weight: torch.Tensor | None = None

    @cached_property
    def real(self) -> torch.Tensor:
        """``(batch, entities)`` bool on the host: a real entity, not padding.
        The gate trains on every one, players included."""
        return self.mask.detach().bool().cpu()

    @cached_property
    def gate_scored(self) -> torch.Tensor:
        """``(batch, entities)`` bool on the host: where the gate trains, every
        real entity of a record that trains it (root spec § 11.4) the rule
        mask leaves a weight on (FR-061)."""
        scored = self.real
        if self.gated is not None:
            scored = scored & self.gated.detach().bool().cpu().unsqueeze(-1)
        if self.gate_weight is not None:
            scored = scored & (self.gate_weight.detach().cpu() > 0)
        return scored

    def gate_weights(self) -> torch.Tensor | None:
        """``(batch, entities)`` float on the host: the gate's weight at each
        entity, read where :attr:`gate_scored` is set; None weighs one."""
        return None if self.gate_weight is None else self.gate_weight.detach().cpu().float()

    @cached_property
    def affected(self) -> torch.Tensor:
        """The real entities the record's *target* says were affected; never
        the prediction's, or a head that gates everything off would escape
        the field terms."""
        return self.real & self.gate.detach().bool().cpu()

    def scored(self, name: str) -> torch.Tensor:
        """Where field ``name`` trains: an affected entity the field is
        present at (FR-081); a legality field, every real entity it is
        present at, whatever the gate's target (root spec § 11.4)."""
        spec = FIELDS_BY_NAME.get(name)
        where = self.affected if spec is None or spec.gated else self.real
        where = where & self.fields.present[name].detach().bool().cpu()
        weight = self.fields.weights.get(name)
        if weight is not None:
            where = where & (weight.detach().cpu() > 0).unsqueeze(-1)
        return where

    def field_weights(self, name: str) -> torch.Tensor | None:
        """``(batch, entities)`` float on the host: field ``name``'s weight at
        each entity under the rule mask (FR-061), read where :meth:`scored`
        is set; None weighs one."""
        weight = self.fields.weights.get(name)
        if weight is None:
            return None
        return weight.detach().cpu().float().unsqueeze(-1).expand_as(self.mask)

    def fields_on(self, device, *, non_blocking: bool = False) -> EntityTargetBatch:
        """The batch with its field values on ``device``. The gate, mask and
        index stay where they are: the loss picks its rows on the host, so
        picking them never waits on the device."""
        return replace(self, fields=self.fields.to(device, non_blocking=non_blocking))

    def pinned(self) -> EntityTargetBatch:
        """The batch with what reaches the device, the field values and
        ``index``, in page-locked host memory; the gate and mask stay host
        tensors, since the loss picks its rows from them on the host."""
        return replace(self, fields=self.fields.pinned(), index=self.index.pin_memory())

    def select(self, rows) -> EntityTargetBatch:
        return EntityTargetBatch(
            self.gate[rows], self.fields.select(rows), self.mask[rows], self.index[rows],
            None if self.gated is None else self.gated[rows],
            None if self.gate_weight is None else self.gate_weight[rows],
        )
