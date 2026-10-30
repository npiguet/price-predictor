"""A batch's per-entity targets, as tensors; kept apart from the
torch-free ``effect_head_layout`` so curation never loads torch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import torch


@dataclass(frozen=True)
class EntityFieldTargets:
    """A batch's per-entity field targets, each beside where it is present.

    ``values[name]`` is ``(batch, entities)``, or ``(batch, entities, arity)``
    for a multi-binary field; ``present[name]`` is a ``(batch, entities)``
    bool mask that stays on the host whatever device the values move to.
    """

    values: Mapping[str, torch.Tensor]
    present: Mapping[str, torch.Tensor]

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def to(self, device, *, non_blocking: bool = False) -> EntityFieldTargets:
        return EntityFieldTargets(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.values.items()
            },
            self.present,
        )

    def pinned(self) -> EntityFieldTargets:
        """The values in page-locked host memory, ready for a ``non_blocking``
        :meth:`to`; ``present`` stays where it is, never leaving the host."""
        return EntityFieldTargets(
            {name: value.pin_memory() for name, value in self.values.items()},
            self.present,
        )

    def select(self, rows) -> EntityFieldTargets:
        return EntityFieldTargets(
            {name: value[rows] for name, value in self.values.items()},
            {name: mask[rows] for name, mask in self.present.items()},
        )


@dataclass(frozen=True)
class EntityTargetBatch:
    """One batch's per-entity targets, aligned column for column.

    ``gate`` is 1 where an entity was affected, ``mask`` 1 for a real entity,
    and ``index`` the trunk-output column each entity is gathered from; all
    three are ``(batch, entities)``. Kept whole, so the loss, the margins and
    the constant-predictor floor all read exactly the entities the head did,
    selected by :attr:`real`, :attr:`affected` and :meth:`scored`. No slots:
    the two masks are cached, since the loss reads ``affected`` once per field.
    """

    gate: torch.Tensor
    fields: EntityFieldTargets
    mask: torch.Tensor
    index: torch.Tensor

    @cached_property
    def real(self) -> torch.Tensor:
        """``(batch, entities)`` bool on the host: a real entity, not padding.
        The gate trains on every one, players included."""
        return self.mask.detach().bool().cpu()

    @cached_property
    def affected(self) -> torch.Tensor:
        """The real entities the record's *target* says were affected; never
        the prediction's, or a head that gates everything off would escape
        the field terms."""
        return self.real & self.gate.detach().bool().cpu()

    def scored(self, name: str) -> torch.Tensor:
        """Where field ``name`` trains: an affected entity the field is
        present at (FR-081)."""
        return self.affected & self.fields.present[name].detach().bool().cpu()

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
        )
