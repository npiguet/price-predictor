"""A batch's per-entity targets, as tensors; kept apart from the
torch-free ``effect_head_layout`` so curation never loads torch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

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

    @classmethod
    def everywhere(cls, values: Mapping[str, torch.Tensor]) -> EntityFieldTargets:
        """``values`` present at every entity."""
        return cls(
            dict(values),
            {
                name: torch.ones(value.shape[:2], dtype=torch.bool)
                for name, value in values.items()
            },
        )

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

    def select(self, rows) -> EntityFieldTargets:
        return EntityFieldTargets(
            {name: value[rows] for name, value in self.values.items()},
            {name: mask[rows] for name, mask in self.present.items()},
        )


@dataclass(frozen=True, slots=True)
class EntityTargetBatch:
    """One batch's per-entity targets, aligned column for column.

    ``gate`` is 1 where an entity was affected, ``mask`` 1 for a real entity,
    and ``index`` the trunk-output column each entity is gathered from; all
    three are ``(batch, entities)``. Kept whole, so the loss, the margins and
    the constant-predictor floor all read exactly the entities the head did.
    """

    gate: torch.Tensor
    fields: EntityFieldTargets
    mask: torch.Tensor
    index: torch.Tensor

    def fields_on(self, device, *, non_blocking: bool = False) -> EntityTargetBatch:
        """The batch with its field values on ``device``. The gate, mask and
        index stay where they are: the loss picks its rows on the host, so
        picking them never waits on the device."""
        return replace(self, fields=self.fields.to(device, non_blocking=non_blocking))

    def select(self, rows) -> EntityTargetBatch:
        return EntityTargetBatch(
            self.gate[rows], self.fields.select(rows), self.mask[rows], self.index[rows],
        )
