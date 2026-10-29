"""A batch's per-entity field targets, as tensors; kept apart from the
torch-free ``effect_head_layout`` so curation never loads torch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
