"""A batch's per-entity field targets, as tensors.

Apart from :mod:`effects.domain.effect_head_layout`, which names the fields and
needs no torch, so the curation passes that derive targets record by record
never load it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EntityFieldTargets:
    """A batch's per-entity field targets, each beside where it is present.

    ``values[name]`` is ``(batch, entities)`` — ``(batch, entities, arity)`` for
    a multi-binary field — and ``present[name]`` the ``(batch, entities)`` bool
    mask of the entities whose record supervises the field. A field is scored
    only where the gate's target fires *and* it is present.

    ``present`` stays on the host whatever device the values move to: the loss
    picks its rows there, so no step waits for the device to say which.
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
        """The values on ``device``; the presence masks stay where they are."""
        return EntityFieldTargets(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.values.items()
            },
            self.present,
        )

    def select(self, rows) -> EntityFieldTargets:
        """The records ``rows`` indexes, every field and mask alike."""
        return EntityFieldTargets(
            {name: value[rows] for name, value in self.values.items()},
            {name: mask[rows] for name, mask in self.present.items()},
        )
