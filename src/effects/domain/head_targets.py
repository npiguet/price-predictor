"""What every auxiliary head's targets share: a dataclass of tensors that can
be pinned, sent to a device, cut to some of its rows, or read back on the host.

Each head names the target tensors its loss reads on the device (:attr:`ON_DEVICE`);
the rest are masks and indices it selects rows by on the host and stay there.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import ClassVar, Self

import torch


class HeadTargetParts:
    """The tensor-carrying operations of a head's targets dataclass: every
    field holds a tensor or None."""

    __slots__ = ()

    #: The fields the head's loss reads on the device.
    ON_DEVICE: ClassVar[tuple[str, ...]] = ()

    def _mapped(self, move) -> Self:
        """The targets with ``move(name, tensor)`` applied to each tensor
        held, a field left None staying so."""
        return replace(self, **{
            part.name: None if getattr(self, part.name) is None
            else move(part.name, getattr(self, part.name))
            for part in fields(self)  # type: ignore[arg-type]
        })

    def pinned(self) -> Self:
        """Every tensor in page-locked host memory, so a ``non_blocking``
        copy does not hold the host."""
        return self._mapped(lambda _name, tensor: tensor.pin_memory())

    def to(self, device: torch.device | str) -> Self:
        """The tensors the loss reads on ``device`` moved there without
        blocking the host; the others stay where they are."""
        return self._mapped(
            lambda name, tensor:
            tensor.to(device, non_blocking=True) if name in self.ON_DEVICE else tensor
        )

    def on_host(self) -> Self:
        """Every tensor detached and on the host."""
        return self._mapped(lambda _name, tensor: tensor.detach().cpu())

    def rows(self, selected: torch.Tensor) -> Self:
        """The targets of the records ``selected`` (a boolean per record)
        picks, every tensor detached and on the host."""
        return self._mapped(lambda _name, tensor: tensor.detach().cpu()[selected])
