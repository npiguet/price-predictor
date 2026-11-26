"""What every head's loss shares: the rows a host mask selects, the average
over a batch's rows and the scaling by loss denominators (FR-062)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Rows:
    """The rows a host mask selects: tested and indexed on the host, where the
    mask was built, and copied to the device once for gathering there."""

    host: torch.Tensor
    device: torch.Tensor

    @classmethod
    def of(cls, mask: torch.Tensor, device: torch.device) -> Rows:
        host = mask.detach().bool().cpu().flatten().nonzero().squeeze(1)
        return cls(host, host.to(device, non_blocking=True))

    @property
    def empty(self) -> bool:
        return self.host.numel() == 0

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor``'s selected rows, gathered wherever ``tensor`` lives."""
        return tensor.index_select(0, self.host if tensor.device.type == "cpu" else self.device)


def per_row(total: torch.Tensor, rows: int) -> torch.Tensor:
    return total / max(int(rows), 1)


def scaled(sums: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Per-term sums divided by their loss denominators (FR-062): ``scale``
    holds each term's reciprocal, in the same order; None leaves them as they
    are."""
    return sums.sum() if scale is None else (sums * scale).sum()
