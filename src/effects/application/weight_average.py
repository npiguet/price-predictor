"""The averaged weights validation, stopping and saving read (research T4).

An exponential moving average of the encoder's, the model's and the identity
table's parameters, updated after each optimizer step at rate
``grad_accum / (steps_per_epoch × ema_epochs)``: ``--steps-per-epoch`` counts
batches, so the rate spans ``--ema-epochs`` epochs of optimizer steps. It
starts from the weights at the end of warmup, since the warmup's weights are
the least trained of the run and the average would carry them for epochs.

One copy of every parameter, on the parameters' device. Validation swaps the
copy into the modules and back by exchanging the tensors' storage, so it adds
no second copy, and the trained weights come back bit for bit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping

import torch


def average_rate(*, grad_accum: int, steps_per_epoch: int, ema_epochs: float) -> float:
    """The weight each optimizer step's weights get in the average, at most 1."""
    return min(1.0, grad_accum / (steps_per_epoch * ema_epochs))


class WeightAverage:
    """The average over ``modules``' parameters, by module name and parameter
    name; empty until :meth:`after_optimizer_step` reaches ``warmup``."""

    def __init__(
        self, modules: Mapping[str, torch.nn.Module | None], *, rate: float, warmup: int,
    ) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"an average's rate is in (0, 1], not {rate}")
        self.rate = rate
        self.warmup = warmup
        self._parameters = {
            owner: dict(module.named_parameters())
            for owner, module in modules.items() if module is not None
        }
        #: ``owner -> name -> averaged tensor``; None before warmup's end.
        self._average: dict[str, dict[str, torch.Tensor]] | None = None
        #: The training step (batches) after which the average began.
        self.start_step: int | None = None

    @property
    def started(self) -> bool:
        return self._average is not None

    def after_optimizer_step(self, steps_taken: int) -> None:
        """Fold in the weights an optimizer step left, ``steps_taken`` being
        the run's training steps including this one: the first step at or
        past warmup's end copies the weights, every later one lerps."""
        if self._average is None:
            if steps_taken >= self.warmup:
                self._start(steps_taken)
            return
        averaged, current = self._pairs()
        if averaged:  # the foreach kernels refuse an empty list
            with torch.no_grad():
                torch._foreach_lerp_(averaged, current, self.rate)

    def _start(self, steps_taken: int) -> None:
        with torch.no_grad():
            self._average = {
                owner: {name: param.detach().clone() for name, param in params.items()}
                for owner, params in self._parameters.items()
            }
        self.start_step = steps_taken

    def _pairs(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        averaged: list[torch.Tensor] = []
        current: list[torch.Tensor] = []
        for owner, params in self._parameters.items():
            for name, param in params.items():
                averaged.append(self._average[owner][name])
                current.append(param.data)
        return averaged, current

    def _swap(self) -> None:
        """Exchange every parameter's storage with its average's."""
        for owner, params in self._parameters.items():
            row = self._average[owner]
            for name, param in params.items():
                param.data, row[name] = row[name], param.data

    @contextlib.contextmanager
    def applied(self) -> Iterator[bool]:
        """The modules holding the averaged weights for the block, the trained
        ones restored after it; the current weights, unchanged, before the
        average starts. Yields whether the average is in use."""
        if self._average is None:
            yield False
            return
        self._swap()
        try:
            yield True
        finally:
            self._swap()

    def nbytes(self) -> int:
        """The device memory the average holds."""
        return sum(
            tensor.numel() * tensor.element_size()
            for params in self._parameters.values() for tensor in params.values()
        )

    def state_dict(self) -> dict:
        """``state`` (the averaged tensors, None before warmup's end) and
        ``start_step``, as ``resume.pt`` carries them."""
        return {
            "state": (
                None if self._average is None else {
                    owner: {name: tensor.detach().clone() for name, tensor in row.items()}
                    for owner, row in self._average.items()
                }
            ),
            "start_step": self.start_step,
        }

    def load_state_dict(self, state: Mapping | None) -> None:
        """Restore what :meth:`state_dict` wrote; None or an unstarted state
        leaves the average to start at warmup's end, as the run it continues
        would have."""
        averaged = None if state is None else state.get("state")
        if averaged is None:
            self._average, self.start_step = None, None
            return
        if averaged.keys() != self._parameters.keys() or any(
            averaged[owner].keys() != params.keys()
            for owner, params in self._parameters.items()
        ):
            raise ValueError("the saved weight average covers other parameters than this run's")
        self._average = {
            owner: {
                name: averaged[owner][name].to(
                    device=param.device, dtype=param.dtype,
                ).clone()
                for name, param in params.items()
            }
            for owner, params in self._parameters.items()
        }
        self.start_step = state["start_step"]
