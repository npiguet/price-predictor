"""The CUDA reservation cap and the per-shard cache release (FR-095).

The GPU work itself is untestable without a real card, so what is pinned here
is when the loop makes its two CUDA calls. Every torch.cuda.* function used
below is a stub or a spy — nothing here touches a real device, matching the
rest of this package's CPU-only unit suite.
"""

from __future__ import annotations

import pytest
import torch

from effects.application.train_effect_model import TrainEffectModelConfig
from effects.application.training_loop import CUDA_MEMORY_FRACTION, TrainingLoop
from tests.unit.effects.builders import make_training_loop


class TestCudaReservationAndCacheRelease:
    """The loop's two CUDA calls, made on a CUDA device and on no other."""

    @pytest.fixture
    def calls(self, monkeypatch) -> list[tuple]:
        calls: list[tuple] = []
        monkeypatch.setattr(
            torch.cuda, "set_per_process_memory_fraction",
            lambda fraction: calls.append(("cap", fraction)),
        )
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("release",)))
        return calls

    @staticmethod
    def _on(device: str) -> TrainingLoop:
        loop = make_training_loop(TrainEffectModelConfig(corpus="unused"))
        loop.device = torch.device(device)
        return loop

    def test_neither_call_happens_on_cpu(self, calls):
        loop = self._on("cpu")
        loop._cap_cuda_memory()
        loop._release_cache()
        assert calls == []

    def test_a_cuda_loop_caps_its_reservation_and_releases_the_cache(self, calls):
        loop = self._on("cuda")
        loop._cap_cuda_memory()
        loop._release_cache()
        assert calls == [("cap", CUDA_MEMORY_FRACTION), ("release",)]
