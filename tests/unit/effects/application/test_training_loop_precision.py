"""bf16 autocast and the CUDA reservation cap (FR-095).

The GPU work itself is untestable without a real card, so what is pinned here
is when the device half's autocast flag is set and when the loop makes its two
CUDA calls. Every torch.cuda.* function used below is a stub or a spy —
nothing here touches a real device, matching the rest of this package's
CPU-only unit suite.
"""

from __future__ import annotations

import pytest
import torch

from effects.application.step_loss import autocast_enabled
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.application.training_loop import CUDA_MEMORY_FRACTION, TrainingLoop
from tests.unit.effects.builders import make_training_loop


class TestAutocastEnabled:
    """The pure helper the device half's flag is computed from (FR-095)."""

    def test_a_cpu_device_is_never_autocast(self):
        assert autocast_enabled(torch.device("cpu")) is False

    def test_a_cuda_device_follows_bf16_support(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert autocast_enabled(torch.device("cuda")) is True

    def test_a_cuda_device_without_bf16_support_is_not_autocast(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        assert autocast_enabled(torch.device("cuda")) is False

    def test_a_masked_empty_device_is_not_autocast(self, monkeypatch):
        """`is_available()` and `device_count()` can disagree under
        `CUDA_VISIBLE_DEVICES=""` (the former reads the driver's raw,
        unmasked count); a masked-empty device must read as unsupported
        rather than raise out of `is_bf16_supported()`."""
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        def _boom():
            raise AssertionError("Invalid device id")

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", _boom)
        assert autocast_enabled(torch.device("cuda")) is False


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
