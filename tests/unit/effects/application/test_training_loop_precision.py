"""bf16 autocast and the CUDA reservation cap (FR-095).

The GPU work itself is untestable without a real card, so what is pinned here
is the *wiring*: which flag `_loss_for` reads to decide whether to enter
`torch.autocast`, and which of the three CUDA calls `execute` makes and when.
Every torch.cuda.* function used below is a spy or a stub — nothing here
touches a real device, matching the rest of this package's CPU-only unit
suite.
"""

from __future__ import annotations

import torch

from effects.application import training_loop as loop_module
from effects.application.train_effect_model import (
    HeldOutCards,
    TrainEffectModelConfig,
)
from effects.application.training_loop import TrainingLoop, autocast_enabled
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.domain.conftest import (  # noqa: F401
    ability_key,
    make_record,
    snapshot,
)


class _AutocastSpy:
    """Stands in for `torch.autocast`: records its kwargs, does nothing else."""

    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestAutocastEnabled:
    """The pure helper `_loss_for`'s flag is computed from (FR-095)."""

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


def _bare_loop(monkeypatch) -> TrainingLoop:
    """A loop built with no corpus, sized only for `_loss_for`'s own plumbing.

    Forces a real ``cpu`` device regardless of what this machine's
    ``torch.cuda.is_available()`` says: on this box it can read ``True`` even
    under ``CUDA_VISIBLE_DEVICES=""`` (its raw device count ignores the
    mask — the same disagreement ``autocast_enabled`` guards against), which
    would otherwise move this test's tensors onto a real GPU by accident.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )


def _run_loss_for(monkeypatch, loop, *, autocast: bool):
    """Call the real `_loss_for` with every internal step stubbed out.

    What is under test is only whether the call is wrapped in
    `torch.autocast(dtype=torch.bfloat16, enabled=...)` — not the batching,
    targets or loss math, which their own suites already cover. Forcing
    `loop.autocast` directly (rather than faking a CUDA device) is the
    fallback the task called for: a device fake would have to fake every
    other CUDA-touching call `execute` and `SurfaceBatcher` make along the
    way, none of which this test is about.
    """
    loop.autocast = autocast
    _AutocastSpy.calls = []
    monkeypatch.setattr(loop_module.torch, "autocast", _AutocastSpy)
    monkeypatch.setattr(TrainingLoop, "_batcher", lambda self, *a, **k: (
        type("B", (), {"build": staticmethod(lambda records, encoder: ({}, []))})()
    ))
    monkeypatch.setattr(loop_module, "derive_targets", lambda record: None)

    index = torch.zeros((1, 1), dtype=torch.long)
    gate = torch.zeros((1, 1))
    mask = torch.zeros((1, 1))
    monkeypatch.setattr(
        loop_module, "entity_target_tensors",
        lambda surfaces, targets, fields: (gate, {}, mask, index),
    )
    monkeypatch.setattr(
        loop_module, "per_entity_loss",
        lambda *a, **k: (torch.tensor(0.0), {}),
    )

    class _Plan:
        records = [object()]

    class _Model:
        def __call__(self, **batch):
            return "hidden"

        def per_entity(self, hidden):
            return torch.zeros((1, 1, 3))

    result = loop._loss_for(
        _Plan(), encoder=None, model=_Model(), tokenizer=None, sidecars=None,
        widths={}, step=0, fields=("dummy",),
    )
    assert result is not None
    return _AutocastSpy.calls


class TestLossForAutocast:
    def test_disabled_on_a_cpu_loop(self, monkeypatch):
        calls = _run_loss_for(monkeypatch, _bare_loop(monkeypatch), autocast=False)
        assert calls == [{"device_type": "cuda", "dtype": torch.bfloat16, "enabled": False}]

    def test_enabled_when_the_loop_says_so(self, monkeypatch):
        calls = _run_loss_for(monkeypatch, _bare_loop(monkeypatch), autocast=True)
        assert calls == [{"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}]


class _Stub(torch.nn.Module):
    """A module with one parameter, so backward and the optimizer have work."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2))


class _Metrics:
    affected_gate_f1 = 0.5
    zone_outcome_accuracy = 0.5
    mean_poisson_deviance = 0.5


def _run_execute(monkeypatch, tmp_path, make_record, *, cuda: bool):  # noqa: F811
    """Run one real epoch/shard with everything below the loop stubbed.

    Mirrors `test_training_loop_execute.py`'s fixture (same stubs for the
    encoder, head, tokenizer, sidecars, feature widths, loss and both
    validation passes), plus what faking a CUDA device without a real GPU
    needs: `torch.cuda.is_available`/`is_bf16_supported` answer the way the
    fake device should, `nn.Module.to` and `torch.zeros` stop trying to
    actually touch a card that is not there, and `set_per_process_memory_
    fraction`/`empty_cache` are spies rather than real CUDA calls.
    """
    combat = [make_record(record_id=f"r{i}", game_id="g0",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(4)]
    shard = tmp_path / "train.jsonl.gz"
    write_shard(shard, combat)
    samples = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        samples[stratum] = tmp_path / f"{stratum}.jsonl.gz"
        write_shard(samples[stratum], combat[:2])

    model, encoder = _Stub(), _Stub()

    monkeypatch.setattr(loop_module, "AbilityEncoder", lambda config: encoder)
    monkeypatch.setattr(loop_module, "EffectModel", lambda config: model)
    monkeypatch.setattr(
        TrainingLoop, "_build_tokenizer",
        lambda self: type("T", (), {"vocab_size": 8})(),
    )
    from effects.infrastructure.sidecar_io import SidecarCache

    monkeypatch.setattr(TrainingLoop, "_build_sidecars", lambda self: SidecarCache({}))
    monkeypatch.setattr(
        TrainingLoop, "_feature_widths",
        lambda self, records: dict.fromkeys(loop_module.SlotKind, 4),
    )
    monkeypatch.setattr(
        TrainingLoop, "_loss_for",
        lambda self, plan, *a, **k: (model.p.sum(), {}),
    )
    monkeypatch.setattr(
        TrainingLoop, "_validate", lambda self, records, *a, **k: 1.0,
    )
    monkeypatch.setattr(loop_module, "measure", lambda *a, **k: _Metrics())

    memory_fraction_calls: list[float] = []
    empty_cache_calls: list[None] = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda fraction: memory_fraction_calls.append(fraction),
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if cuda else 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: cuda)

    if cuda:
        # The loop asks nothing else of the "device" than these two things:
        # a place tensors can be created without erroring, and the modules
        # accepting `.to(device)`. Neither cares that the device is not real.
        monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
        real_zeros = torch.zeros
        monkeypatch.setattr(
            torch, "zeros",
            lambda *a, **k: real_zeros(*a, **{k2: v for k2, v in k.items() if k2 != "device"}),
        )

    config = TrainEffectModelConfig(
        corpus=str(tmp_path), epochs=1, steps_per_epoch=2,
        shards_per_epoch=1, batch_size=2, seed=7,
        model_output=tmp_path / "out",
    )
    code = TrainingLoop(
        config, held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[shard], validation_samples=samples,
        holdout_permille=20, holdout_max_carriers=8, gate_one_records=0,
        rarity={}, corpus_digest="",
    ).execute()
    assert code == 0
    return memory_fraction_calls, empty_cache_calls


class TestCudaReservationAndCacheRelease:
    def test_neither_call_happens_on_cpu(self, monkeypatch, tmp_path, make_record):  # noqa: F811
        fraction_calls, cache_calls = _run_execute(
            monkeypatch, tmp_path, make_record, cuda=False,
        )
        assert fraction_calls == []
        assert cache_calls == []

    def test_the_fraction_is_capped_once_on_a_cuda_loop(
        self, monkeypatch, tmp_path, make_record,  # noqa: F811
    ):
        fraction_calls, _ = _run_execute(monkeypatch, tmp_path, make_record, cuda=True)
        assert fraction_calls == [loop_module.CUDA_MEMORY_FRACTION]
        assert loop_module.CUDA_MEMORY_FRACTION == 0.9

    def test_the_cache_is_released_once_per_shard_on_a_cuda_loop(
        self, monkeypatch, tmp_path, make_record,  # noqa: F811
    ):
        # One epoch, one shard: exactly one shard is trained, so exactly one
        # release.
        _, cache_calls = _run_execute(monkeypatch, tmp_path, make_record, cuda=True)
        assert len(cache_calls) == 1
