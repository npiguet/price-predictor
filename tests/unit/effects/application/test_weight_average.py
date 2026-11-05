"""The averaged weights (research T4): when they start, how they move, and
that reading them leaves the trained weights exactly as they were."""

from __future__ import annotations

import pytest
import torch

from effects.application.weight_average import WeightAverage, average_rate


def _modules():
    torch.manual_seed(0)
    return {"encoder": torch.nn.Linear(3, 2), "model": torch.nn.Linear(2, 1)}


def _step(modules, by: float) -> None:
    """An optimizer step's worth of change: every parameter moves by ``by``."""
    with torch.no_grad():
        for module in modules.values():
            for param in module.parameters():
                param.add_(by)


def _weights(modules) -> dict[str, torch.Tensor]:
    return {
        f"{owner}.{name}": value.detach().clone()
        for owner, module in modules.items()
        for name, value in module.state_dict().items()
    }


def test_the_rate_spans_ema_epochs_of_optimizer_steps():
    """``--steps-per-epoch`` counts batches, so accumulation raises the rate."""
    assert average_rate(grad_accum=1, steps_per_epoch=5000, ema_epochs=1) == 1 / 5000
    assert average_rate(grad_accum=4, steps_per_epoch=5000, ema_epochs=2) == 4 / 10000
    assert average_rate(grad_accum=8, steps_per_epoch=2, ema_epochs=1) == 1.0


def test_a_rate_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        WeightAverage(_modules(), rate=0.0, warmup=1)


def test_nothing_is_averaged_before_warmup_ends_and_its_end_copies_the_weights():
    modules = _modules()
    average = WeightAverage(modules, rate=0.5, warmup=3)
    for steps_taken in (1, 2):
        _step(modules, 1.0)
        average.after_optimizer_step(steps_taken)
        assert not average.started
    _step(modules, 1.0)
    average.after_optimizer_step(3)
    assert average.started and average.start_step == 3
    trained = _weights(modules)
    with average.applied() as averaged:
        assert averaged
        assert all(torch.equal(v, _weights(modules)[k]) for k, v in trained.items())


def test_each_later_step_lerps_toward_the_current_weights():
    modules = _modules()
    start = _weights(modules)
    average = WeightAverage(modules, rate=0.25, warmup=1)
    average.after_optimizer_step(1)  # starts: the average is ``start``
    _step(modules, 4.0)
    average.after_optimizer_step(2)  # start + 0.25 * 4 = start + 1
    _step(modules, 4.0)
    average.after_optimizer_step(3)  # (start + 1) + 0.25 * (start + 8 - start - 1)
    with average.applied():
        for name, value in _weights(modules).items():
            assert torch.allclose(value, start[name] + 1.0 + 0.25 * 7.0), name


def test_reading_the_average_restores_the_trained_weights_bit_for_bit():
    modules = _modules()
    average = WeightAverage(modules, rate=0.1, warmup=1)
    average.after_optimizer_step(1)
    _step(modules, 0.3)
    average.after_optimizer_step(2)
    trained = _weights(modules)
    params = [p for m in modules.values() for p in m.parameters()]
    with average.applied():
        assert any(
            not torch.equal(v, _weights(modules)[k]) for k, v in trained.items()
        )
    after = _weights(modules)
    assert all(torch.equal(v, after[k]) for k, v in trained.items())
    # The same parameter objects: the optimizer's state stays keyed to them.
    assert params == [p for m in modules.values() for p in m.parameters()]


def test_before_warmup_ends_the_current_weights_are_read():
    """Story 7 acceptance 4: such an epoch is validated on them."""
    modules = _modules()
    average = WeightAverage(modules, rate=0.1, warmup=10)
    trained = _weights(modules)
    with average.applied() as averaged:
        assert not averaged
        assert all(torch.equal(v, _weights(modules)[k]) for k, v in trained.items())


def test_the_state_round_trips_and_an_unstarted_one_starts_at_warmups_end():
    modules = _modules()
    average = WeightAverage(modules, rate=0.5, warmup=1)
    average.after_optimizer_step(1)
    _step(modules, 2.0)
    average.after_optimizer_step(2)
    saved = average.state_dict()
    assert saved["start_step"] == 1 and set(saved["state"]) == {"encoder", "model"}

    restored = WeightAverage(modules, rate=0.5, warmup=1)
    restored.load_state_dict(saved)
    with average.applied():
        expected = _weights(modules)
    with restored.applied():
        assert all(torch.equal(v, _weights(modules)[k]) for k, v in expected.items())

    unstarted = WeightAverage(modules, rate=0.5, warmup=5)
    unstarted.load_state_dict({"state": None, "start_step": None})
    assert not unstarted.started
    unstarted.after_optimizer_step(5)
    assert unstarted.start_step == 5


def test_an_absent_module_is_left_out_and_a_foreign_state_refused():
    modules = {**_modules(), "identity_table": None}
    average = WeightAverage(modules, rate=0.5, warmup=1)
    average.after_optimizer_step(1)
    assert set(average.state_dict()["state"]) == {"encoder", "model"}
    assert average.nbytes() == 4 * (3 * 2 + 2 + 2 * 1 + 1)
    with pytest.raises(ValueError, match="other parameters"):
        average.load_state_dict({"state": {"encoder": {}}, "start_step": 1})
