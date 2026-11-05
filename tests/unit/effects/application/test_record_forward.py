"""The one chunked forward over records (research E2): validation's loss,
gate 1's margins, the breakdowns, the probes and the control score all read
the model through it, in evaluation mode, without a gradient, under
validation's autocast."""

from __future__ import annotations

import math
from itertools import batched
from types import SimpleNamespace

import pytest
import torch

from effects.application import record_forward
from effects.application.record_forward import RecordForward, entity_chunks
from effects.application.step_loss import StepLoss
from effects.application.surface_batching import Materializer
from effects.application.validation_pass import score_steps
from effects.domain.baselines import BaselineMasks
from effects.domain.effect_losses import LossParts
from effects.domain.surface_collation import entity_index, per_entity_targets
from tests.unit.effects.builders import StubLosses, fixture_run, one_parameter_module


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("forward"))


def _forward(run) -> RecordForward:
    return RecordForward(run.encoder, run.model, run.scoring)


def _batcher(run):
    batcher = run.preparer.scoring_batcher()
    batcher.materializer = run.scoring
    return batcher


class TestEntityChunks:
    def test_each_chunk_is_the_model_run_once_over_it(self, run, records):
        """What ``model_loading.forward_chunks`` read, through the one forward:
        the per-entity head at each surface's entities and the unmasked targets."""
        chosen = records[:40]
        fields = tuple(run.fields[1])
        got = list(entity_chunks(chosen, _forward(run), _batcher(run), fields, size=16))

        assert [len(chunk) for chunk, _, _ in got] == [16, 16, 8]
        assert [r.record_id for chunk, _, _ in got for r in chunk] == [
            r.record_id for r in chosen]
        run.encoder.eval()
        run.model.eval()
        batcher = _batcher(run)
        with torch.no_grad():
            for (chunk, gathered, targets), expected in zip(got, batched(chosen, 16)):
                built = batcher.build(list(expected), run.encoder)
                hidden = run.model(built.inputs)
                want = run.model.per_entity_at(hidden, entity_index(built.surfaces))
                assert torch.equal(gathered, want.float())
                assert gathered.device.type == "cpu"
                assert torch.equal(targets.gate, per_entity_targets(
                    list(expected), built.surfaces, fields).gate)
        run.encoder.train()
        run.model.train()

    def test_the_record_heads_read_the_same_trunk(self, run, records):
        chosen = records[:8]
        forward = _forward(run)
        (chunk,) = forward.records(chosen, _batcher(run), size=8)
        with forward.running():
            verdict = chunk.outputs.verdict()
            again = run.model.verdict(chunk.outputs.hidden).float()
        assert verdict.shape == (8, again.shape[1]) and torch.equal(verdict, again)
        assert chunk.outputs.per_entity().shape[:2] == chunk.outputs.hidden.shape[:2]


class TestRunning:
    def test_modules_train_again_after_the_block_and_no_gradient_inside(self):
        encoder, model = one_parameter_module(), one_parameter_module()
        forward = RecordForward(encoder, model, None)
        with forward.running():
            assert not encoder.training and not model.training
            assert not torch.is_grad_enabled()
        assert encoder.training and model.training
        assert torch.is_grad_enabled()

    def test_a_module_in_evaluation_mode_stays_there(self):
        encoder, model = one_parameter_module(), one_parameter_module()
        encoder.eval()
        model.eval()
        with RecordForward(encoder, model, None).running():
            pass
        assert not encoder.training and not model.training

    def test_a_failing_block_still_puts_training_back(self):
        encoder, model = one_parameter_module(), one_parameter_module()
        with pytest.raises(RuntimeError), RecordForward(encoder, model, None).running():
            raise RuntimeError("boom")
        assert encoder.training and model.training


class TestAutocast:
    def test_off_on_the_cpu_and_without_a_materializer(self):
        module = one_parameter_module()
        cpu = Materializer(torch.device("cpu"), BaselineMasks())
        assert not RecordForward(module, module, cpu).autocast
        assert not RecordForward(module, module, None).autocast

    def test_it_follows_validations_rule(self, monkeypatch):
        """The rule ``StepLossComputer`` scores validation under, so a probe, a
        control score and a validation pass read one precision."""
        asked = []
        monkeypatch.setattr(record_forward, "autocast_enabled",
                            lambda device: asked.append(device) or True)
        module = one_parameter_module()
        device = torch.device("cpu")
        assert RecordForward(module, module, Materializer(device, BaselineMasks())).autocast
        assert asked == [device]


class TestSteps:
    def test_each_step_is_scored_by_the_loss_computers_device_half(self):
        encoder, model = one_parameter_module(), one_parameter_module()
        seen = []

        def loss(prepared):
            seen.append((prepared, encoder.training, torch.is_grad_enabled()))
            return torch.tensor(float(prepared))

        forward = RecordForward(encoder, model, None)
        scored = list(forward.steps([1, 2, 3], StubLosses(loss)))
        assert [step for step, _ in scored] == [1, 2, 3]
        assert [float(c.shipped) for _, c in scored] == [1.0, 2.0, 3.0]
        assert all(not training and not grad for _, training, grad in seen)
        assert encoder.training

    def test_score_steps_means_the_shipped_loss_and_feeds_every_collector(self):
        added, finished = [], []
        collector = SimpleNamespace(
            wants_parts=False, add=lambda p, c: added.append(p),
            finish=lambda: finished.append(True))
        losses = StubLosses(lambda prepared: StepLoss(
            total=torch.tensor(9.0), shipped=torch.tensor(float(prepared)),
            parts=LossParts(), terms={}))
        module = one_parameter_module()
        value = score_steps([2, 4], RecordForward(module, module, None), losses,
                            (collector,), device=torch.device("cpu"))
        assert value == 3.0
        assert added == [2, 4] and finished == [True]
        assert math.isnan(score_steps([], RecordForward(module, module, None), losses,
                                      device=torch.device("cpu")))


def test_the_probes_predict_through_it(run, records, monkeypatch):
    """``common.predict_all`` reads each chunk's whole per-entity head and its
    verdict head through the one forward, row by row."""
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[4] / "scripts"))
    from effect_knowledge_probes import common

    chosen = records[:10]
    probe_model = SimpleNamespace(encoder=run.encoder, model=run.model, batcher=_batcher(run))
    got = common.predict_all(
        probe_model, chosen, lambda outputs, verdict, surface, row: (
            outputs.shape[0], len(surface.slots), float(verdict[row, 0])), chunk=4)

    assert [size for size, _, _ in got] == [4] * 8 + [2] * 2
    run.encoder.eval()
    run.model.eval()
    with torch.no_grad():
        built = _batcher(run).build(chosen, run.encoder)
        verdict = run.model.verdict(run.model(built.inputs)).float()
    run.encoder.train()
    run.model.train()
    assert [slots for _, slots, _ in got] == [len(s.slots) for s in built.surfaces]
    assert [value for _, _, value in got] == pytest.approx(verdict[:, 0].tolist(), abs=1e-5)
