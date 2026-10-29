"""What ``TrainingLoop.execute`` wires together, with the tensors stubbed out.

Everything below the loop is replaced — the encoder, the head, the tokenizer,
the sidecars, the loss and both validation passes — so what is left under test
is the order the loop does things in: which field set each pass is handed, when
the early stopper forgets its best, and what mode the modules are in when an
epoch starts. Those are the joins the per-piece suites cannot see.
"""

from __future__ import annotations

import pytest
import torch

from effects.application import step_preparation
from effects.application import training_loop as loop_module
from effects.application.step_preparation import StepPreparer
from effects.application.train_effect_model import (
    EarlyStopper,
    TrainEffectModelConfig,
)
from effects.application.training_loop import StepLoss, TrainingLoop
from effects.domain.effect_losses import LossParts
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.builders import make_training_loop
from tests.unit.effects.domain.conftest import (  # noqa: F401
    ability_key,
    make_record,
    snapshot,
)


class _Stub(torch.nn.Module):
    """A module with one parameter, so backward and the optimizer have work."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2))


@pytest.fixture
def run(tmp_path, make_record, monkeypatch):  # noqa: F811
    """Run two epochs and return the ordered log of what the loop called."""
    combat = [make_record(record_id=f"r{i}", game_id=f"g{i % 2}",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(8)]
    shard = tmp_path / "train.jsonl.gz"
    write_shard(shard, combat)
    samples = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        samples[stratum] = tmp_path / f"{stratum}.jsonl.gz"
        write_shard(samples[stratum], combat[:2])

    model, encoder = _Stub(), _Stub()
    events: list[tuple] = []

    monkeypatch.setattr(loop_module, "AbilityEncoder", lambda config: encoder)
    monkeypatch.setattr(loop_module, "EffectModel", lambda config: model)
    monkeypatch.setattr(
        step_preparation, "load_tokenizer",
        lambda *a, **k: type("T", (), {"vocab_size": 8})(),
    )
    # The script-API vocabulary is read off every real sidecar; none here.
    monkeypatch.setattr(TrainingLoop, "_script_vocabularies", lambda self: ([], []))
    monkeypatch.setattr(
        loop_module, "feature_widths",
        lambda records: dict.fromkeys(loop_module.SlotKind, 4),
    )

    def device_loss(self, prepared, *args, **kwargs):
        events.append(("loss", id(prepared["fields"]), model.training))
        loss = model.p.sum()
        return StepLoss(total=loss, shipped=loss, parts=LossParts(), terms={})

    def validate(self, records, *args, parts=None, fields=None, margins=None, **kwargs):
        events.append(("validate", id(fields), margins is not None))
        return 1.0

    monkeypatch.setattr(TrainingLoop, "_device_loss", device_loss)
    # The step's host half reads real surfaces off real records; what it is
    # handed stands in for what it would have prepared.
    monkeypatch.setattr(
        StepPreparer, "prepare_step",
        lambda self, plan, *a, fields=None, **k: {"fields": fields},
    )
    monkeypatch.setattr(TrainingLoop, "_validate", validate)

    resets = []
    original = EarlyStopper.reset
    monkeypatch.setattr(
        EarlyStopper, "reset",
        lambda self: (resets.append(1), original(self))[1],
    )

    config = TrainEffectModelConfig(
        corpus=str(tmp_path), epochs=2, curriculum_epoch=2, steps_per_epoch=2,
        shards_per_epoch=1, batch_size=2, seed=7, keyword_expand_p=0.25,
        model_output=tmp_path / "out",
        # The stubbed halves live in this process; a worker would run the
        # real ones.
        prefetch_workers=0,
    )
    code = make_training_loop(config, training_shards=[shard], validation_samples=samples).execute()
    return code, events, resets, model


def test_the_run_completes(run):
    code, _, _, _ = run
    assert code == 0


def test_an_epoch_scores_the_field_set_it_trained_with(run):
    """Every pass of an epoch is handed the same object (FR-082).

    Not merely an equal one: the training loss and the two validation strata
    read ``fields`` by identity from the epoch, so a pass that rebuilt its own
    would be scoring a different objective from the number printed beside it.
    """
    _, events, _, _ = run
    seen = [fields for _kind, fields, _payload in events]
    runs = [key for index, key in enumerate(seen) if index == 0 or key != seen[index - 1]]
    # Two epochs, each handing one object to every pass it makes, and the
    # curriculum boundary is the only place the object changes.
    assert len(runs) == 2
    assert len(set(runs)) == 2
    for epoch_fields in runs:
        kinds = {kind for kind, fields, _ in events if fields == epoch_fields}
        assert kinds == {"loss", "validate"}


def test_the_stopper_forgets_its_best_once_at_the_curriculum_boundary(run):
    _, _, resets, _ = run
    assert len(resets) == 1


def test_the_gate_one_margins_ride_the_card_disjoint_pass_alone(run):
    """One validation pass per epoch feeds the progress margins, and no second
    pass over the sample scores them."""
    _, events, _, _ = run
    fed = [payload for kind, _, payload in events if kind == "validate"]
    assert fed == [True, False, True, False]


def test_the_second_epoch_starts_in_training_mode(run):
    """Epoch 2 would otherwise train a model whose dropout and noise are off,
    and nothing in the log would say so."""
    _, events, _, model = run
    assert all(payload for kind, _, payload in events if kind == "loss")
    # And the loop leaves the modules the way an epoch needs to find them.
    assert model.training
