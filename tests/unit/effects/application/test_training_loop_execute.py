"""What ``TrainingLoop.execute`` wires together, with the tensors stubbed out.

Everything below the loop is replaced — the encoder, the head, the tokenizer,
the sidecars, the loss and both validation passes — so what is left under test
is the order the loop does things in: which field set each pass is handed, when
the stopper forgets its best control score, and what mode the modules are in when an
epoch starts. Those are the joins the per-piece suites cannot see.
"""

from __future__ import annotations

import pytest

from effects.application.step_preparation import StepPreparer
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.application.training_schedule import ControlScoreStopper
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.shard_io import write_shard
from tests.unit.effects.builders import (
    StubLosses,
    make_training_loop,
    stub_below_the_loop,
    stub_validator,
    write_validation_samples,
)


@pytest.fixture
def run(tmp_path, make_record, monkeypatch):
    """Run two epochs and return the ordered log of what the loop called."""
    combat = [make_record(record_id=f"r{i}", game_id=f"g{i % 2}",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(8)]
    shard = tmp_path / "train.jsonl.gz"
    write_shard(shard, combat)
    samples = write_validation_samples(tmp_path, combat[:2])

    model, _encoder = stub_below_the_loop(monkeypatch)
    events: list[tuple] = []

    def device_loss(prepared):
        events.append(("loss", id(prepared["fields"]), model.training))
        return model.p.sum()

    def validate(fields, margins):
        events.append(("validate", id(fields), margins))

    # The step's host half reads real surfaces off real records; what it is
    # handed stands in for what it would have prepared.
    monkeypatch.setattr(
        StepPreparer, "prepare_training_step",
        lambda self, plan, *a, fields=None, **k: ({"fields": fields}, 0),
    )

    resets = []
    original = ControlScoreStopper.reset
    monkeypatch.setattr(
        ControlScoreStopper, "reset",
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
    loop = make_training_loop(config, training_shards=[shard], validation_samples=samples)
    loop.losses = StubLosses(device_loss)
    stub_validator(loop, validate)
    return loop.execute(), events, resets, model


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


def test_empty_validation_samples_are_refused_before_a_model_is_built(tmp_path):
    """No record to measure feature widths from: the CLI maps it to exit 1."""
    from effects.application.train_effect_model import MissingTrainingInputError

    samples = write_validation_samples(tmp_path, [])
    loop = make_training_loop(
        TrainEffectModelConfig(corpus=str(tmp_path), prefetch_workers=0),
        validation_samples=samples,
    )
    with pytest.raises(MissingTrainingInputError, match="nothing to measure feature widths"):
        loop.execute()


def test_the_best_control_score_epoch_is_the_one_checkpoint_and_patience_counts_from_it(
    tmp_path, make_record, monkeypatch,
):
    """Story 8 acceptance 4, FR-069: higher is better; the best epoch's weights
    are the checkpoint, its resume state is copied to resume-best.pt, and
    --patience counts the epochs since without a new best."""
    import torch

    from effects.infrastructure.effect_model_store import EffectModelStore

    combat = [make_record(record_id=f"r{i}", game_id=f"g{i % 2}",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(8)]
    shard = tmp_path / "train.jsonl.gz"
    write_shard(shard, combat)
    samples = write_validation_samples(tmp_path, combat[:2])
    model, _encoder = stub_below_the_loop(monkeypatch)
    monkeypatch.setattr(
        StepPreparer, "prepare_training_step",
        lambda self, plan, *a, fields=None, **k: ({"fields": fields}, 0),
    )
    out = tmp_path / "out"
    config = TrainEffectModelConfig(
        corpus=str(tmp_path), epochs=6, curriculum_epoch=1, steps_per_epoch=2,
        shards_per_epoch=1, batch_size=2, seed=7, model_output=out, prefetch_workers=0,
        patience=2,
    )
    loop = make_training_loop(config, training_shards=[shard], validation_samples=samples)
    loop.losses = StubLosses(lambda prepared: model.p.sum())
    stub_validator(loop, controls=[0.2, 0.5, 0.3, 0.1, 0.9])

    assert loop.execute() == 0
    best = EffectModelStore(out).load()
    assert best.epoch == 2 and best.best_control_score == 0.5
    assert len(list(out.glob("2*.pt"))) == 2          # epochs 1 and 2 set a best
    kept = torch.load(out / "resume-best.pt", weights_only=False)
    assert kept["epoch"] == 2 and kept["best_control_score"] == 0.5
    last = torch.load(out / "resume.pt", weights_only=False)
    # Epochs 3 and 4 set no best: patience 2 stops the run before epoch 5.
    assert last["epoch"] == 4
    assert last["stopper"]["since_best"] == 2
