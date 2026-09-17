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

from effects.application import training_loop as loop_module
from effects.application.train_effect_model import (
    EarlyStopper,
    HeldOutCards,
    TrainEffectModelConfig,
)
from effects.application.training_loop import TrainingLoop
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.record_io import write_shard
from effects.infrastructure.sidecar_io import SidecarCache
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


class _Metrics:
    affected_gate_f1 = 0.5
    zone_outcome_accuracy = 0.5
    mean_poisson_deviance = 0.5


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
        TrainingLoop, "_build_tokenizer",
        lambda self: type("T", (), {"vocab_size": 8})(),
    )
    monkeypatch.setattr(TrainingLoop, "_build_sidecars", lambda self: SidecarCache({}))
    monkeypatch.setattr(
        TrainingLoop, "_feature_widths",
        lambda self, records: dict.fromkeys(loop_module.SlotKind, 4),
    )

    def loss_for(self, plan, *args, fields=None, **kwargs):
        events.append(("loss", id(fields), model.training))
        return model.p.sum(), {}

    def validate(self, records, *args, parts=None, fields=None, **kwargs):
        events.append(("validate", id(fields), model.training))
        return 1.0

    def measure(records, enc, mdl, batcher, *, fields):
        events.append(("measure", id(fields), batcher.keyword_expand_p))
        # The real one scores in eval mode and never switches back, which is
        # what the next epoch's first step has to survive.
        enc.eval()
        mdl.eval()
        return _Metrics()

    monkeypatch.setattr(TrainingLoop, "_loss_for", loss_for)
    monkeypatch.setattr(TrainingLoop, "_validate", validate)
    monkeypatch.setattr(loop_module, "measure", measure)

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
    )
    code = TrainingLoop(
        config, held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[shard], validation_samples=samples,
        holdout_permille=20, holdout_max_carriers=8, gate_one_records=0,
        rarity={}, corpus_digest="",
    ).execute()
    return code, events, resets, model


def test_the_run_completes(run):
    code, _, _, _ = run
    assert code == 0


def test_an_epoch_scores_the_field_set_it_trained_with(run):
    """Every pass of an epoch is handed the same object (FR-082).

    Not merely an equal one: the training loss, the two validation strata and
    the gate-1 probe all read ``fields`` by identity from the epoch, so a pass
    that rebuilt its own would be scoring a different objective from the number
    printed beside it.
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
        assert kinds == {"loss", "validate", "measure"}


def test_the_stopper_forgets_its_best_once_at_the_curriculum_boundary(run):
    _, _, resets, _ = run
    assert len(resets) == 1


def test_the_gate_one_probe_runs_without_the_augmentations(run):
    _, events, _, _ = run
    expansions = [payload for kind, _, payload in events if kind == "measure"]
    assert expansions == [0.0, 0.0]


def test_the_second_epoch_starts_in_training_mode(run):
    """``measure`` scores in eval mode and never switches back.

    Epoch 2 would then train a model whose dropout and noise are off, and
    nothing in the log would say so.
    """
    _, events, _, model = run
    assert all(payload for kind, _, payload in events if kind == "loss")
    # And the loop leaves the modules the way an epoch needs to find them.
    assert model.training
