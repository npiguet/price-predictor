"""Worker processes prepare the host half; the run is the same run.

The determinism tests train two real epochs end to end on the gen-3 fixture
records — split into three shards, with every draw the host half takes switched
on and a validation game the split names, which no training shard holds — once with
the host half in this process and twice with it in worker processes. A step's
draws are seeded by its place in the run, so where it was prepared, by how many
workers, must not move a single parameter; and two runs with one seed must be
one run.

The rest pin the plumbing: a packed step unpacks to the tensors it was packed
from, a step prepared the way a worker prepares it equals one prepared here,
and a worker's exception reaches the training process with its traceback and
leaves no process behind.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from effects.application import training_loop as loop_module
from effects.application.run_context import preparer_settings
from effects.application.step_preparation import (
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.train_effect_model import ValidationGameInTrainingError
from effects.application.training_schedule import Curriculum
from effects.domain.sampling import sampling_class
from effects.infrastructure.tensor_packing import pack, unpack
from tests.unit.effects.application.training_run_support import (
    SEED,
    Capture,
    RunPlan,
    TrainingRunner,
    every_record_selection,
)
from tests.unit.effects.builders import (
    fixture_training_config,
    make_training_loop,
)

pytestmark = pytest.mark.usefixtures("single_thread")


def _assert_same(first: dict, second: dict) -> None:
    assert first["losses"] == second["losses"]
    assert first["shares"] == second["shares"]
    assert first["control"] == second["control"]
    assert first["parameters"].keys() == second["parameters"].keys()
    for name, value in first["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    _assert_same_average(first["average"], second["average"])


def _assert_same_average(first: dict, second: dict) -> None:
    assert first["start_step"] == second["start_step"]
    assert first["state"].keys() == second["state"].keys()
    for owner, tensors in first["state"].items():
        assert tensors.keys() == second["state"][owner].keys()
        for name, value in tensors.items():
            assert torch.equal(value, second["state"][owner][name]), f"{owner}.{name}"


@pytest.fixture(scope="module")
def in_process(single_thread, corpus, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        runner = TrainingRunner(monkeypatch, corpus, Capture())
        return runner.run(tmp_path_factory.mktemp("serial"), RunPlan(workers=0))


def test_the_fixture_run_exercises_what_it_claims(in_process):
    """Steps trained, both epochs logged, and each scored by the control
    score: per stratum its four components and score, then control and best
    (FR-075)."""
    assert len(in_process["losses"]) >= 12
    assert len(in_process["shares"]) == 2
    assert len(in_process["control"]) == 2
    for line in in_process["control"]:
        for stratum in ("card-disjoint", "game-disjoint"):
            assert f"{stratum} probes " in line
        for name in ("grids", "rule_tests", "deviance", "score", "| control", "| best"):
            assert name in line


def test_worker_processes_train_exactly_what_the_in_process_path_trains(
    runner, tmp_path, in_process,
):
    workers = runner.run(tmp_path / "two", RunPlan(workers=2))
    assert workers["workers"] and not in_process["workers"]
    _assert_same(in_process, workers)
    assert not multiprocessing.active_children()


def test_a_seed_repeats_its_run_whatever_the_worker_count(
    runner, tmp_path, in_process,
):
    first = runner.run(tmp_path / "a", RunPlan(workers=3))
    second = runner.run(tmp_path / "b", RunPlan(workers=3))
    _assert_same(first, second)
    _assert_same(in_process, first)


def test_another_seed_is_another_run(runner, tmp_path, in_process):
    """So the equalities above are not the vacuous kind."""
    other = runner.run(tmp_path / "other", RunPlan(workers=0, seed=SEED + 1))
    assert other["losses"] != in_process["losses"]
    assert any(
        not torch.equal(value, other["parameters"][name])
        for name, value in in_process["parameters"].items()
    )


# ── resuming an interrupted run ─────────────────────────────────────────


def test_a_run_resumed_after_an_epoch_is_the_uninterrupted_run(
    runner, tmp_path, in_process,
):
    """Interrupted after epoch 1 and resumed, the run ends where it would have.

    Epoch 2 is the curriculum epoch, so the resumed epoch also crosses the
    early stopper's reset and the sparse fields' switch-on.
    """
    out = tmp_path / "interrupted"
    first = runner.run(out, RunPlan(workers=0, interrupt_after=1))
    assert (out / "resume.pt").exists()
    second = runner.run(out, RunPlan(workers=2, resume=True))

    assert first["losses"] + second["losses"] == in_process["losses"]
    assert first["shares"] + second["shares"] == in_process["shares"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    # The averaged weights too, which validation read: the losses above
    # include both epochs' validation passes.
    _assert_same_average(in_process["average"], second["average"])
    assert not multiprocessing.active_children()


def test_a_resume_reuses_the_loss_denominators_the_run_started_with(
    runner, tmp_path,
):
    """FR-062: computed once at startup from the training shards, floored,
    kept in ``resume.pt`` and the checkpoint, and read back on resume rather
    than recomputed."""
    from effects.infrastructure.effect_model_store import EffectModelStore, load_resume_state

    out = tmp_path / "denominators"
    runner.run(out, RunPlan(workers=0, interrupt_after=1))
    assert any("Loss denominators from" in m for m in runner.caplog.messages)
    stored = load_resume_state(out)["loss_denominators"]
    values = [v for terms in stored["values"].values() for v in terms.values()]
    assert values and min(values) >= stored["floor"] == 0.04
    assert "gate" in stored["values"]["per_entity"]

    runner.run(out, RunPlan(workers=0, resume=True))
    assert not any("Loss denominators from" in m for m in runner.caplog.messages)
    assert load_resume_state(out)["loss_denominators"] == stored
    assert EffectModelStore(out).load().loss_denominators == stored


def test_a_resume_file_holding_keys_no_longer_written_resumes_the_same_run(
    runner, tmp_path, in_process,
):
    """Resume files on disk still carry the ``finished`` flag and the noise
    state's ``step``, which nothing read; an interrupted run must continue
    across their removal exactly as it would have."""
    from effects.infrastructure.effect_model_store import (
        load_resume_state,
        save_resume_state,
    )

    out = tmp_path / "older"
    first = runner.run(out, RunPlan(workers=0, interrupt_after=1))
    state = load_resume_state(out)
    state["finished"] = False
    state["noise"]["step"] = state["step"]
    save_resume_state(out, state)
    second = runner.run(out, RunPlan(workers=0, resume=True))

    assert first["losses"] + second["losses"] == in_process["losses"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name


def test_a_run_resumed_for_more_epochs_warms_up_as_the_run_it_continues(
    runner, tmp_path, in_process,
):
    """``--epochs`` may grow on resume, but the warmup was fixed by the run
    that started: at 40 epochs it would still be warming up through epoch 2."""
    out = tmp_path / "extended"
    first = runner.run(out, RunPlan(workers=0, interrupt_after=1))
    second = runner.run(out, RunPlan(
        workers=0, resume=True, interrupt_after=2, config={"epochs": 40},
    ))

    assert first["losses"] + second["losses"] == in_process["losses"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name


# ── the averaged weights (research T4, story 7 acceptance 4) ─────────────────────


def test_the_average_starts_one_epoch_after_the_curriculum_switch(in_process):
    """The fixture's warmup is one step and its switch step 6, so the average
    starts at step 12, epoch 2's last, from that step's weights.
    ``test_curriculum_resume.py`` sees it move in the epoch after."""
    average = in_process["average"]
    assert average["start_step"] == 12
    assert set(average["state"]) == {"encoder", "model"}
    for owner, tensors in average["state"].items():
        for name, value in tensors.items():
            assert torch.equal(value, in_process["parameters"][f"{owner}.{name}"]), name


def test_a_saved_checkpoint_holds_the_averaged_weights(runner, tmp_path):
    from effects.infrastructure.effect_model_store import EffectModelStore

    out = tmp_path / "averaged"
    run = runner.run(out, RunPlan(workers=0, config={"save_every_epoch": True}))
    saved = EffectModelStore(out / "epochs").load(out / "epochs" / "epoch-02.pt")
    averaged = run["average"]["state"]
    for name, value in saved.model_state.items():
        assert torch.equal(value, averaged["model"][name]), name
    for name, value in saved.encoder_state.items():
        assert torch.equal(value, averaged["encoder"][name]), name
    assert saved.training_settings["ema_epochs"] == 1.0 and saved.scored_under().mask == "hard"
    # FR-069: the epoch's own control score, and the set it was scored against.
    state = run["stopper"]
    assert saved.best_control_score == state["best"]
    assert saved.averaged and saved.control_score is not None
    assert saved.selection_set_digest == every_record_selection().digest


def test_an_epoch_ending_before_warmup_is_no_best_candidate(
    runner, tmp_path,
):
    """Story 7 acceptance 4: at 40 epochs of 6 steps warmup lasts 12 steps,
    so epoch 1 is validated on the current weights, saves no best checkpoint
    and leaves the stopper's patience untouched."""
    from effects.infrastructure.effect_model_store import load_resume_state

    out = tmp_path / "warming"
    run = runner.run(out, RunPlan(workers=0, interrupt_after=1, config={"epochs": 40}))
    assert run["losses"]
    state = load_resume_state(out)
    assert state["warmup"] == 12
    assert state["weight_average"] == {"state": None, "start_step": None}
    assert state["stopper"] == {
        "measure": "control_score", "best": float("-inf"), "since_best": 0}
    assert state["best_control_score"] is None
    assert not (out / "latest.pt").exists() and not (out / "resume-best.pt").exists()


def test_a_resume_across_warmup_starts_the_average_at_warmups_end(
    runner, tmp_path,
):
    """Interrupted while warming up, the resumed run starts the average where
    the uninterrupted run does."""
    whole = runner.run(tmp_path / "whole", RunPlan(
        workers=0, interrupt_after=3, config={"epochs": 40},
    ))
    out = tmp_path / "split"
    runner.run(out, RunPlan(workers=0, interrupt_after=1, config={"epochs": 40}))
    resumed = runner.run(out, RunPlan(
        workers=0, resume=True, interrupt_after=3, config={"epochs": 40},
    ))
    assert whole["average"]["start_step"] == resumed["average"]["start_step"] == 12
    _assert_same_average(whole["average"], resumed["average"])


@pytest.mark.parametrize("flag, value", [
    ("rule_mask", "soft"), ("deviance_floor", 0.05), ("ema_epochs", 2.0),
    ("stratum_weight", 0.25), ("relation_values", False),
])
def test_a_resume_changing_a_training_objective_flag_is_refused(
    runner, tmp_path, flag, value,
):
    """Story 7 acceptance 5, for each flag contracts/cli.md lists."""
    from effects.domain.rule_mask import RuleMask
    from effects.infrastructure.effect_model_store import ResumeMismatchError

    if flag == "rule_mask":
        value = RuleMask(value)
    out = tmp_path / "interrupted"
    runner.run(out, RunPlan(workers=0, interrupt_after=1))
    with pytest.raises(ResumeMismatchError, match=flag):
        runner.run(out, RunPlan(workers=0, resume=True, config={flag: value}))


def test_a_resume_under_another_seed_is_refused(runner, tmp_path):
    from effects.infrastructure.effect_model_store import ResumeMismatchError

    out = tmp_path / "interrupted"
    runner.run(out, RunPlan(workers=0, interrupt_after=1))
    with pytest.raises(ResumeMismatchError, match="seed"):
        runner.run(out, RunPlan(workers=0, seed=SEED + 1, resume=True))


def test_resuming_a_run_that_reached_its_last_epoch_trains_nothing(
    runner, tmp_path,
):
    out = tmp_path / "finished"
    runner.run(out, RunPlan(workers=0))
    again = runner.run(out, RunPlan(workers=0, resume=True))
    assert again["losses"] == []


def _stopped_early(out: Path) -> None:
    """Rewrite a finished run's resume file as one the early stopper ended."""
    from effects.infrastructure.effect_model_store import (
        load_resume_state,
        save_resume_state,
    )

    state = load_resume_state(out)
    state["stopper"]["since_best"] = 5
    save_resume_state(out, state)


def test_a_run_stopped_early_stays_stopped_under_the_same_patience(
    runner, tmp_path,
):
    out = tmp_path / "stopped"
    runner.run(out, RunPlan(workers=0))
    _stopped_early(out)
    again = runner.run(out, RunPlan(workers=0, resume=True, config={"epochs": 3}))
    assert again["losses"] == []


def test_a_run_stopped_early_continues_with_no_early_stopping(
    runner, tmp_path,
):
    """``--patience 0`` reopens it and trains to ``--epochs`` (FR-063f)."""
    out = tmp_path / "stopped"
    runner.run(out, RunPlan(workers=0))
    _stopped_early(out)
    again = runner.run(out, RunPlan(
        workers=0, resume=True, config={"epochs": 3, "patience": 0},
    ))
    assert again["losses"]  # it trained epoch 3


def test_every_epoch_is_kept_and_the_best_one_s_resume_state(
    runner, tmp_path,
):
    out = tmp_path / "kept"
    runner.run(out, RunPlan(workers=0, config={"save_every_epoch": True}))
    assert sorted(p.name for p in (out / "epochs").iterdir()) == [
        "epoch-01.pt", "epoch-02.pt"]
    assert (out / "resume-best.pt").exists()


def test_a_zero_learning_rate_factor_trains_nothing(runner, tmp_path):
    """The factor reaches every step's rate: at 0 no weight moves (FR-063f)."""
    run = runner.run(tmp_path / "frozen", RunPlan(workers=0, config={"lr_scale": 0.0}))
    for name, value in run["initial"].items():
        assert torch.equal(value, run["parameters"][f"model.{name}"]), name


def test_resuming_without_a_resume_file_fails(runner, tmp_path):
    with pytest.raises(FileNotFoundError, match="resume"):
        runner.run(tmp_path / "empty", RunPlan(workers=0, resume=True))


# ── the plumbing ────────────────────────────────────────────────────────


def test_a_packed_step_unpacks_to_the_tensors_it_was_packed_from():
    """Every dtype a step carries, a scalar, an empty tensor and a view."""
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    message = StepReady(
        prepared=None, buckets={"1": 3}, families={"Draw": 3},
    )
    payload = {
        "float": torch.randn(3, 5),
        "long": torch.arange(7),
        "bool": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.5),
        "empty": torch.zeros(0, 4, dtype=torch.long),
        "view": base[:, 1::2],
        "nested": [(torch.ones(2, dtype=torch.int32), "text")],
    }
    message.prepared = payload  # any picklable structure travels the same way
    packed = pack(message)
    # One buffer, however many tensors: the cost of crossing a process is
    # paid per tensor on Windows.
    assert isinstance(packed.flat, torch.Tensor) and packed.flat.dim() == 1
    restored = unpack(packed)
    assert restored.buckets == {"1": 3} and restored.families == {"Draw": 3}
    for key in ("float", "long", "bool", "scalar", "empty", "view"):
        got, want = restored.prepared[key], payload[key]
        assert got.dtype == want.dtype and torch.equal(got, want), key
    assert torch.equal(restored.prepared["nested"][0][0], torch.ones(2, dtype=torch.int32))
    assert restored.prepared["nested"][0][1] == "text"


def test_a_step_prepared_as_a_worker_prepares_it_equals_one_prepared_here(
    monkeypatch, corpus, records,
):
    """The worker's resources, built from settings, and the packing between."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = fixture_training_config(corpus, corpus.root / "unused")
    loop = make_training_loop(config, rarity={"x": 3})
    settings = preparer_settings(loop.spec)
    assert settings.heads == frozenset({"value", "api", "mlm"})
    fields = Curriculum(config.curriculum_epoch).fields_for(
        1, frozenset(sampling_class(r) for r in records),
    )
    task = ShardTask(
        shard=Path("fixture"), epoch=1, position=2, budget=5, fields=fields,
    )
    here = list(StepPreparer(settings, truncations=loop.reporter.truncations).shard_steps(
        task, records,
    ))
    worker = StepPreparer(settings)
    there = [
        unpack(pack(m)) if isinstance(m, StepReady) else m
        for m in worker.steps(task, records)
    ]
    assert isinstance(here[0], ShardOpened) and here[0] == there[0]
    assert len(here) == 1 + task.budget and len(there) == 2 + task.budget
    for mine, theirs in zip(here[1:], there[1:-1]):
        assert mine.buckets == theirs.buckets and mine.families == theirs.families
        _assert_tensors_equal(mine.prepared, theirs.prepared)


def _assert_tensors_equal(left, right, path="step") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and torch.equal(left, right), path
        return
    for part_left, part_right, part_path in _paired_parts(left, right, path):
        _assert_tensors_equal(part_left, part_right, part_path)


def _paired_parts(left, right, path: str) -> list[tuple[Any, Any, str]]:
    """The parts of two non-tensor values, pairwise with their paths; a value
    with no parts is compared whole."""
    if isinstance(left, dict):
        return _mapping_parts(left, right, path)
    if isinstance(left, (list, tuple)):
        return _sequence_parts(left, right, path)
    if hasattr(left, "__dataclass_fields__"):
        return [
            (getattr(left, name), getattr(right, name), f"{path}.{name}")
            for name in left.__dataclass_fields__
        ]
    assert left == right, path
    return []


def _mapping_parts(left: dict, right: dict, path: str) -> list[tuple[Any, Any, str]]:
    assert left.keys() == right.keys(), path
    return [(left[key], right[key], f"{path}.{key}") for key in left]


def _sequence_parts(left, right, path: str) -> list[tuple[Any, Any, str]]:
    assert len(left) == len(right), path
    return [(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right))]


def _with_shards(corpus, shards):
    return type("Corpus", (), {
        **{k: v for k, v in vars(corpus).items() if not k.startswith("__")},
        "shards": shards,
    })


def test_a_worker_exception_surfaces_with_its_traceback_and_no_process_survives(
    runner, tmp_path,
):
    """An unreadable shard: raised at its own shard, in the training process,
    as the exception the in-process path raises, the worker's traceback
    attached."""
    from effects.domain.loss_denominators import LossDenominators

    missing = tmp_path / "missing.jsonl.gz"
    # The startup pre-pass would read the shard first, in this process.
    runner.monkeypatch.setattr(
        loop_module, "loss_denominators",
        lambda *a, floor, **k: LossDenominators.of({}, floor=floor),
    )
    with pytest.raises(FileNotFoundError) as raised:
        replace(runner, corpus=_with_shards(runner.corpus, [missing])).run(
            tmp_path / "out", RunPlan(workers=2),
        )
    notes = "".join(getattr(raised.value, "__notes__", ()))
    assert "Traceback" in notes and "step-preparation worker" in notes
    assert "missing.jsonl.gz" in str(raised.value)
    assert any("preparing it failed" in m for m in runner.caplog.messages)
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("workers", [0, 2])
def test_a_training_shard_holding_a_validation_game_is_refused(
    runner, records, tmp_path, workers,
):
    """The manifest places the game in a validation stratum, which build-corpus
    never writes into training/: the shard and the manifest disagree, and
    training around the game would hide it. Refused under any worker count,
    as the exception the CLI maps to exit 1."""
    from effects.infrastructure.shard_io import write_shard

    tainted = tmp_path / "tainted.jsonl.gz"
    write_shard(tainted, records)
    with pytest.raises(ValidationGameInTrainingError, match=runner.corpus.held):
        replace(runner, corpus=_with_shards(runner.corpus, [tainted])).run(
            tmp_path / "out", RunPlan(workers=workers),
        )
    assert not multiprocessing.active_children()


def test_an_interrupt_mid_shard_leaves_no_process_behind(runner, tmp_path):
    """Ctrl+C lands in the training loop with workers blocked on full queues."""
    with pytest.raises(KeyboardInterrupt):
        runner.run(tmp_path / "out", RunPlan(workers=2, interrupt_at=3))
    assert not multiprocessing.active_children()
