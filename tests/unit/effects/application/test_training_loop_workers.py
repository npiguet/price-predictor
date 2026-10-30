"""Worker processes prepare the host half; the run is the same run.

The determinism tests train two real epochs end to end on the gen-1 fixture
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

import logging
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from effects.application import run_context
from effects.application import training_loop as loop_module
from effects.application.run_context import preparer_settings
from effects.application.step_loss import StepLossComputer
from effects.application.step_preparation import (
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.train_effect_model import (
    ValidationGameInTrainingError,
    fields_for_epoch,
)
from effects.domain.ability_encoder import AbilityEncoder
from effects.domain.effect_head_input import feature_widths
from effects.domain.effect_model import EffectModel
from effects.domain.sampling import sampling_class
from effects.domain.split import (
    CorpusSplit,
)
from effects.infrastructure.tensor_packing import pack, unpack
from tests.unit.effects.builders import (
    GEN1_TRAINING_SEED,
    gen1_training_config,
    make_training_loop,
)

SEED = GEN1_TRAINING_SEED


class _Interrupted(Exception):
    """Stands in for the machine turning off after an epoch."""


class _RecordedLosses(StepLossComputer):
    """The real device half, noting every loss it computes, training's and
    validation's, and pressing Ctrl+C on the ``interrupt_at``-th."""

    def __init__(self, config, device, sink: list[float], interrupt_at) -> None:
        super().__init__(config, device)
        self.sink = sink
        self.interrupt_at = interrupt_at
        self.calls = 0

    def device_loss(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        computed = super().device_loss(*args, **kwargs)
        self.sink.append(float(computed.total.detach()))
        return computed


def record_losses(loop, sink: list[float], *, interrupt_at=None) -> None:
    """Swap ``loop``'s device half, training's and validation's, for one that
    records into ``sink``."""
    recorded = _RecordedLosses(loop.config, loop.device, sink, interrupt_at)
    loop.losses = loop.validator.losses = recorded


def _run(
    monkeypatch, corpus, out: Path, caplog, *, workers: int, seed: int = SEED,
    interrupt_after: int | None = None, resume: bool = False,
    interrupt_at: int | None = None, **config,
) -> dict:
    """Two epochs end to end; every loss, the final weights, the share lines.

    ``interrupt_after`` stops the run once that epoch's resume file is written,
    as a machine turning off would; ``resume`` continues from that file.
    ``interrupt_at`` presses Ctrl+C on that device loss, counted from 1.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if interrupt_after is not None:
        real_save = loop_module.save_resume

        def save_then_stop(run, epoch, **kwargs):
            real_save(run, epoch, **kwargs)
            if epoch == interrupt_after:
                raise _Interrupted

        monkeypatch.setattr(loop_module, "save_resume", save_then_stop)
    built: dict[str, torch.nn.Module] = {}
    initial: dict[str, torch.Tensor] = {}
    losses: list[float] = []

    def encoder(config):
        built["encoder"] = AbilityEncoder(config)
        return built["encoder"]

    def model(config):
        # The trunk at a size a unit suite can train on a CPU; the workers
        # never build it, so the patch reaches every run alike.
        built["model"] = EffectModel(replace(
            config, d_model=32, n_layers=1, n_heads=2, ff_dim=64,
        ))
        initial.update({
            name: value.detach().clone()
            for name, value in built["model"].state_dict().items()
        })
        return built["model"]

    monkeypatch.setattr(run_context, "AbilityEncoder", encoder)
    monkeypatch.setattr(run_context, "EffectModel", model)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="effects.application"):
        loop = make_training_loop(
            gen1_training_config(
                corpus, out, prefetch_workers=workers, seed=seed, resume=resume,
                **config,
            ),
            split=CorpusSplit(
                held_out_cards=(), card_disjoint_games=frozenset(),
                game_disjoint_games=frozenset({corpus.held}),
            ),
            training_shards=corpus.shards,
            validation_samples={
                "card-disjoint": corpus.sample, "game-disjoint": corpus.sample,
            },
        )
        record_losses(loop, losses, interrupt_at=interrupt_at)
        try:
            code = loop.execute()
        except _Interrupted:
            code = None
    monkeypatch.undo()
    assert code in (0, None)
    return {
        "losses": losses,
        "parameters": {
            f"{owner}.{name}": value.detach().clone()
            for owner, module in built.items()
            for name, value in module.state_dict().items()
        },
        "shares": [m for m in caplog.messages if "by rarity bucket" in m],
        "workers": any("worker process" in m for m in caplog.messages),
        "initial": initial,
    }


def _assert_same(first: dict, second: dict) -> None:
    assert first["losses"] == second["losses"]
    assert first["shares"] == second["shares"]
    assert first["parameters"].keys() == second["parameters"].keys()
    for name, value in first["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name


@pytest.fixture(scope="module")
def in_process(corpus, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        handler = _Capture()
        return _run(
            monkeypatch, corpus, tmp_path_factory.mktemp("serial"), handler,
            workers=0,
        )


class _Capture:
    """``caplog`` for a module-scoped fixture, which cannot request it."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._handler = _ListHandler(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def at_level(self, level, logger):
        import contextlib

        @contextlib.contextmanager
        def scope():
            target = logging.getLogger(logger)
            previous = target.level
            target.setLevel(level)
            target.addHandler(self._handler)
            try:
                yield
            finally:
                target.removeHandler(self._handler)
                target.setLevel(previous)

        return scope()


class _ListHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record) -> None:
        self.sink.append(record.getMessage())


def test_the_fixture_run_exercises_what_it_claims(in_process):
    """Steps trained and both epochs logged."""
    assert len(in_process["losses"]) >= 14
    assert len(in_process["shares"]) == 2


def test_worker_processes_train_exactly_what_the_in_process_path_trains(
    monkeypatch, corpus, tmp_path, caplog, in_process,
):
    workers = _run(monkeypatch, corpus, tmp_path / "two", caplog, workers=2)
    assert workers["workers"] and not in_process["workers"]
    _assert_same(in_process, workers)
    assert not multiprocessing.active_children()


def test_a_seed_repeats_its_run_whatever_the_worker_count(
    monkeypatch, corpus, tmp_path, caplog, in_process,
):
    first = _run(monkeypatch, corpus, tmp_path / "a", caplog, workers=3)
    second = _run(monkeypatch, corpus, tmp_path / "b", caplog, workers=3)
    _assert_same(first, second)
    _assert_same(in_process, first)


def test_another_seed_is_another_run(monkeypatch, corpus, tmp_path, caplog, in_process):
    """So the equalities above are not the vacuous kind."""
    other = _run(
        monkeypatch, corpus, tmp_path / "other", caplog, workers=0, seed=SEED + 1,
    )
    assert other["losses"] != in_process["losses"]
    assert any(
        not torch.equal(value, other["parameters"][name])
        for name, value in in_process["parameters"].items()
    )


# ── resuming an interrupted run ─────────────────────────────────────────


def test_a_run_resumed_after_an_epoch_is_the_uninterrupted_run(
    monkeypatch, corpus, tmp_path, caplog, in_process,
):
    """Interrupted after epoch 1 and resumed, the run ends where it would have.

    Epoch 2 is the curriculum epoch, so the resumed epoch also crosses the
    early stopper's reset and the sparse fields' switch-on.
    """
    out = tmp_path / "interrupted"
    first = _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=1)
    assert (out / "resume.pt").exists()
    second = _run(monkeypatch, corpus, out, caplog, workers=2, resume=True)

    assert first["losses"] + second["losses"] == in_process["losses"]
    assert first["shares"] + second["shares"] == in_process["shares"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    assert not multiprocessing.active_children()


def test_a_resume_file_holding_keys_no_longer_written_resumes_the_same_run(
    monkeypatch, corpus, tmp_path, caplog, in_process,
):
    """Resume files on disk still carry the ``finished`` flag and the noise
    state's ``step``, which nothing read; an interrupted run must continue
    across their removal exactly as it would have."""
    from effects.infrastructure.effect_model_store import (
        load_resume_state,
        save_resume_state,
    )

    out = tmp_path / "older"
    first = _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=1)
    state = load_resume_state(out)
    state["finished"] = False
    state["noise"]["step"] = state["step"]
    save_resume_state(out, state)
    second = _run(monkeypatch, corpus, out, caplog, workers=0, resume=True)

    assert first["losses"] + second["losses"] == in_process["losses"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name


def test_a_run_resumed_for_more_epochs_warms_up_as_the_run_it_continues(
    monkeypatch, corpus, tmp_path, caplog, in_process,
):
    """``--epochs`` may grow on resume, but the warmup was fixed by the run
    that started: at 40 epochs it would still be warming up through epoch 2."""
    out = tmp_path / "extended"
    first = _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=1)
    second = _run(
        monkeypatch, corpus, out, caplog, workers=0, resume=True, epochs=40,
        interrupt_after=2,
    )

    assert first["losses"] + second["losses"] == in_process["losses"]
    for name, value in in_process["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name


def test_a_resume_under_another_seed_is_refused(monkeypatch, corpus, tmp_path, caplog):
    from effects.infrastructure.effect_model_store import ResumeMismatchError

    out = tmp_path / "interrupted"
    _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=1)
    with pytest.raises(ResumeMismatchError, match="seed"):
        _run(monkeypatch, corpus, out, caplog, workers=0, seed=SEED + 1, resume=True)


def test_resuming_a_run_that_reached_its_last_epoch_trains_nothing(
    monkeypatch, corpus, tmp_path, caplog,
):
    out = tmp_path / "finished"
    _run(monkeypatch, corpus, out, caplog, workers=0)
    again = _run(monkeypatch, corpus, out, caplog, workers=0, resume=True)
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
    monkeypatch, corpus, tmp_path, caplog,
):
    out = tmp_path / "stopped"
    _run(monkeypatch, corpus, out, caplog, workers=0)
    _stopped_early(out)
    again = _run(monkeypatch, corpus, out, caplog, workers=0, resume=True, epochs=3)
    assert again["losses"] == []


def test_a_run_stopped_early_continues_with_no_early_stopping(
    monkeypatch, corpus, tmp_path, caplog,
):
    """``--patience 0`` reopens it and trains to ``--epochs`` (FR-063f)."""
    out = tmp_path / "stopped"
    _run(monkeypatch, corpus, out, caplog, workers=0)
    _stopped_early(out)
    again = _run(
        monkeypatch, corpus, out, caplog, workers=0, resume=True, epochs=3, patience=0,
    )
    assert again["losses"]  # it trained epoch 3


def test_every_epoch_is_kept_and_the_best_one_s_resume_state(
    monkeypatch, corpus, tmp_path, caplog,
):
    out = tmp_path / "kept"
    _run(monkeypatch, corpus, out, caplog, workers=0, save_every_epoch=True)
    assert len(list((out / "epochs").glob("2*.pt"))) == 2
    assert (out / "epochs" / "latest.pt").exists()
    assert (out / "resume-best.pt").exists()


def test_a_zero_learning_rate_factor_trains_nothing(monkeypatch, corpus, tmp_path, caplog):
    """The factor reaches every step's rate: at 0 no weight moves (FR-063f)."""
    run = _run(monkeypatch, corpus, tmp_path / "frozen", caplog, workers=0, lr_scale=0.0)
    for name, value in run["initial"].items():
        assert torch.equal(value, run["parameters"][f"model.{name}"]), name


def test_resuming_without_a_resume_file_fails(monkeypatch, corpus, tmp_path, caplog):
    with pytest.raises(FileNotFoundError, match="resume"):
        _run(monkeypatch, corpus, tmp_path / "empty", caplog, workers=0, resume=True)


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
    config = gen1_training_config(corpus, corpus.root / "unused")
    loop = make_training_loop(config, rarity={"x": 3})
    settings = preparer_settings(loop.spec, widths=feature_widths(records))
    assert settings.heads == frozenset({"value", "api", "mlm"})
    fields = fields_for_epoch(
        config, present=frozenset(sampling_class(r) for r in records), epoch=1,
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
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_tensors_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tensors_equal(a, b, f"{path}[{index}]")
    elif hasattr(left, "__dataclass_fields__"):
        for name in left.__dataclass_fields__:
            _assert_tensors_equal(
                getattr(left, name), getattr(right, name), f"{path}.{name}",
            )
    else:
        assert left == right, path


def _with_shards(corpus, shards):
    return type("Corpus", (), {
        **{k: v for k, v in vars(corpus).items() if not k.startswith("__")},
        "shards": shards,
    })


def test_a_worker_exception_surfaces_with_its_traceback_and_no_process_survives(
    monkeypatch, corpus, tmp_path, caplog,
):
    """An unreadable shard: raised at its own shard, in the training process,
    as the exception the in-process path raises, the worker's traceback
    attached."""
    missing = tmp_path / "missing.jsonl.gz"
    with pytest.raises(FileNotFoundError) as raised:
        _run(monkeypatch, _with_shards(corpus, [missing]), tmp_path / "out", caplog, workers=2)
    notes = "".join(getattr(raised.value, "__notes__", ()))
    assert "Traceback" in notes and "step-preparation worker" in notes
    assert "missing.jsonl.gz" in str(raised.value)
    assert any("preparing it failed" in m for m in caplog.messages)
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("workers", [0, 2])
def test_a_training_shard_holding_a_validation_game_is_refused(
    monkeypatch, corpus, records, tmp_path, caplog, workers,
):
    """The manifest places the game in a validation stratum, which build-corpus
    never writes into training/: the shard and the manifest disagree, and
    training around the game would hide it. Refused under any worker count,
    as the exception the CLI maps to exit 1."""
    from effects.infrastructure.shard_io import write_shard

    tainted = tmp_path / "tainted.jsonl.gz"
    write_shard(tainted, records)
    with pytest.raises(ValidationGameInTrainingError, match=corpus.held):
        _run(
            monkeypatch, _with_shards(corpus, [tainted]), tmp_path / "out", caplog,
            workers=workers,
        )
    assert not multiprocessing.active_children()


def test_an_interrupt_mid_shard_leaves_no_process_behind(
    monkeypatch, corpus, tmp_path, caplog,
):
    """Ctrl+C lands in the training loop with workers blocked on full queues."""
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, corpus, tmp_path / "out", caplog, workers=2, interrupt_at=3)
    assert not multiprocessing.active_children()
