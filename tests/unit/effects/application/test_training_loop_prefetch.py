"""The shard prefetch: shard k+1 is read while shard k takes its steps.

Everything below the epoch loop is stubbed — the loader, the per-shard training
call, the encoder, the head and both validation passes — so what is left under
test is when the reads happen relative to the steps, and that moving them
changed neither the order the shards are trained in nor which records each one
is trained on. Both fakes sleep, so the assertions read a clock rather than a
call order: an implementation that submitted the next load and then waited for
it before stepping would pass an ordering check and overlap nothing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from effects.application import training_loop as loop_module
from effects.application.train_effect_model import (
    HeldOutCards,
    TrainEffectModelConfig,
    epoch_shards,
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

#: Shards in the epoch, and the seed whose draw orders them.
SHARDS = 4
SEED = 7
#: What a fake load and a fake shard of training each cost. Long enough that
#: an overlap is unambiguous on a loaded machine, short enough to pay four
#: times in a unit suite.
PAUSE = 0.05


class _Stub(torch.nn.Module):
    """A module with one parameter, so the optimizer has something to hold."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2))


class _Metrics:
    affected_gate_f1 = 0.5
    zone_outcome_accuracy = 0.5
    mean_poisson_deviance = 0.5


class _SpyPool:
    """A one-thread pool that remembers how it was built and shut down."""

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdowns = 0
        self.inner = ThreadPoolExecutor(*args, **kwargs)

    def submit(self, fn, *args, **kwargs):
        return self.inner.submit(fn, *args, **kwargs)

    def shutdown(self, *args, **kwargs):
        self.shutdowns += 1
        return self.inner.shutdown(*args, **kwargs)


@pytest.fixture
def run(tmp_path, make_record, monkeypatch):  # noqa: F811
    """Run one four-shard epoch and return what happened, and when."""
    combat = [make_record(record_id=f"r{i}", game_id=f"g{i % 2}",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(4)]
    samples = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        samples[stratum] = tmp_path / f"{stratum}.jsonl.gz"
        write_shard(samples[stratum], combat[:2])
    shards = [tmp_path / f"train{index}.jsonl.gz" for index in range(SHARDS)]

    real_load = loop_module.load_shard
    marks: list[tuple[str, str, str, float]] = []
    trained: list[tuple[str, object]] = []
    pools: list[_SpyPool] = []

    def mark(what: str, which: str, phase: str) -> None:
        marks.append((what, which, phase, time.perf_counter()))

    def start(*, failing: str | None = None) -> int:
        def load(path):
            path = Path(path)
            if path in set(samples.values()):
                return real_load(path)
            mark("load", path.name, "start")
            time.sleep(PAUSE)
            if path.name == failing:
                raise RuntimeError(f"unreadable shard {path.name}")
            mark("load", path.name, "finish")
            return [f"{path.name} records"]

        def train_on_shard(self, shard, budget, records, waited, *,
                           step, taken, **kwargs):
            mark("train", shard.name, "start")
            time.sleep(PAUSE)
            trained.append((shard.name, records))
            mark("train", shard.name, "finish")
            return step + budget, taken + budget

        def pool(*args, **kwargs):
            pools.append(_SpyPool(*args, **kwargs))
            return pools[-1]

        def measure(records, encoder, model, batcher, *, fields):
            return _Metrics()

        model, encoder = _Stub(), _Stub()
        monkeypatch.setattr(loop_module, "load_shard", load)
        # ``raising=False`` so a loop with no pool at all still runs and fails
        # on what it did rather than on the spy having nothing to replace.
        monkeypatch.setattr(
            loop_module, "ThreadPoolExecutor", pool, raising=False,
        )
        monkeypatch.setattr(loop_module, "measure", measure)
        monkeypatch.setattr(loop_module, "AbilityEncoder", lambda config: encoder)
        monkeypatch.setattr(loop_module, "EffectModel", lambda config: model)
        monkeypatch.setattr(TrainingLoop, "_train_on_shard", train_on_shard)
        monkeypatch.setattr(
            TrainingLoop, "_validate",
            lambda self, records, *args, **kwargs: 1.0,
        )
        monkeypatch.setattr(
            TrainingLoop, "_build_tokenizer",
            lambda self: type("T", (), {"vocab_size": 8})(),
        )
        monkeypatch.setattr(
            TrainingLoop, "_build_sidecars", lambda self: SidecarCache({}),
        )
        monkeypatch.setattr(
            TrainingLoop, "_feature_widths",
            lambda self, records: dict.fromkeys(loop_module.SlotKind, 4),
        )
        config = TrainEffectModelConfig(
            corpus=str(tmp_path), epochs=1, steps_per_epoch=SHARDS,
            shards_per_epoch=SHARDS, batch_size=2, seed=SEED,
            model_output=tmp_path / "out",
        )
        return TrainingLoop(
            config,
            held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
            inherited=None, training_shards=shards, validation_samples=samples,
            holdout_permille=20, holdout_max_carriers=8, gate_one_records=0,
            rarity={}, corpus_digest="",
        ).execute()

    drawn = epoch_shards(shards, epoch=1, per_epoch=SHARDS, seed=SEED)
    return type(
        "Harness", (),
        {"start": staticmethod(start), "marks": marks, "trained": trained,
         "pools": pools, "drawn": [path.name for path in drawn]},
    )


def _at(marks, what: str, which: str, phase: str) -> float:
    """When the one mark of this kind, shard and phase was taken."""
    return next(when for kind, name, stage, when in marks
                if kind == what and name == which and stage == phase)


def test_each_shard_loads_while_its_predecessor_trains(run):
    """The overlap itself: the read of k+1 begins before k stops stepping.

    A shard costs about 1.3 s of gzip and JSON to read, against a shard of
    steps that now takes seconds, so a serial read is the GPU standing still
    for about a sixth of the epoch.
    """
    assert run.start() == 0
    for current, following in zip(run.drawn, run.drawn[1:]):
        assert _at(run.marks, "load", following, "start") < _at(
            run.marks, "train", current, "finish",
        )


def test_the_shards_are_trained_in_the_drawn_order_with_their_own_records(run):
    """The prefetch moves when a shard is read, never which shard or what.

    The seeded draw is the whole of an epoch's composition, so a change that
    reordered it — or handed one shard's records to another's steps — would be
    a different corpus reported under the same seed.
    """
    run.start()
    assert [name for name, _ in run.trained] == run.drawn
    assert [records for _, records in run.trained] == [
        [f"{name} records"] for name in run.drawn
    ]


def test_a_failing_load_surfaces_at_its_own_shard(run):
    """Raised when the third shard's turn comes, not when its read did.

    A prefetch that let the exception out early would attribute it to whatever
    shard was training at the time, and one that swallowed it would train the
    epoch on three shards while reporting four.
    """
    with pytest.raises(RuntimeError, match=run.drawn[2]):
        run.start(failing=run.drawn[2])
    assert [name for name, _ in run.trained] == run.drawn[:2]


def test_the_loader_is_one_thread_and_is_shut_down(run):
    """One epoch, one pool, shut down however the epoch ended."""
    run.start()
    assert [pool.kwargs for pool in run.pools] == [{"max_workers": 1}]
    assert run.pools[0].shutdowns == 1


def test_the_loader_is_shut_down_when_a_load_raises(run):
    with pytest.raises(RuntimeError):
        run.start(failing=run.drawn[2])
    assert run.pools[0].shutdowns == 1
