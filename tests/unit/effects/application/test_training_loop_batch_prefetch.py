"""The batch prefetch: step k+1's host half is prepared while step k trains.

The equivalence test trains real steps on the gen-1 fixture records, with every
draw the host half takes switched on — keyword expansion, context dropout, the
MLM mask, and a shard short enough that its plans cross into a second shuffle —
and asserts the run with the prefetch is the run without it, bit for bit. Each
step draws from a generator seeded by its place in the run, so a prefetch that
handed one step's draws to another would change what a later step trains on,
and the parameters would show it.

The rest stub the halves and read a clock, like the shard prefetch's suite: an
implementation that submitted the next step and then waited for it before
stepping would pass an ordering check and overlap nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest
import torch

from effects.application.extract_keyword_definitions import load_keyword_definitions
from effects.application.step_preparation import ShardTask, StepPreparer
from effects.application.surface_batching import Materializer
from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    fields_for_epoch,
)
from effects.application.training_loop import (
    EpochTally,
    RunContext,
    StepLoss,
    TrainingLoop,
    parameter_groups,
)
from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
from effects.domain.baselines import baseline_masks
from effects.domain.effect_head_input import SlotKind, feature_widths
from effects.domain.effect_losses import LossParts
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.sampling import sampling_class
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache, script_vocabularies
from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from tests.unit.effects.builders import make_batcher, make_preparer, make_training_loop

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"
_SIDECAR_ROOTS = {
    "cardsfolder": _FIXTURES / "gen1-sidecars" / "cardsfolder",
    "tokenscripts": _FIXTURES / "gen1-sidecars" / "tokenscripts",
}
#: Steps per run: more than the 53 fixture records fill at this batch size,
#: so the planner starts a second shuffle partway through the shard.
STEPS = 9
BATCH = 8
SEED = 11
#: What a stubbed host half and a stubbed device half each cost.
PAUSE = 0.03


@pytest.fixture(scope="module")
def records():
    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def definitions():
    return load_keyword_definitions(_FIXTURES / "keyword-definitions.json")

def _run(encoder, model, optimizer) -> RunContext:
    """The run a shard trains in, with no configs: nothing here saves."""
    return RunContext(
        encoder_config=None, model_config=None, encoder=encoder, model=model,
        optimizer=optimizer, warmup=1,
    )


def _config(directory: Path, **overrides) -> TrainEffectModelConfig:
    settings = dict(
        corpus="unused", seed=SEED, batch_size=BATCH, steps_per_epoch=STEPS,
        epochs=1, curriculum_epoch=1, e_dim=16, encoder_layers=1,
        encoder_d_model=32,
        # Every draw the host half takes, at rates that make each one fire.
        keyword_expand_p=0.5, context_dropout=0.2, mlm_weight=0.1,
        mlm_mask_prob=0.3, value_weight=0.05, api_weight=0.05, e_noise=0.1,
        # The script surface, which the fixture sidecars carry.
        vocab_path=directory / "vocab-script.txt",
        keyword_definitions=_FIXTURES / "keyword-definitions.json",
        cards_folders=tuple(_SIDECAR_ROOTS.values()),
    )
    settings.update(overrides)
    return TrainEffectModelConfig(**settings)


def _loop(monkeypatch, directory: Path, *, prefetch: bool, **overrides) -> TrainingLoop:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return make_training_loop(
        _config(directory, **overrides),
        prefetch_batches=prefetch,
    )


def _vocabulary(directory: Path, records, definitions) -> None:
    """A vocabulary covering every text the fixture encodes and expands to."""
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    batcher = make_batcher(
        tokenizer=bare, sidecars=SidecarCache(dict(_SIDECAR_ROOTS)),
        masks=baseline_masks("full"), surface="script", e_dim=16,
    )
    texts = [*batcher.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    vocab = dict(base)
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    save_vocabulary(vocab, directory / "vocab-script.txt")


def _prepared_loop(loop: TrainingLoop, records) -> None:
    """The loop's host half and device half, as ``execute`` would build them."""
    loop.api_types, loop.param_keys = script_vocabularies(list(_SIDECAR_ROOTS.values()))
    assert loop.api_types, "the fixture should exercise the script-API head"
    loop.preparer = StepPreparer(
        loop._preparer_settings(feature_widths(records)), truncations=loop.truncations,
    )
    loop._training = Materializer(loop.device, loop.masks, noise=loop.noise)


def _shard(loop, records, budget, fields):
    """One shard's prepared steps, in this process, and the task they are for."""
    task = ShardTask(
        shard=Path("gen1-records.jsonl.gz"), epoch=1, position=1,
        budget=budget, fields=fields,
    )
    return task, loop._local_steps(task, records)


def _train(monkeypatch, directory, records, definitions, *, prefetch: bool):
    """``STEPS`` real steps on the fixture shard; what they computed, and where."""
    _vocabulary(directory, records, definitions)
    loop = _loop(monkeypatch, directory, prefetch=prefetch)
    _prepared_loop(loop, records)
    tokenizer, widths = loop.preparer.tokenizer, loop.preparer.widths

    torch.manual_seed(SEED)
    encoder = AbilityEncoder(AbilityEncoderConfig(
        vocab_size=tokenizer.vocab_size, e_dim=loop.config.e_dim,
        d_model=loop.config.encoder_d_model, n_layers=loop.config.encoder_layers,
    ))
    model = EffectModel(EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
        e_dim=loop.config.e_dim, vocab_size=tokenizer.vocab_size,
        n_api_types=len(loop.api_types), n_param_keys=len(loop.param_keys),
        encoder_d_model=loop.config.encoder_d_model,
        d_model=32, n_layers=1, n_heads=2, ff_dim=64,
    ))
    encoder.train()
    model.train()
    optimizer = torch.optim.AdamW(parameter_groups(encoder, model), lr=1e-3)

    losses: list[tuple[float, float]] = []
    terms: list[dict[str, float]] = []
    threads: list[str] = []
    real_prepare = StepPreparer.prepare_step
    real_loss = TrainingLoop._device_loss
    real_mlm = TrainingLoop._mlm_backward

    def prepare(self, *args, **kwargs):
        threads.append(threading.current_thread().name)
        return real_prepare(self, *args, **kwargs)

    def device_loss(self, *args, **kwargs):
        computed = real_loss(self, *args, **kwargs)
        losses.append((float(computed.total.detach()), float(computed.shipped.detach())))
        return computed

    def mlm_backward(self, loss, *args, **kwargs):
        # Captured after the MLM pass, which runs after the main backward, so
        # the step's terms include it.
        real_mlm(self, loss, *args, **kwargs)
        terms.append({
            name: float(value.detach()) for name, value in loss.terms.items()
        })

    monkeypatch.setattr(StepPreparer, "prepare_step", prepare)
    monkeypatch.setattr(TrainingLoop, "_device_loss", device_loss)
    monkeypatch.setattr(TrainingLoop, "_mlm_backward", mlm_backward)
    fields = fields_for_epoch(
        loop.config, present=frozenset(sampling_class(r) for r in records), epoch=1,
    )
    task, steps = _shard(loop, records, STEPS, fields)
    run = _run(encoder, model, optimizer)
    tally = EpochTally(running=torch.zeros(()))
    loop._train_on_shard(task, steps, 0.0, run, tally, of=1)
    monkeypatch.undo()
    parameters = {
        f"{owner}.{name}": value.detach().clone()
        for owner, module in (("encoder", encoder), ("model", model))
        for name, value in module.state_dict().items()
    }
    return {
        "losses": losses, "terms": terms, "threads": threads,
        "parameters": parameters, "running": float(tally.running),
        "counters": (run.step, tally.taken),
    }


def test_the_prefetch_trains_exactly_what_the_serial_loop_trains(
    monkeypatch, records, definitions, tmp_path,
):
    serial = _train(monkeypatch, tmp_path, records, definitions, prefetch=False)
    ahead = _train(monkeypatch, tmp_path, records, definitions, prefetch=True)

    # The host half really did move: off the main thread with the prefetch,
    # on it without.
    assert set(serial["threads"]) == {threading.main_thread().name}
    assert all(name.startswith("batch-prefetch") for name in ahead["threads"])
    assert len(ahead["threads"]) == STEPS

    assert serial["counters"] == ahead["counters"] == (STEPS, STEPS)
    # Every training-only head ran, so its draws were on the line too.
    assert {"mlm", "api", "value"} <= set(serial["terms"][0])
    assert serial["losses"] == ahead["losses"]
    assert serial["terms"] == ahead["terms"]
    assert serial["running"] == ahead["running"]
    assert serial["parameters"].keys() == ahead["parameters"].keys()
    for name, value in serial["parameters"].items():
        assert torch.equal(value, ahead["parameters"][name]), name


# ── the overlap, with both halves stubbed ───────────────────────────────


class _Stub(torch.nn.Module):
    """A module with one parameter, so backward and the optimizer have work."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2))


def _stubbed(monkeypatch, directory, records, *, failing_at: int | None = None):
    """Run a shard with sleeping halves; what each half did, and when."""
    marks: list[tuple[str, int, str, float]] = []
    prepared: list[int] = []
    model = _Stub()

    def mark(what: str, which: int, phase: str) -> None:
        marks.append((what, which, phase, time.perf_counter()))

    def prepare(self, plan, *args, **kwargs):
        which = len(prepared)
        prepared.append(which)
        mark("prepare", which, "start")
        time.sleep(PAUSE)
        if which == failing_at:
            raise RuntimeError(f"step {which} could not be prepared")
        mark("prepare", which, "finish")
        return which

    def device_loss(self, prepared, *args, **kwargs):
        mark("train", prepared, "start")
        time.sleep(PAUSE)
        mark("train", prepared, "finish")
        loss = model.p.sum()
        return StepLoss(total=loss, shipped=loss, parts=LossParts(), terms={})

    monkeypatch.setattr(StepPreparer, "prepare_step", prepare)
    monkeypatch.setattr(TrainingLoop, "_device_loss", device_loss)
    loop = _loop(monkeypatch, directory, prefetch=True)
    loop.preparer = make_preparer(directory, cards_folders=tuple(_SIDECAR_ROOTS.values()))
    task, steps = _shard(loop, records, 4, ())
    run = _run(_Stub(), model, torch.optim.SGD(model.parameters(), lr=0.1))
    loop._train_on_shard(
        task, steps, 0.0, run, EpochTally(running=torch.zeros(())), of=1,
    )
    return marks, prepared


def _at(marks, what: str, which: int, phase: str) -> float:
    return next(when for kind, index, stage, when in marks
                if kind == what and index == which and stage == phase)


def test_each_step_is_prepared_while_its_predecessor_trains(monkeypatch, records, tmp_path):
    marks, prepared = _stubbed(monkeypatch, tmp_path, records)
    assert prepared == [0, 1, 2, 3]
    for current in range(3):
        assert _at(marks, "prepare", current + 1, "start") < _at(
            marks, "train", current, "finish",
        )
    # In step order, each step training on its own preparation.
    assert [which for kind, which, phase, _ in marks
            if kind == "train" and phase == "start"] == [0, 1, 2, 3]


def test_a_failing_preparation_surfaces_at_its_own_step(monkeypatch, records, tmp_path):
    """Raised on the main thread when step 2's turn comes, with the pool gone."""
    with pytest.raises(RuntimeError, match="step 2"):
        _stubbed(monkeypatch, tmp_path, records, failing_at=2)
    assert not any(
        thread.name.startswith("batch-prefetch") for thread in threading.enumerate()
    )


def test_the_shard_line_reports_the_batch_wait(monkeypatch, records, caplog, tmp_path):
    with caplog.at_level(logging.INFO, logger="effects.application.training_loop"):
        _stubbed(monkeypatch, tmp_path, records)
    assert any("batch wait" in message for message in caplog.messages)
