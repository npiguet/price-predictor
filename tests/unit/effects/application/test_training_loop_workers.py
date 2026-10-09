"""Worker processes prepare the host half; the run is the same run.

The determinism tests train two real epochs end to end on the gen-1 fixture
records — split into three shards, with every draw the host half takes switched
on and a validation game the inherited split keeps out of training — once with
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

from effects.application import training_loop as loop_module
from effects.application.extract_keyword_definitions import load_keyword_definitions
from effects.application.step_preparation import (
    PreparerSettings,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.surface_batching import SurfaceBatcher
from effects.application.train_effect_model import (
    CorpusSplit,
    HeldOutCards,
    TrainEffectModelConfig,
    fields_for_epoch,
    sampling_class,
    variant_masks,
)
from effects.application.training_loop import TrainingLoop
from effects.domain.ability_encoder import AbilityEncoder
from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
from effects.domain.effect_model import EffectModel
from effects.infrastructure.record_io import read_shard, write_shard
from effects.infrastructure.sidecar_io import SidecarCache
from effects.infrastructure.step_workers import pack, unpack
from price_predictor.infrastructure.tokenizer_store import save_vocabulary

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"
_CARDS = (
    _FIXTURES / "gen1-sidecars" / "cardsfolder",
    _FIXTURES / "gen1-sidecars" / "tokenscripts",
)
_DEFINITIONS = _FIXTURES / "keyword-definitions.json"
SEED = 11
SHARDS = 3


@pytest.fixture(scope="module")
def records():
    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, records):
    """Three training shards, the validation sample, and a vocabulary file."""
    root = tmp_path_factory.mktemp("corpus")
    games = sorted({record.game_id for record in records})
    held = games[0]
    shards = []
    for index in range(SHARDS):
        path = root / f"train{index}.jsonl.gz"
        # Every shard keeps a few of the held game's records, which the
        # inherited split must keep out of training wherever it lands.
        write_shard(path, [
            r for i, r in enumerate(records)
            if i % SHARDS == index or r.game_id == held
        ])
        shards.append(path)
    sample = root / "validation.jsonl.gz"
    write_shard(sample, records[:24])

    sidecars = SidecarCache({"cardsfolder": _CARDS[0], "tokenscripts": _CARDS[1]})
    definitions = load_keyword_definitions(_DEFINITIONS)
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    batcher = SurfaceBatcher(
        tokenizer=bare, sidecars=sidecars, masks=variant_masks("full"),
        surface="script", e_dim=16, widths={}, device=torch.device("cpu"),
    )
    texts = [*batcher.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    vocab = dict(base)
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    vocab_path = root / "vocab-script.txt"
    save_vocabulary(vocab, vocab_path)
    return type("Corpus", (), {
        "root": root, "shards": shards, "sample": sample, "held": held,
        "vocab": vocab_path,
    })


def _config(corpus, out: Path, **overrides) -> TrainEffectModelConfig:
    settings = dict(
        corpus=str(corpus.root), cards_folders=_CARDS,
        keyword_definitions=_DEFINITIONS, vocab_path=corpus.vocab,
        model_output=out, seed=SEED, batch_size=8, steps_per_epoch=7,
        shards_per_epoch=SHARDS, epochs=2, curriculum_epoch=2, patience=5,
        e_dim=16, encoder_layers=1, encoder_d_model=32,
        # Every draw the host half takes, at rates that make each one fire.
        keyword_expand_p=0.5, context_dropout=0.2, mlm_weight=0.1,
        mlm_mask_prob=0.3, value_weight=0.05, api_weight=0.05, e_noise=0.1,
    )
    settings.update(overrides)
    return TrainEffectModelConfig(**settings)


def _run(
    monkeypatch, corpus, out: Path, caplog, *, workers: int, seed: int = SEED,
) -> dict:
    """Two epochs end to end; every loss, the final weights, the share lines."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    built: dict[str, torch.nn.Module] = {}
    losses: list[float] = []
    real_loss = TrainingLoop._loss_for

    def encoder(config):
        built["encoder"] = AbilityEncoder(config)
        return built["encoder"]

    def model(config):
        # The trunk at a size a unit suite can train on a CPU; the workers
        # never build it, so the patch reaches every run alike.
        built["model"] = EffectModel(replace(
            config, d_model=32, n_layers=1, n_heads=2, ff_dim=64,
        ))
        return built["model"]

    def loss_for(self, *args, **kwargs):
        computed = real_loss(self, *args, **kwargs)
        if computed is not None:
            losses.append(float(computed[0].detach()))
        return computed

    monkeypatch.setattr(loop_module, "AbilityEncoder", encoder)
    monkeypatch.setattr(loop_module, "EffectModel", model)
    monkeypatch.setattr(TrainingLoop, "_loss_for", loss_for)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="effects.application.training_loop"):
        code = TrainingLoop(
            _config(corpus, out, prefetch_workers=workers, seed=seed),
            held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
            inherited=CorpusSplit(
                held_out_cards=(), card_disjoint_games=frozenset(),
                game_disjoint_games=frozenset({corpus.held}),
            ),
            training_shards=corpus.shards,
            validation_samples={
                "card-disjoint": corpus.sample, "game-disjoint": corpus.sample,
            },
            holdout_permille=20, holdout_max_carriers=8, gate_one_records=0,
            rarity={}, corpus_digest="",
        ).execute()
    monkeypatch.undo()
    assert code == 0
    return {
        "losses": losses,
        "parameters": {
            f"{owner}.{name}": value.detach().clone()
            for owner, module in built.items()
            for name, value in module.state_dict().items()
        },
        "shares": [m for m in caplog.messages if "by rarity bucket" in m],
        "held back": [m.split("|")[2] for m in caplog.messages if "held back" in m],
        "workers": any("worker process" in m for m in caplog.messages),
    }


def _assert_same(first: dict, second: dict) -> None:
    assert first["losses"] == second["losses"]
    assert first["shares"] == second["shares"]
    assert first["held back"] == second["held back"]
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
    """Steps trained, both epochs logged, and the held game held back."""
    assert len(in_process["losses"]) >= 14
    assert len(in_process["shares"]) == 2
    assert all(" 0 held back" not in line for line in in_process["held back"])


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
    config = _config(corpus, corpus.root / "unused")
    loop = TrainingLoop(
        config, held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=20, holdout_max_carriers=8, rarity={"x": 3},
    )
    tokenizer = loop._build_tokenizer()
    sidecars = loop._build_sidecars()
    widths = loop._feature_widths(records)
    loop.api_types, loop.param_keys = loop._script_vocabularies()
    heads = frozenset({"value", "api", "mlm"})
    fields = fields_for_epoch(
        config, present=frozenset(sampling_class(r) for r in records), epoch=1,
    )
    task = ShardTask(
        shard=Path("fixture"), epoch=1, position=2, budget=5, fields=fields,
    )
    here = list(loop.shard_steps(
        task, records, tokenizer, sidecars, widths, heads=heads,
    ))
    worker = StepPreparer(PreparerSettings(
        config=config, seed=loop.seed, held_out=loop.held_out, inherited=None,
        rarity={"x": 3}, api_types=tuple(loop.api_types),
        param_keys=tuple(loop.param_keys), widths=widths, heads=heads,
    ))
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


def test_a_worker_exception_surfaces_with_its_traceback_and_no_process_survives(
    monkeypatch, corpus, tmp_path, caplog,
):
    """An unreadable shard: raised at its own shard, in the training process."""
    missing = tmp_path / "missing.jsonl.gz"
    corpus_with_hole = type("Corpus", (), {
        **{k: v for k, v in vars(corpus).items() if not k.startswith("__")},
        "shards": [missing],
    })
    with pytest.raises(RuntimeError) as raised:
        _run(monkeypatch, corpus_with_hole, tmp_path / "out", caplog, workers=2)
    assert "Traceback" in str(raised.value)
    assert "missing.jsonl.gz" in str(raised.value)
    assert any("preparing it failed" in m for m in caplog.messages)
    assert not multiprocessing.active_children()


def test_an_interrupt_mid_shard_leaves_no_process_behind(
    monkeypatch, corpus, tmp_path, caplog,
):
    """Ctrl+C lands in the training loop with workers blocked on full queues."""
    real_loss = TrainingLoop._loss_for
    calls = []

    def interrupted(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real_loss(self, *args, **kwargs)

    monkeypatch.setattr(TrainingLoop, "_loss_for", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, corpus, tmp_path / "out", caplog, workers=2)
    assert not multiprocessing.active_children()
