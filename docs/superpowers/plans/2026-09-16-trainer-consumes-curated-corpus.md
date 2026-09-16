# Trainer Consumes the Curated Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train-effect-model` read everything corpus-shaped from the curated corpus and fix the four training-time mechanics the first run exposed: joint clipping, mid-epoch curriculum, the Poisson floor, and per-shard mixture resampling.

**Architecture:** `--corpus` becomes the only input. The validation samples come from `validation/samples/`, the gate-1 count from the manifest, and the startup sweep goes. A shard trains as a weighted shuffle without replacement over its records. The optimizer holds one parameter group per module so `clip_per_group` clips them apart. The sparse-field curriculum starts at an epoch boundary, validation scores the field set the epoch trained with, and the early stopper resets there. Each epoch also prints gate F1, zone accuracy and deviance on the card-disjoint sample.

**Tech Stack:** Python 3.12, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-corpus-and-trainer-rework.md` (FR-081, FR-082, FR-086, FR-088b, FR-089, FR-095, FR-125 amendments). Depends on the build-corpus plan having landed: the trainer reads `validation/samples/` and `per_stratum["gate-one"]`.

## Global Constraints

- Hardcoded constants stay hardcoded (FR-095): `LEARNING_RATE = 1e-4`, `MAX_GRAD_NORM = 1.0`, `WEIGHT_DECAY = 0.01`, warmup 5%.
- Best-checkpoint selection stays on card-disjoint validation loss (FR-089).
- Every unit test runs without a GPU and without a corpus on disk.
- Run tests from the repository root with `python -m pytest`.
- Before editing any file under `specs/` or `experiments/` (Tasks 7 and 8), load the `feature-workflow` skill: those files carry binding wording and structure rules.

---

## File map

| File | Responsibility |
|---|---|
| Modify `src/effects/domain/effect_model.py` | `_poisson(..., full=True)` |
| Modify `src/effects/application/train_effect_model.py` | `batches_without_replacement`, `curriculum_epoch` config, `run()` requires `--corpus`, `EarlyStopper.reset` |
| Modify `src/effects/application/training_loop.py` | load samples, two optimizer groups, per-group norms, epoch-boundary curriculum, gate metrics per epoch, no mixture |
| Modify `src/effects/infrastructure/cli.py` | drop raw-corpus flags, `--curriculum-epoch` |
| Delete `src/effects/infrastructure/shard_sweep.py` and its tests | the sweep |
| Modify `specs/023-ability-effect-model/spec.md`, `quickstart.md` | FR amendments, § train |
| Tests | `tests/unit/effects/domain/test_effect_model.py`, `tests/unit/effects/application/test_sampling.py`, `tests/unit/effects/application/test_training_loop_groups.py` (new), `tests/unit/effects/application/test_curriculum.py` (new) |

---

### Task 1: Full Poisson NLL

**Files:**
- Modify: `src/effects/domain/effect_model.py:410-415`
- Test: `tests/unit/effects/domain/test_effect_model.py`

- [ ] **Step 1: Write the failing test**

Append to the class holding `test_a_count_field_uses_a_poisson_loss`:

```python
    def test_the_count_loss_is_nonnegative_at_its_optimum(self):
        """`full=True` adds the Stirling term, so a perfect prediction scores ~0
        instead of k - k ln k, which is -306 for a target of 88 (FR-081)."""
        from effects.domain.effect_model import _poisson
        target = torch.tensor([88.0, 3.0, 1.0])
        loss = _poisson(torch.log(target), target)
        assert loss.item() >= 0.0
        assert loss.item() < 3.0 * 0.6   # Stirling's residual is under 0.6 per element
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/effects/domain/test_effect_model.py -k nonnegative -v`
Expected: FAIL, `loss.item()` is about -308

- [ ] **Step 3: Change one argument**

```python
def _poisson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson NLL on a log-rate prediction, summed.

    ``full=True`` adds the Stirling approximation of ``log(k!)``, which is
    constant in the prediction and so changes no gradient — but it moves the
    minimum of every term to about zero. Without it a term's floor is
    ``k - k·ln k``, and a batch holding one 88-card draw reads as -306 at a
    perfect prediction, which is what made the first run's training loss
    track batch composition rather than the model.
    """
    return functional.poisson_nll_loss(
        prediction, target, log_input=True, full=True, reduction="sum",
    )
```

- [ ] **Step 4: Run the model tests**

Run: `python -m pytest tests/unit/effects/domain/test_effect_model.py tests/unit/effects/application/test_gate_one.py -v`
Expected: all pass. `test_a_count_field_uses_a_poisson_loss` may assert a numeric value; if it does, recompute it with `full=True` and update the expected number in the test.

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/effect_model.py tests/unit/effects/domain/test_effect_model.py
git commit -m "fix(effects): full Poisson NLL so count terms have a zero floor"
```

---

### Task 2: Batches without replacement

**Files:**
- Modify: `src/effects/application/train_effect_model.py` (after `plan_batch`)
- Test: `tests/unit/effects/application/test_sampling.py` (append a class)

**Interfaces:**
- Produces: `weighted_order(weights: Sequence[float], rng: random.Random) -> list[int]`: indices in weighted-shuffle order (Efraimidis–Spirakis keys `u ** (1 / w)`, largest first).
- Produces: `batches_without_replacement(records: Sequence[EffectRecord], weights: Sequence[float], *, batch_size: int, rng: random.Random) -> Iterator[BatchPlan]`: an endless iterator; each pass is one weighted shuffle cut into full batches, the ragged tail dropped, then a new shuffle. Yields nothing for empty `records`.

- [ ] **Step 1: Write the failing tests**

```python
class TestBatchesWithoutReplacement:
    """A shard is one weighted shuffle per pass, not a draw with replacement (FR-086)."""

    def _records(self, make_record, n):
        return [make_record(record_id=f"r{i}", game_id=f"g{i % 4}") for i in range(n)]

    def test_one_pass_visits_every_record_exactly_once(self, make_record):
        records = self._records(make_record, 10)
        batches = batches_without_replacement(records, [1.0] * 10, batch_size=5, rng=random.Random(1))
        seen = [r.record_id for _ in range(2) for r in next(batches).records]
        assert sorted(seen) == sorted(r.record_id for r in records)

    def test_the_ragged_tail_is_dropped_and_a_new_pass_begins(self, make_record):
        records = self._records(make_record, 7)
        batches = batches_without_replacement(records, [1.0] * 7, batch_size=3, rng=random.Random(1))
        first_pass = [next(batches).records for _ in range(2)]
        assert all(len(b) == 3 for b in first_pass)
        third = next(batches).records
        assert len(third) == 3                        # a fresh shuffle, not the 1-record tail

    def test_a_heavier_record_comes_earlier_on_average(self, make_record):
        records = self._records(make_record, 2)
        firsts = Counter()
        for seed in range(200):
            batches = batches_without_replacement(records, [1.0, 10.0], batch_size=1, rng=random.Random(seed))
            firsts[next(batches).records[0].record_id] += 1
        assert firsts["r1"] > 150

    def test_records_are_grouped_by_game(self, make_record):
        records = self._records(make_record, 8)
        plan = next(batches_without_replacement(records, [1.0] * 8, batch_size=8, rng=random.Random(3)))
        assert set(plan.by_game) == {"g0", "g1", "g2", "g3"}
        assert all(r.game_id == game for game, group in plan.by_game.items() for r in group)

    def test_no_records_yields_nothing(self):
        assert list(batches_without_replacement([], [], batch_size=4, rng=random.Random(0))) == []

    def test_a_batch_larger_than_the_shard_is_the_whole_shard(self, make_record):
        records = self._records(make_record, 3)
        plan = next(batches_without_replacement(records, [1.0] * 3, batch_size=32, rng=random.Random(0)))
        assert len(plan.records) == 3
```

Add `import random` and `from collections import Counter` at the top of the test module if missing, and `batches_without_replacement` to its imports from `train_effect_model`. The module already uses a `make_record`-style fixture; if it builds records differently, use its existing helper.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_sampling.py -k WithoutReplacement -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

After `plan_batch` in `train_effect_model.py`:

```python
def weighted_order(weights: Sequence[float], rng: random.Random) -> list[int]:
    """Indices in a weighted shuffle: each drawn once, heavier ones earlier.

    Efraimidis–Spirakis: key ``u ** (1 / w)`` per item, sorted descending, is a
    sample without replacement in proportion to ``w``. One shuffle per pass is
    what lets a shard be seen whole rather than resampled from with
    replacement, which is how a sixteen-record shard was replayed 640 times.
    """
    keys = [
        rng.random() ** (1.0 / max(float(weight), 1e-9)) for weight in weights
    ]
    return sorted(range(len(keys)), key=keys.__getitem__, reverse=True)


def batches_without_replacement(
    records: Sequence[EffectRecord], weights: Sequence[float], *,
    batch_size: int, rng: random.Random,
) -> Iterator[BatchPlan]:
    """Endless full batches over ``records``: one weighted shuffle per pass.

    A batch smaller than ``batch_size`` is yielded only when the whole shard
    is smaller; otherwise the ragged tail is dropped and the next pass
    starts, so every step trains on a full batch.
    """
    if not records:
        return
    size = min(batch_size, len(records))
    while True:
        order = weighted_order(weights, rng)
        for start in range(0, len(order) - size + 1, size):
            plan: dict[str, list[EffectRecord]] = defaultdict(list)
            for index in order[start:start + size]:
                plan[records[index].game_id].append(records[index])
            yield BatchPlan(dict(plan))
```

Add `Iterator` to the `collections.abc` import if absent.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/effects/application/test_sampling.py -v`
Expected: all pass (the old `TestBatchPlanning` still passes; `plan_batch` is deleted in Task 7).

- [ ] **Step 5: Commit**

```bash
git add src/effects/application/train_effect_model.py tests/unit/effects/application/test_sampling.py
git commit -m "feat(effects): weighted batches without replacement over a shard"
```

---

### Task 3: Epoch-boundary curriculum and a resettable stopper

**Files:**
- Modify: `src/effects/application/train_effect_model.py` (`TrainEffectModelConfig`, `EarlyStopper`)
- Modify: `src/effects/infrastructure/cli.py` (`--curriculum-epoch` replaces `--curriculum-step`)
- Test: `tests/unit/effects/application/test_curriculum.py` (new)

**Interfaces:**
- `TrainEffectModelConfig.curriculum_epoch: int = 3` replaces `curriculum_step`. New property `curriculum_step` returns `(curriculum_epoch - 1) * steps_per_epoch`, so the sparse group enables at the first step of epoch `curriculum_epoch`; `curriculum_epoch <= 1` means from step zero.
- `EarlyStopper.reset() -> None`: forgets the best and the count.
- New helper `fields_for_epoch(config, *, present, epoch) -> tuple[FieldSpec, ...]`: `active_fields(present_classes=present, step=(epoch - 1) * config.steps_per_epoch, curriculum_step=config.curriculum_step)`, the field set every step of `epoch` trains with and validation scores.

- [ ] **Step 1: Write the failing tests**

```python
"""The sparse-field curriculum starts at an epoch boundary (FR-082)."""

from __future__ import annotations

from effects.application.train_effect_model import (
    EarlyStopper,
    TrainEffectModelConfig,
    fields_for_epoch,
)
from effects.domain.effect_model import (
    CLASS_RESOLUTION_EFFECT,
    FieldGroup,
    PER_ENTITY_FIELDS,
)

PRESENT = frozenset({CLASS_RESOLUTION_EFFECT})


def _sparse(fields):
    return [f for f in fields if f.group is not FieldGroup.DENSE]


def test_the_curriculum_step_is_the_first_step_of_the_curriculum_epoch():
    config = TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=3)
    assert config.curriculum_step == 10_000
    assert TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=1).curriculum_step == 0


def test_sparse_fields_are_off_before_the_curriculum_epoch_and_on_from_it():
    config = TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=3)
    assert _sparse(fields_for_epoch(config, present=PRESENT, epoch=2)) == []
    on = _sparse(fields_for_epoch(config, present=PRESENT, epoch=3))
    assert on and all(f in PER_ENTITY_FIELDS for f in on)


def test_the_stopper_resets():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(1.0)
    assert not stopper.update(2.0)
    stopper.reset()
    assert stopper.best == float("inf") and stopper.since_best == 0
    assert stopper.update(5.0)          # a worse number is a new best after the reset
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_curriculum.py -v`
Expected: FAIL with `ImportError: cannot import name 'fields_for_epoch'`

- [ ] **Step 3: Implement**

In `TrainEffectModelConfig`, replace `curriculum_step: int = 10_000` with:

```python
    #: The epoch whose first step enables the sparse field group (FR-082).
    #: An epoch rather than a step so validation before and after the switch
    #: scores different field sets at an epoch boundary, never inside one:
    #: the first run enabled the group at shard 244 of epoch 2 and validated
    #: twenty-five untrained heads at that epoch's end, a 134-loss spike.
    curriculum_epoch: int = 3
```

and add to the class:

```python
    @property
    def curriculum_step(self) -> int:
        """The optimizer step the sparse group enables at; 0 for epoch 1."""
        return max(0, self.curriculum_epoch - 1) * self.steps_per_epoch
```

Add the helper next to `warmup_steps`:

```python
def fields_for_epoch(config: TrainEffectModelConfig, *, present: frozenset[str], epoch: int):
    """The field set every step of ``epoch`` trains with, and validation scores."""
    from effects.domain.effect_model import active_fields

    return active_fields(
        present_classes=present,
        step=(epoch - 1) * config.steps_per_epoch,
        curriculum_step=config.curriculum_step,
    )
```

In `EarlyStopper`:

```python
    def reset(self) -> None:
        """Forget the best: the objective changed, so no earlier number compares."""
        self.best = float("inf")
        self.since_best = 0
```

In `cli.py`, replace `parser.add_argument("--curriculum-step", type=int, default=10000)` with:

```python
    parser.add_argument(
        "--curriculum-epoch", type=int, default=3,
        help="Epoch whose first step enables the sparse field group (default: 3)",
    )
```

and update the `TrainEffectModelConfig(...)` construction in the train command to pass `curriculum_epoch=args.curriculum_epoch`. Grep `curriculum_step` across `src/` and `tests/`: every remaining use must read the property, and any test constructing the config with `curriculum_step=` must switch to `curriculum_epoch=`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/effects/application/test_curriculum.py tests/unit/effects -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/effects/application/train_effect_model.py src/effects/infrastructure/cli.py tests/
git commit -m "feat(effects): curriculum at an epoch boundary, resettable early stopper"
```

---

### Task 4: One optimizer group per module

**Files:**
- Modify: `src/effects/application/training_loop.py` (`execute`, `_train_on_shard`, `_format_norms`; delete `module_grad_norms`)
- Test: `tests/unit/effects/application/test_training_loop_groups.py` (new)

**Interfaces:**
- Produces: `parameter_groups(encoder, model, identity_table=None) -> list[dict]`: `[{"name": "encoder", "params": [...]}, {"name": "head", "params": [...]}]` plus `{"name": "identity", ...}` when given. Module-level in `training_loop.py`.
- `_format_norms` prints `(clipped per group at 1)`.

- [ ] **Step 1: Write the failing test**

```python
"""The encoder and the head are clipped apart (FR-095)."""

from __future__ import annotations

import torch

from effects.application.training_loop import parameter_groups
from price_predictor.infrastructure.torch_training import clip_per_group


def test_groups_are_named_and_cover_every_parameter():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    groups = parameter_groups(encoder, head)
    assert [g["name"] for g in groups] == ["encoder", "head"]
    assert sum(len(g["params"]) for g in groups) == len(list(encoder.parameters())) + len(list(head.parameters()))
    identity = torch.nn.Embedding(4, 2)
    assert [g["name"] for g in parameter_groups(encoder, head, identity)][-1] == "identity"


def test_a_large_head_gradient_does_not_scale_the_encoder():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    for p in encoder.parameters():
        p.grad = torch.full_like(p, 0.01)           # tiny encoder gradient
    for p in head.parameters():
        p.grad = torch.full_like(p, 100.0)          # huge head gradient
    optimizer = torch.optim.AdamW(parameter_groups(encoder, head), lr=1e-4)
    encoder_before = torch.cat([p.grad.flatten() for p in encoder.parameters()]).norm().item()
    norms = clip_per_group(optimizer, max_norm=1.0)
    encoder_after = torch.cat([p.grad.flatten() for p in encoder.parameters()]).norm().item()
    assert set(norms) == {"encoder", "head"}
    assert abs(encoder_after - encoder_before) < 1e-6      # under the norm: untouched
    assert torch.cat([p.grad.flatten() for p in head.parameters()]).norm().item() <= 1.0 + 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_training_loop_groups.py -v`
Expected: FAIL with `ImportError: cannot import name 'parameter_groups'`

- [ ] **Step 3: Implement**

In `training_loop.py`, replace `module_grad_norms` (and its docstring) with:

```python
def parameter_groups(encoder, model, identity_table=None) -> list[dict]:
    """One optimizer group per module, named for ``clip_per_group``.

    Clipped apart rather than together (FR-095): the head's gradient norm ran
    five to twenty times the encoder's in the first run, and a joint clip at
    1.0 scaled the encoder's update by the head's norm — the one path the
    effect loss reaches the encoder through, cut to a twentieth.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters())},
        {"name": "head", "params": list(model.parameters())},
    ]
    if identity_table is not None:
        groups.append({"name": "identity", "params": list(identity_table.parameters())})
    return groups
```

In `execute`, replace the `trainable = [...]` / `optimizer = torch.optim.AdamW(trainable, ...)` block with:

```python
        optimizer = torch.optim.AdamW(
            parameter_groups(encoder, model, self.identity_table),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
```

In `_train_on_shard`, delete the `if due: norms = module_grad_norms(...)` block and change the clip line to keep the pre-clip norms on the reporting step:

```python
            if (step + 1) % self.config.grad_accum == 0:
                clipped = clip_per_group(optimizer, max_norm=MAX_GRAD_NORM)
                if due:
                    norms = clipped
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
```

In `_format_norms`, change the suffix to `f"\n  |g| {body} (clipped per group at {MAX_GRAD_NORM:g})"`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/effects/application/test_training_loop_groups.py tests/unit/effects -q`
Expected: all pass. If a test referenced `module_grad_norms`, delete that test: its subject no longer exists.

- [ ] **Step 5: Commit**

```bash
git add src/effects/application/training_loop.py tests/unit/effects/application/test_training_loop_groups.py
git commit -m "fix(effects): clip the encoder and the head as separate optimizer groups"
```

---

### Task 5: Read the corpus; drop the sweep and the mixture

**Files:**
- Modify: `src/effects/application/train_effect_model.py` (`run`, `TrainEffectModelConfig`)
- Modify: `src/effects/application/training_loop.py` (`__init__`, `_capture_validation` → `_load_validation`, `execute`, `_train_on_shard`, `_pools` → `_weighted`)
- Modify: `src/effects/infrastructure/cli.py` (train parser)
- Delete: `src/effects/infrastructure/shard_sweep.py`, `tests/unit/effects/application/test_validation_capture.py`
- Test: `tests/unit/effects/application/test_training_loop_validation.py` (new)

**Interfaces:**
- `TrainingLoop.__init__(config, *, held_out, inherited, training_shards, validation_samples: Mapping[str, Path], gate_one_records: int, rarity, corpus_digest)`. `validation_shards` is gone.
- `TrainingLoop._load_validation()`: fills `self.card_disjoint`, `self.game_disjoint`, `self.probe` (first `PROBE_RECORDS` of the card-disjoint sample), `self.present` (classes in both samples).
- `TrainingLoop._weighted(records, sidecars) -> list[float]`: rarity weights for a shard's records (the body of `_pools` minus the per-class split).
- `run()` returns 1 with a logged error when `config.corpus is None`.
- Removed config fields and flags: `records_dir`, `reserved_shards`, `workers`, `kind_mix`, `holdout_permille`, `holdout_max_carriers`, `min_holdout_records`, `split_from` handling that derived a split (the `--split-from` flag stays only to inherit vocabulary/keyword paths for variants: check `require_split_from` and keep what it needs).

- [ ] **Step 1: Write the failing tests**

```python
"""The trainer reads its validation set from the corpus (FR-089, FR-125 withdrawn)."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.train_effect_model import (
    HeldOutCards,
    TrainEffectModelConfig,
    run,
)
from effects.application.training_loop import TrainingLoop
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.record_io import write_shard


@pytest.fixture
def samples(tmp_path, make_record):
    card = [make_record(record_id=f"c{i}", game_id="cg") for i in range(5)]
    game = [make_record(record_id=f"g{i}", game_id="gg", kind=RecordKind.COMBAT,
                        payload=CombatPayload()) for i in range(3)]
    paths = {"card-disjoint": tmp_path / "card-disjoint.jsonl.gz",
             "game-disjoint": tmp_path / "game-disjoint.jsonl.gz"}
    write_shard(paths["card-disjoint"], card)
    write_shard(paths["game-disjoint"], game)
    return paths


def test_the_loop_loads_both_samples_and_the_probe(samples):
    loop = TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples=samples,
        gate_one_records=0, rarity={}, corpus_digest="",
    )
    loop._load_validation()
    assert [r.record_id for r in loop.card_disjoint] == [f"c{i}" for i in range(5)]
    assert len(loop.game_disjoint) == 3
    assert loop.probe and loop.probe[0].record_id == "c0"
    assert loop.present == {"resolution-effect", "combat"}


def test_run_refuses_to_train_without_a_corpus(caplog):
    assert run(TrainEffectModelConfig()) == 1
    assert "--corpus" in caplog.text
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_training_loop_validation.py -v`
Expected: FAIL with `TypeError` (unexpected keyword `validation_samples`) and an assertion on the second test.

- [ ] **Step 3: `run()` reads the corpus only**

In `run()`, replace the whole `if config.corpus is not None: ... else: ...` block with:

```python
    if config.corpus is None:
        logger.error(
            "train-effect-model reads a curated corpus: pass --corpus DIR, built "
            "with `python -m effects build-corpus`. Training straight from raw "
            "shards was withdrawn with FR-125."
        )
        return 1
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(Path(config.corpus))
    manifest = store.load()
    require_matching_surface(
        manifest_surface=manifest.surface, vocab_path=Path(config.vocab_path),
        corpus=config.corpus,
    )
    training_shards = corpus_shards(store.training_dir)
    if not training_shards:
        logger.error("No training shards under %s.", store.training_dir)
        return 1
    validation_samples = {
        stratum: store.sample_path(stratum) for stratum in ("card-disjoint", "game-disjoint")
    }
    missing = [str(p) for p in validation_samples.values() if not p.exists()]
    if missing:
        logger.error(
            "The corpus at %s has no validation samples (%s). Rebuild it with a "
            "build-corpus that writes validation/samples/.", config.corpus, ", ".join(missing),
        )
        return 1
    held_out = HeldOutCards(
        names=frozenset(manifest.held_out_cards), script_files=frozenset(),
        texts=frozenset(manifest.held_out_texts),
    )
    inherited = CorpusSplit(
        held_out_cards=manifest.held_out_cards,
        card_disjoint_games=frozenset(manifest.card_disjoint_games),
        game_disjoint_games=frozenset(manifest.game_disjoint_games),
    )
    logger.info(
        "Curated corpus at %s: %d training shards, %d gate-one record(s), "
        "%d card(s) held out.",
        config.corpus, len(training_shards), manifest.per_stratum.get("gate-one", 0),
        len(manifest.held_out_cards),
    )

    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config,
        held_out=held_out,
        inherited=inherited,
        training_shards=training_shards,
        validation_samples=validation_samples,
        gate_one_records=manifest.per_stratum.get("gate-one", 0),
        rarity=manifest.rarity,
        corpus_digest=manifest.digest(),
    ).execute()
```

Delete from `TrainEffectModelConfig`: `records_dir`, `holdout_permille`, `holdout_max_carriers`, `min_holdout_records` (keep `MIN_HOLDOUT_RECORDS` as the constant `check_holdout` defaults to), `workers`, `kind_mix`, `reserved_shards`. Delete `validate_corpus_flags` and `CORPUS_EXCLUSIVE_FLAGS` if nothing else uses them (grep first). Keep `split_from` and `require_split_from` for the variants.

- [ ] **Step 4: The loop loads samples and trains without a mixture**

In `TrainingLoop.__init__`, replace the `validation_shards` parameter with `validation_samples: Mapping[str, Path]` and add `gate_one_records: int`; store both. Delete `_class_quota` and `_sample_is_full`. Replace `_capture_validation` with:

```python
    def _load_validation(self) -> None:
        """Read the corpus's fixed samples; nothing is drawn here (FR-089)."""
        self.card_disjoint = load_shard(self.validation_samples["card-disjoint"])
        self.game_disjoint = load_shard(self.validation_samples["game-disjoint"])
        self.probe = self.card_disjoint[:PROBE_RECORDS] or self.game_disjoint[:PROBE_RECORDS]
        self.present = {
            sampling_class(record) for record in (*self.card_disjoint, *self.game_disjoint)
        }
        for stratum, records in (("card-disjoint", self.card_disjoint),
                                 ("game-disjoint", self.game_disjoint)):
            logger.info(
                "%s validation: %d records over %d games, %d classes",
                stratum, len(records), len({r.game_id for r in records}),
                len({sampling_class(r) for r in records}),
            )
```

In `execute`: call `self._load_validation()` in place of `_capture_validation()`; delete the `mix = renormalize_mix(...)` and the two mixture log lines; change the `check_holdout` call to `unique_text_records=self.gate_one_records` (drop the `text_of` computation if nothing else uses it); pass no `mix` to `_train_on_shard`.

Replace `_pools` with:

```python
    def _weighted(self, records: list, sidecars: SidecarCache) -> list[float]:
        """Rarity weights for a shard's records, from the manifest's table (FR-086)."""
        def text_of(record: EffectRecord) -> str | None:
            return ability_text_of(record, sidecars, self.surface)

        if self.rarity is not None and not self._rarity_reported:
            self._rarity_reported = True
            found, distinct = rarity_coverage((text_of(r) for r in records), self.rarity)
            share = 100.0 * found / distinct if distinct else 0.0
            report = logger.info if found else logger.warning
            report(
                "Rarity table names %d of this shard's %d distinct ability "
                "text(s) (%.1f%%); the rest weigh by this shard's own game count.",
                found, distinct, share,
            )
        return sample_weights(records, text_of, rarity=self.rarity)
```

In `_train_on_shard`: drop the `mix` parameter; replace `pools, weights = self._pools(training, sidecars)` with `weights = self._weighted(training, sidecars)` and `batches = batches_without_replacement(training, weights, batch_size=self.config.batch_size, rng=self.rng)`; replace `plan = plan_batch(...)` with `plan = next(batches)`; update the final `del` line. Import `batches_without_replacement` and drop `plan_batch`, `renormalize_mix`, `parse_kind_mix` from the imports.

Delete `src/effects/infrastructure/shard_sweep.py` and `tests/unit/effects/application/test_validation_capture.py`; drop the `sweep` import from `training_loop.py`.

- [ ] **Step 5: CLI**

In the train parser, delete `--records-dir`, `--kind-mix`, `--workers`, `--holdout-permille`, `--holdout-max-carriers`, `--min-holdout-records`, `--reserved-shards`. Make `--corpus` `required=True` if it is optional today. Update the config construction accordingly. Grep `records_dir` and `reserved_shards` under `src/effects` for the remaining raw-corpus helpers (`reserve_shards`, `reserved_shard_indices`, `resolve_holdout`, `records_name_held_out`, `shard_games` when only the sweep used it) and delete each one that has no caller left, with its tests.

- [ ] **Step 6: Run the full effects suite**

Run: `python -m pytest tests/unit/effects -q`
Expected: all pass. Tests of deleted helpers are deleted with them; tests that built a `TrainEffectModelConfig(records_dir=...)` switch to `corpus=`.

- [ ] **Step 7: Commit**

```bash
git add -A src/effects tests/unit/effects
git commit -m "refactor(effects): the trainer reads the curated corpus and derives nothing"
```

---

### Task 6: Validation scores the epoch's field set and prints gate metrics

**Files:**
- Modify: `src/effects/application/training_loop.py` (`execute`, `_validate`, `_loss_for`)
- Test: `tests/unit/effects/application/test_curriculum.py` (append)

**Interfaces:**
- `_loss_for(..., step, ...)` gains `fields` (a tuple from `fields_for_epoch`) and stops calling `active_fields` itself when given one; `_validate(..., fields=...)` passes it through.
- The epoch line becomes: `epoch %d | train %.4f | card-disjoint %.4f | game-disjoint %.4f | gate F1 %.3f | zone acc %.3f | deviance %.3f`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_loop_resets_the_stopper_when_the_curriculum_switches(monkeypatch):
    """Losses before and after the sparse group enables do not compare (FR-082)."""
    from effects.application import training_loop

    calls = []

    class Stopper:
        def __init__(self, patience): self.best = float("inf"); self.since_best = 0
        def update(self, loss): calls.append(("update", loss)); return True
        def reset(self): calls.append(("reset", None))
        @property
        def should_stop(self): return False

    monkeypatch.setattr(training_loop, "EarlyStopper", Stopper)
    config = TrainEffectModelConfig(corpus="x", steps_per_epoch=10, curriculum_epoch=2, epochs=3)
    assert training_loop.resets_at(config, epoch=1) is False
    assert training_loop.resets_at(config, epoch=2) is True
    assert training_loop.resets_at(config, epoch=3) is False
```

Add `resets_at(config, *, epoch) -> bool` to `training_loop.py`: `epoch == config.curriculum_epoch and config.curriculum_epoch > 1`. The monkeypatched stopper is there so the executor can extend this test to drive `execute()` with a stub model if they choose; the assertion on `resets_at` is the deliverable.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_curriculum.py -k resets -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'resets_at'`

- [ ] **Step 3: Implement**

In `training_loop.py`:

```python
def resets_at(config, *, epoch: int) -> bool:
    """Whether the early stopper forgets its best at the start of ``epoch``.

    True once, at the curriculum epoch: the loss gains twenty-five field terms
    there, so an earlier best would be a different objective's number.
    """
    return config.curriculum_epoch > 1 and epoch == config.curriculum_epoch
```

In `execute`, inside the epoch loop before the shard walk:

```python
            fields = fields_for_epoch(
                self.config, present=frozenset(self.present), epoch=epoch,
            )
            if resets_at(self.config, epoch=epoch):
                stopper.reset()
                logger.info(
                    "epoch %d enables the sparse field group (%d fields now); "
                    "the early stopper starts over", epoch, len(fields),
                )
```

Thread `fields` into `_train_on_shard(..., fields=fields)` and from there into `_loss_for(..., fields=fields)`; in `_loss_for`, replace the `present = ...; fields = active_fields(...)` lines with the parameter (keep a fallback `if fields is None: fields = active_fields(...)` for the gate-1 evaluator, which builds its own). Call `_validate(..., fields=fields)` for both strata and pass it through.

After the two validations, compute the gate metrics on the card-disjoint sample and log them on the epoch line:

```python
            from effects.application.gate_one import measure

            metrics = measure(
                self.card_disjoint, encoder, model,
                self._batcher(tokenizer, sidecars, widths), fields=fields,
            )
            logger.info(
                "epoch %d | train %.4f | card-disjoint %.4f | game-disjoint %.4f | "
                "gate F1 %.3f | zone acc %.3f | deviance %.3f%s",
                result.epoch, result.train_loss, result.card_disjoint_loss,
                result.game_disjoint_loss, metrics.affected_gate_f1,
                metrics.zone_outcome_accuracy, metrics.mean_poisson_deviance,
                _format_parts(card_parts),
            )
```

`measure` puts the models in eval mode and back; `_validate` already does the same. The batcher for validation must be built with `context_dropout=0.0` and `keyword_expand_p=0.0`: check `_batcher` and add a `training: bool = True` parameter that zeroes both when `False`, and pass `training=False` here and in `_validate`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/effects -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/effects/application/training_loop.py tests/unit/effects/application/test_curriculum.py
git commit -m "feat(effects): validation scores the epoch's field set and prints gate metrics"
```

---

### Task 7: Remove `plan_batch` and update the docs

**Files:**
- Modify: `src/effects/application/train_effect_model.py` (delete `plan_batch`)
- Modify: `tests/unit/effects/application/test_sampling.py` (delete `TestBatchPlanning`)
- Modify: `specs/023-ability-effect-model/spec.md`, `quickstart.md`

- [ ] **Step 1: Delete**

Delete `plan_batch` and `TestBatchPlanning`. Grep `plan_batch` under `src/` and `tests/` to confirm nothing else calls it.

- [ ] **Step 2: Spec**

Apply the FR-081, FR-082, FR-086, FR-088b, FR-095 and FR-125 amendments from the design record verbatim to `spec.md`. In `quickstart.md` § train, remove `--records-dir`, `--reserved-shards`, `--kind-mix`, `--workers`, `--curriculum-step` from the flag table, add `--curriculum-epoch`, and replace the paragraph on the startup sweep with one sentence: the trainer reads `validation/samples/` and never sweeps.

- [ ] **Step 3: Run everything**

Run: `python -m pytest tests/unit -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add -A src/effects tests/unit/effects specs/023-ability-effect-model
git commit -m "docs(effects): spec and quickstart follow the trainer rework"
```

---

### Task 8: Train and evaluate

**Files:** none (operational). Runs after the rebuilt corpus from the build-corpus plan exists.

- [ ] **Step 1: Train the full model**

```bash
python -u -m effects train-effect-model --corpus output/effects/corpus/ \
    --variant-scripts output/effects/variant-scripts/ --vocab-path models/effects/vocab-script.txt \
    | Out-File -Encoding utf8 models/effects/effect-model/$(Get-Date -Format MM.dd).log
```

Watch for, in the first epoch: `|g| encoder X, head Y (clipped per group at 1)` with the encoder norm no longer scaled by the head's; a shard line whose `loss` never goes negative; no `Sampling mixture` line; a `gate F1` figure on the epoch line.

- [ ] **Step 2: Train the identity baseline**

Only after Step 1 has finished. A second process on the GPU falls back to system memory and runs about fifty times slower.

```bash
python -u -m effects train-effect-model --variant identity --corpus output/effects/corpus/ \
    --variant-scripts output/effects/variant-scripts/ --vocab-path models/effects/vocab-script.txt \
    --split-from models/effects/effect-model/latest.pt
```

- [ ] **Step 3: Evaluate**

```bash
python -m effects evaluate-effect-model --checkpoint models/effects/effect-model/latest.pt \
    --baseline identity=models/effects/effect-model/identity/latest.pt
```

Record gate 1's three margins in the experiments log beside the per-epoch `gate F1` line from Step 1, so the two agree on what the card-disjoint sample measured.

---

## Self-review

- **Spec coverage.** FR-081 (Task 1), FR-082 (Tasks 3, 6), FR-086 (Tasks 2, 5), FR-088b (Task 5: `gate_one_records`), FR-089 (Task 5: samples), FR-095 (Task 4), FR-125 withdrawn (Task 5), docs (Task 7), baselines (Task 8).
- **Type consistency.** `fields_for_epoch(config, *, present, epoch)` from Task 3 is what Task 6 calls; `batches_without_replacement(records, weights, *, batch_size, rng)` from Task 2 is what Task 5 calls; `parameter_groups(encoder, model, identity_table=None)` from Task 4 is used in `execute`; `TrainingLoop(..., validation_samples=..., gate_one_records=...)` in Task 5 matches the test.
- **Ordering.** Task 5 deletes the sweep and its tests; Task 6 relies on Task 3's `fields_for_epoch` and Task 5's `self.present`. Tasks 1, 2, 3, 4 are independent of each other and can be done in any order before 5.
