# Curated Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `python -m effects build-corpus`, which reads the raw effect-record corpus once and writes a fixed training corpus, two fixed validation strata, and a manifest — so repeated training runs differ only in the hyperparameters under test.

**Architecture:** Two parallel passes over the shard corpus. Pass one surveys: per-provenance-key game sets and record counts, per-class counts, which games name a held-out card, and a bounded per-key heap of record hashes that yields an exact per-text cap threshold. The main process folds provenance keys to ability texts through one `SidecarCache`, decides the split and the per-class write targets, and writes a manifest. Pass two re-reads each shard, filters by those decisions, and writes gzip shards into `training/`, `validation/card-disjoint/` and `validation/game-disjoint/`. `train-effect-model --corpus` then reads the dataset instead of the raw corpus.

**Tech Stack:** Python 3.14, `concurrent.futures.ProcessPoolExecutor`, `hashlib.blake2b`, `heapq`, `gzip`, pytest.

**Spec:** `specs/2026-09-05-ability-effect-model.md` § Curated corpus; `specs/023-ability-effect-model/spec.md` FR-135 … FR-147. Rationale: `experiments/2026-09-04-ability-effect-model-design.md` § "Training reads half a corpus whose shape collection chose".

## Global Constraints

- `effects` imports from `sealed` and `price_predictor` only over the surface FR-002 declares, and **never** from `sealed.application`. `tests/unit/effects/test_import_boundaries.py` asserts it.
- Hexagonal layout: `domain` holds pure logic with no I/O, `application` orchestrates, `infrastructure` touches the filesystem and subprocesses. Domain modules must not import `application` or `infrastructure`.
- Never hash with the built-in `hash()` over `str` — it is salted per process (FR-088a). Use `zlib.crc32` or `hashlib.blake2b`.
- Every function a process pool calls must be a module-level function, so it pickles.
- TDD: write the failing test, run it, watch it fail for the right reason, then implement.
- Card names crossing the converted-text/Forge boundary are folded with `fold_card_name` on **both** sides. Converted text is lowercase; Forge and the records carry printed case.
- Type hints on every new public function. `from __future__ import annotations` at the top of every new module.
- Run tests with `python -m pytest`.

---

### Task 1: The manifest

**Files:**
- Create: `src/effects/domain/corpus_manifest.py`
- Test: `tests/unit/effects/domain/test_corpus_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SourceShard(name: str, size: int)`, `ClassCounts(read: int, kept: int, dropped_by_cap: int, unique_texts: int)`, and `CorpusManifest` with `as_dict() -> dict`, `from_dict(dict) -> CorpusManifest`, `digest() -> str`, and `drift(current: tuple[SourceShard, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/effects/domain/test_corpus_manifest.py
from __future__ import annotations

from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, SourceShard


def manifest(**overrides) -> CorpusManifest:
    base = dict(
        seed=42,
        surface="script",
        vocab_path="models/effects/vocab-script.txt",
        holdout_permille=20,
        holdout_max_carriers=8,
        text_cap=200,
        card_disjoint_text_cap=50,
        game_disjoint_target=1000,
        training_records=0,
        class_mix={"rewrite": 1.0},
        held_out_cards=("soul echo",),
        card_disjoint_games=("run.0-a.1",),
        game_disjoint_games=("run.0-a.2",),
        rarity={"deals 3 damage": 7},
        sources=(SourceShard(name="depleted/run.0-a.jsonl.gz", size=1234),),
        per_class={"rewrite": ClassCounts(read=10, kept=4, dropped_by_cap=6, unique_texts=2)},
        shortfall={},
    )
    base.update(overrides)
    return CorpusManifest(**base)


def test_round_trips_through_a_dict():
    original = manifest()
    assert CorpusManifest.from_dict(original.as_dict()) == original


def test_digest_is_stable_across_instances():
    assert manifest().digest() == manifest().digest()


def test_digest_changes_when_a_recorded_decision_changes():
    assert manifest().digest() != manifest(text_cap=100).digest()


def test_digest_ignores_the_order_a_source_list_was_built_in():
    a = manifest(sources=(
        SourceShard(name="b.jsonl.gz", size=2), SourceShard(name="a.jsonl.gz", size=1),
    ))
    b = manifest(sources=(
        SourceShard(name="a.jsonl.gz", size=1), SourceShard(name="b.jsonl.gz", size=2),
    ))
    assert a.digest() == b.digest()


def test_drift_names_added_and_removed_shards():
    current = (
        SourceShard(name="a.jsonl.gz", size=1),
        SourceShard(name="c.jsonl.gz", size=3),
    )
    added, removed, resized = manifest().drift(current)
    assert added == ("a.jsonl.gz", "c.jsonl.gz")
    assert removed == ("depleted/run.0-a.jsonl.gz",)
    assert resized == ()


def test_drift_reports_a_shard_that_grew():
    current = (SourceShard(name="depleted/run.0-a.jsonl.gz", size=9999),)
    added, removed, resized = manifest().drift(current)
    assert (added, removed) == ((), ())
    assert resized == ("depleted/run.0-a.jsonl.gz",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'effects.domain.corpus_manifest'`

- [ ] **Step 3: Write the implementation**

```python
# src/effects/domain/corpus_manifest.py
"""What a curated dataset records about how it was built (FR-143).

The manifest is the dataset's identity. A training run reads its split, its
rarity table and its caps from here rather than deriving them, and a checkpoint
records ``digest()`` so ``evaluate-effect-model`` can refuse a dataset that has
been rebuilt since (FR-147) — a rebuild is a different split, and scoring the
gates against it would score them partly on games the model trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceShard:
    """One raw shard the dataset was built from, by path and size.

    Size rather than a content hash: a shard is append-only and uniquely named,
    so its length is what changes when a collection run extends the corpus, and
    hashing tens of gigabytes to learn that would cost more than the build.
    """

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class ClassCounts:
    """What one sampling class contributed, read against kept.

    ``unique_texts`` is the number no record count can stand in for: it is what
    says whether the per-text cap trimmed the head or flattened the tail
    (FR-145).
    """

    read: int
    kept: int
    dropped_by_cap: int
    unique_texts: int


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Every decision ``build-corpus`` made, and what it made them from."""

    seed: int
    #: The encoding surface the rarity table's text keys were built on.
    #: `surface_of(vocab_path)` decides it, and a table built on the other
    #: surface keys every text differently while looking exactly as valid.
    surface: str
    vocab_path: str
    holdout_permille: int
    holdout_max_carriers: int
    text_cap: int
    card_disjoint_text_cap: int
    game_disjoint_target: int
    training_records: int
    class_mix: dict[str, float]
    held_out_cards: tuple[str, ...]
    card_disjoint_games: tuple[str, ...]
    game_disjoint_games: tuple[str, ...]
    rarity: dict[str, int]
    sources: tuple[SourceShard, ...]
    per_class: dict[str, ClassCounts]
    shortfall: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [asdict(shard) for shard in self.sources]
        data["per_class"] = {
            name: asdict(counts) for name, counts in self.per_class.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusManifest:
        return cls(
            seed=int(data["seed"]),
            surface=str(data["surface"]),
            vocab_path=str(data["vocab_path"]),
            holdout_permille=int(data["holdout_permille"]),
            holdout_max_carriers=int(data["holdout_max_carriers"]),
            text_cap=int(data["text_cap"]),
            card_disjoint_text_cap=int(data["card_disjoint_text_cap"]),
            game_disjoint_target=int(data["game_disjoint_target"]),
            training_records=int(data["training_records"]),
            class_mix={k: float(v) for k, v in data["class_mix"].items()},
            held_out_cards=tuple(data["held_out_cards"]),
            card_disjoint_games=tuple(data["card_disjoint_games"]),
            game_disjoint_games=tuple(data["game_disjoint_games"]),
            rarity={k: int(v) for k, v in data["rarity"].items()},
            sources=tuple(
                SourceShard(name=s["name"], size=int(s["size"]))
                for s in data["sources"]
            ),
            per_class={
                name: ClassCounts(**{k: int(v) for k, v in counts.items()})
                for name, counts in data["per_class"].items()
            },
            shortfall={k: int(v) for k, v in data["shortfall"].items()},
        )

    def digest(self) -> str:
        """A stable hash over every decision, insensitive to listing order.

        Sorted keys and a sorted source list, so two builds of the same dataset
        agree whatever order the filesystem enumerated shards in.
        """
        payload = self.as_dict()
        payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

    def drift(
        self, current: tuple[SourceShard, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Shards added, removed and resized since the build, each sorted.

        What ``--verify`` reports (FR-144). A resized shard is one a collection
        run appended to after this dataset read it.
        """
        was = {shard.name: shard.size for shard in self.sources}
        now = {shard.name: shard.size for shard in current}
        added = tuple(sorted(set(now) - set(was)))
        removed = tuple(sorted(set(was) - set(now)))
        resized = tuple(
            sorted(name for name in set(was) & set(now) if was[name] != now[name])
        )
        return added, removed, resized
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_manifest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/corpus_manifest.py tests/unit/effects/domain/test_corpus_manifest.py
git commit -m "feat(effects): record what a curated dataset was built from"
```

---

### Task 2: The selection policy

**Files:**
- Create: `src/effects/domain/corpus_curation.py`
- Test: `tests/unit/effects/domain/test_corpus_curation.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `record_hash(record_id: str, *, seed: int) -> int` — a stable value in `[0, 2**64)`.
  - `CapHeap(cap: int)` with `offer(value: int) -> None`, `merge(other: CapHeap) -> None`, `values() -> tuple[int, ...]`, `threshold() -> int | None`.
  - `keeps(value: int, threshold: int | None) -> bool`.
  - `class_targets(available: Mapping[str, int], mix: Mapping[str, float], *, ceiling: int = 0) -> tuple[dict[str, int], dict[str, int]]` returning `(targets, shortfall)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/effects/domain/test_corpus_curation.py
from __future__ import annotations

import pytest

from effects.domain.corpus_curation import (
    CapHeap, class_targets, keeps, record_hash,
)


def test_record_hash_is_stable_and_seed_dependent():
    assert record_hash("run.0-a.7", seed=42) == record_hash("run.0-a.7", seed=42)
    assert record_hash("run.0-a.7", seed=42) != record_hash("run.0-a.7", seed=43)


def test_record_hash_fits_in_64_bits():
    assert 0 <= record_hash("run.0-a.7", seed=42) < 2**64


def test_a_heap_under_its_cap_sets_no_threshold():
    heap = CapHeap(3)
    for value in (10, 20):
        heap.offer(value)
    assert heap.threshold() is None


def test_a_full_heap_keeps_the_cap_smallest_values():
    heap = CapHeap(3)
    for value in (50, 10, 40, 20, 30):
        heap.offer(value)
    assert sorted(heap.values()) == [10, 20, 30]
    assert heap.threshold() == 30


def test_merging_two_heaps_keeps_the_cap_smallest_of_the_union():
    left, right = CapHeap(3), CapHeap(3)
    for value in (50, 10, 40):
        left.offer(value)
    for value in (5, 60, 35):
        right.offer(value)
    left.merge(right)
    assert sorted(left.values()) == [5, 10, 35]


def test_keeps_admits_exactly_the_values_at_or_below_the_threshold():
    assert keeps(30, 30) is True
    assert keeps(31, 30) is False
    assert keeps(10**19, None) is True


def test_class_targets_are_set_by_the_scarcest_class():
    # rewrite supplies 100 against a 10% share, so the whole dataset is 1000.
    targets, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
    )
    assert targets == {"rewrite": 100, "combat": 900}
    assert shortfall == {}


def test_class_targets_never_ask_for_more_than_a_class_holds():
    targets, _ = class_targets(
        {"rewrite": 10, "combat": 10}, {"rewrite": 0.5, "combat": 0.5},
    )
    assert all(targets[name] <= 10 for name in targets)


def test_a_ceiling_scales_every_class_down_together():
    targets, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
        ceiling=500,
    )
    assert targets == {"rewrite": 50, "combat": 450}
    assert shortfall == {}


def test_a_ceiling_above_what_the_mixture_supports_is_reported_as_shortfall():
    _, shortfall = class_targets(
        {"rewrite": 100, "combat": 5000}, {"rewrite": 0.1, "combat": 0.9},
        ceiling=4000,
    )
    assert shortfall == {"rewrite": 300, "combat": 2700}


def test_a_class_the_corpus_lacks_entirely_drops_out_of_the_mixture():
    targets, _ = class_targets({"combat": 500}, {"rewrite": 0.1, "combat": 0.9})
    assert "rewrite" not in targets
    assert targets["combat"] == 500


def test_class_targets_rejects_a_mixture_that_sums_to_nothing():
    with pytest.raises(ValueError, match="no positive share"):
        class_targets({"combat": 5}, {"combat": 0.0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_curation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'effects.domain.corpus_curation'`

- [ ] **Step 3: Write the implementation**

```python
# src/effects/domain/corpus_curation.py
"""Which records a curated training corpus keeps (FR-138, FR-139, FR-140).

Two decisions, both shaped so a worker can apply them to one shard without
talking to any other worker. The per-text cap becomes a numeric threshold on a
hash of the record id, computed once from a survey of the whole corpus; a
worker then keeps a record by comparing one number. The class mixture becomes a
per-class record target, applied the same way.

The cap keeps a *random* subset rather than the first N: a text's earliest
records are its earliest games, and keeping those would describe one opening
board over and over — the bias the mana reservoir sampler exists to avoid, on
the other side of the pipeline.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping

#: uint64, which is the width the threshold comparison is over.
_HASH_BYTES = 8


def record_hash(record_id: str, *, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for one record.

    ``blake2b`` keyed by the seed rather than the built-in ``hash()``, which is
    salted per process: a threshold computed in the survey pass would admit a
    different set of records in the write pass, and nothing would say so
    (FR-088a).
    """
    digest = hashlib.blake2b(
        record_id.encode("utf-8"),
        digest_size=_HASH_BYTES,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


class CapHeap:
    """The ``cap`` smallest record hashes seen for one ability text.

    A max-heap of negated values, so the largest retained hash sits at the root
    and is exactly the threshold that admits ``cap`` records and no more. Held
    bounded rather than collecting every hash: the corpus has tens of millions
    of records and this runs once per unique text.
    """

    __slots__ = ("cap", "_heap")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._heap: list[int] = []

    def offer(self, value: int) -> None:
        if self.cap <= 0:
            return
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, -value)
        elif -value > self._heap[0]:
            heapq.heappushpop(self._heap, -value)

    def merge(self, other: CapHeap) -> None:
        """Absorb another heap's values, keeping the cap smallest of the union.

        Valid because the k smallest of a union are drawn from each side's own
        k smallest, which is what lets the survey run one heap per worker and
        combine them afterwards.
        """
        for negated in other._heap:
            self.offer(-negated)

    def values(self) -> tuple[int, ...]:
        return tuple(-negated for negated in self._heap)

    def threshold(self) -> int | None:
        """The largest retained hash, or None while under the cap.

        None means every record of this text is kept, which is the case for
        every text the cap never reaches — most of them.
        """
        if self.cap <= 0 or len(self._heap) < self.cap:
            return None
        return -self._heap[0]


def keeps(value: int, threshold: int | None) -> bool:
    """Whether a record's hash is admitted by its text's threshold."""
    return True if threshold is None else value <= threshold


def class_targets(
    available: Mapping[str, int],
    mix: Mapping[str, float],
    *,
    ceiling: int = 0,
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class write targets, and the shortfall against a requested ceiling.

    The scarcest class sets the size of the whole dataset: the largest total
    the mixture can be written at is ``min(available[c] / share[c])``, and
    every class then takes its share of that. Filling the abundant classes to
    their own capacity instead would write the mixture the disk happens to
    hold, which is the shape the dataset exists to correct.

    ``ceiling`` (``--training-records``) lowers that total. A ceiling above
    what the mixture supports cannot be met, and the difference is returned per
    class rather than silently under-filled (FR-139).
    """
    shares = {
        name: float(share)
        for name, share in mix.items()
        if share > 0 and available.get(name, 0) > 0
    }
    if not shares:
        raise ValueError(f"no positive share with records available: mix={dict(mix)}")
    total_share = sum(shares.values())
    supported = min(
        available[name] * total_share / share for name, share in shares.items()
    )
    total = min(supported, float(ceiling)) if ceiling > 0 else supported
    targets = {
        name: min(available[name], int(total * share / total_share))
        for name, share in shares.items()
    }
    shortfall: dict[str, int] = {}
    if ceiling > 0 and ceiling > supported:
        for name, share in shares.items():
            missing = int(ceiling * share / total_share) - targets[name]
            if missing > 0:
                shortfall[name] = missing
    return targets, shortfall
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_curation.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/corpus_curation.py tests/unit/effects/domain/test_corpus_curation.py
git commit -m "feat(effects): decide the cap and the mixture as per-worker thresholds"
```

---

### Task 3: A gzip shard writer

**Files:**
- Modify: `src/effects/infrastructure/record_io.py` — append after the `ShardWriter` class (around line 663), before the `SHARD_GLOBS` constant.
- Test: `tests/unit/effects/infrastructure/test_record_io.py` — append.

**Interfaces:**
- Consumes: `format_record_line` and `read_shard`, both already in this module.
- Produces: `write_shard(path: Path, records: Iterable[EffectRecord]) -> int`, returning the number of records written.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/effects/infrastructure/test_record_io.py`. Reuse whatever `EffectRecord` factory that file already has; read the file first and call the existing helper rather than inventing one. If there is none, build a record with `record_from_dict` on a dict copied from an existing test.

```python
def test_write_shard_round_trips_through_the_reader(tmp_path):
    from effects.infrastructure.record_io import read_shard, write_shard

    records = [a_record(record_id=f"run.0-a.{i}") for i in range(5)]
    path = tmp_path / "curated.0-a.jsonl.gz"

    assert write_shard(path, records) == 5

    assert [r.record_id for r in read_shard(path)] == [r.record_id for r in records]


def test_write_shard_creates_the_directory_it_writes_into(tmp_path):
    from effects.infrastructure.record_io import write_shard

    path = tmp_path / "training" / "curated.0-a.jsonl.gz"
    write_shard(path, [a_record(record_id="run.0-a.0")])
    assert path.exists()


def test_write_shard_writes_a_readable_empty_shard_for_no_records(tmp_path):
    from effects.infrastructure.record_io import read_shard, write_shard

    path = tmp_path / "curated.0-a.jsonl.gz"
    assert write_shard(path, []) == 0
    assert list(read_shard(path)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/infrastructure/test_record_io.py -k write_shard -v`
Expected: FAIL with `ImportError: cannot import name 'write_shard'`

- [ ] **Step 3: Write the implementation**

```python
def write_shard(path: Path, records: Iterable[EffectRecord]) -> int:
    """Write a whole shard at once, gzipped; return the record count.

    One gzip member rather than one per 256 records: that blocking exists so a
    worker killed mid-run truncates only its last block, and a curated shard is
    written in a single pass by a process that either finishes or leaves
    nothing behind. ``read_shard`` reads both shapes.
    """
    import gzip

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(format_record_line(record))
            written += 1
    return written
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/infrastructure/test_record_io.py -v`
Expected: PASS, including the three new tests

- [ ] **Step 5: Commit**

```bash
git add src/effects/infrastructure/record_io.py tests/unit/effects/infrastructure/test_record_io.py
git commit -m "feat(effects): write a whole shard in one gzip member"
```

---

### Task 4: The survey pass

**Files:**
- Create: `src/effects/application/build_corpus.py`
- Modify: `src/effects/application/train_effect_model.py` — extract `text_for_key` out of `ability_text_of` (around line 1013)
- Test: `tests/unit/effects/application/test_build_corpus_survey.py`
- Test: `tests/unit/effects/application/test_train_effect_model.py` — one added test for the extraction

**Interfaces:**
- Consumes: `CapHeap`, `record_hash` (Task 2); `SourceShard` (Task 1); `record_names_held_out_card`, `HeldOutCards`, `sampling_class` (existing, `train_effect_model`); `iter_shards`, `read_shard` (existing, `record_io`).
- Produces:
  - `ability_key(record: EffectRecord) -> str | None` — the record's whole `ability` tuple as one canonical string, `None` when it has none.
  - `parse_ability_key(rendered: str) -> tuple[ProvenanceKey, ...]` — the inverse.
  - `game_hash(game_id: str) -> int`.
  - `SurveyConfig(records_dir: str, held_out_names: frozenset[str], held_out_script_files: frozenset[str], text_cap: int, seed: int)`.
  - `init_survey_worker(config: SurveyConfig) -> None` and `survey_shard(relative: str) -> ShardSurvey` — both module-level, so a process pool can use them as initializer and map function.
  - `ShardSurvey` and `Survey` dataclasses, `merge_surveys(parts) -> Survey`, `run_survey(records_dir, *, config, workers, progress_every=50) -> Survey`.
  - In `train_effect_model`: `text_for_key(key, sidecars, surface: str) -> str | None`.

**Why keys and not texts:** resolving a provenance key to its ability text needs a `SidecarCache` over 33,680 sidecar files. Loading one per worker process would cost more than the scan. The survey therefore keys everything on the provenance key, which is in the record, and the main process folds keys to texts once through the one `SidecarCache` it already builds for the holdout.

- [ ] **Step 1: Write the failing test for the `text_for_key` extraction**

Add to `tests/unit/effects/application/test_train_effect_model.py`. Use whatever fake-sidecar helper that file already has for `ability_text_of`; read it first.

```python
def test_text_for_key_and_ability_text_of_agree_on_one_key():
    from effects.application.train_effect_model import ability_text_of, text_for_key

    record = a_record_with_ability()          # existing helper
    sidecars = a_sidecar_cache()              # existing helper
    key = record.ability[0]

    assert text_for_key(key, sidecars, "script") == ability_text_of(
        record, sidecars, "script"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_train_effect_model.py -k text_for_key -v`
Expected: FAIL with `ImportError: cannot import name 'text_for_key'`

- [ ] **Step 3: Extract `text_for_key`**

Replace the body of `ability_text_of` in `src/effects/application/train_effect_model.py`:

```python
def text_for_key(key, sidecars, surface: str) -> str | None:
    """One provenance key's text on ``surface``, or None where it has none.

    The single key-level definition. ``build-corpus`` surveys by provenance key
    and folds to text afterwards, and a second spelling of this would key the
    rarity table differently from every lookup the trainer makes against it.

    Falls back to the key itself where the sidecar carries neither surface, so
    a line that cannot be read is still a distinct unit for rarity weighting
    rather than collapsing into every other unreadable line.
    """
    from effects.domain.ability_encoder import encoding_text

    try:
        line = sidecars.line_for(key)
    except KeyError:
        return None
    if line is None:
        return None
    text = encoding_text(line, sidecars.prose_for(key), surface)
    return text or f"{key.script_file}:{key.trait_kind}:{key.index_within_kind}"


def ability_text_of(
    record: EffectRecord, sidecars, surface: str = "prose",
) -> str | None:
    """The acting line's text on ``surface``, or None where no line acts.

    ``combat`` and the ``attackers``/``blockers`` subkinds have none, which is
    what makes them sample uniformly rather than by rarity.
    """
    if not record.ability:
        return None
    for key in record.ability:
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None
```

- [ ] **Step 4: Run the trainer's whole test module**

Run: `python -m pytest tests/unit/effects/application/test_train_effect_model.py -v`
Expected: PASS, including the new test. `ability_text_of` must keep its existing behaviour exactly — if any existing test fails, the extraction changed semantics and must be fixed, not the test.

- [ ] **Step 5: Write the failing tests for the survey**

```python
# tests/unit/effects/application/test_build_corpus_survey.py
from __future__ import annotations

from effects.application.build_corpus import (
    Survey, SurveyConfig, ability_key, game_hash, init_survey_worker,
    merge_surveys, parse_ability_key, run_survey, survey_shard,
)
from effects.application.train_effect_model import HeldOutCards
from effects.domain.provenance import ProvenanceKey
from effects.infrastructure.record_io import write_shard


def test_ability_key_round_trips_through_its_rendering():
    keys = (
        ProvenanceKey("cardsfolder/s/swamp.txt", 0, "spell", 1),
        ProvenanceKey("tokenscripts/role_wicked.txt", 1, "static", 0),
    )
    record = a_record(ability=keys)
    assert parse_ability_key(ability_key(record)) == keys


def test_ability_key_is_none_for_a_record_with_no_acting_line():
    assert ability_key(a_record(ability=())) is None


def test_game_hash_is_stable_across_calls():
    assert game_hash("run.0-a.3") == game_hash("run.0-a.3")


def test_survey_counts_records_classes_and_games(tmp_path):
    shard = tmp_path / "run.0-a.jsonl.gz"
    write_shard(shard, [
        a_record(record_id="run.0-a.0", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.1", game_id="g1", kind="rewrite"),
        a_record(record_id="run.0-a.2", game_id="g2", kind="combat", ability=()),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path),
            held_out_names=frozenset(),
            held_out_script_files=frozenset(),
            text_cap=200,
            seed=42,
        ),
        workers=1,
    )

    assert survey.records == 3
    assert survey.class_records["rewrite"] == 2
    assert survey.class_records["combat"] == 1
    assert survey.games == frozenset({"g1", "g2"})
    assert survey.held_out_games == frozenset()


def test_survey_marks_every_game_naming_a_held_out_card(tmp_path):
    shard = tmp_path / "run.0-a.jsonl.gz"
    write_shard(shard, [
        a_record(record_id="run.0-a.0", game_id="g1", entity_names=("Soul Echo",)),
        a_record(record_id="run.0-a.1", game_id="g2", entity_names=("Grizzly Bears",)),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path),
            held_out_names=frozenset({"soul echo"}),
            held_out_script_files=frozenset(),
            text_cap=200,
            seed=42,
        ),
        workers=1,
    )

    assert survey.held_out_games == frozenset({"g1"})


def test_survey_counts_games_not_records_for_rarity(tmp_path):
    # One ability, three records, two games: rarity must read 2.
    keys = (ProvenanceKey("cardsfolder/b/bear.txt", 0, "spell", 0),)
    write_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(record_id="run.0-a.0", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.1", game_id="g1", ability=keys),
        a_record(record_id="run.0-a.2", game_id="g2", ability=keys),
    ])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path), held_out_names=frozenset(),
            held_out_script_files=frozenset(), text_cap=200, seed=42,
        ),
        workers=1,
    )

    rendered = ability_key(a_record(ability=keys))
    assert len(survey.key_games[rendered]) == 2
    assert survey.key_records[rendered] == 3


def test_survey_records_each_shards_name_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    write_shard(tmp_path / "depleted" / "run.0-a.jsonl.gz", [a_record()])

    survey = run_survey(
        tmp_path,
        config=SurveyConfig(
            records_dir=str(tmp_path), held_out_names=frozenset(),
            held_out_script_files=frozenset(), text_cap=200, seed=42,
        ),
        workers=1,
    )

    assert [s.name for s in survey.shards] == ["depleted/run.0-a.jsonl.gz"]


def test_merging_shard_surveys_unions_games_and_sums_records():
    config = SurveyConfig(
        records_dir=".", held_out_names=frozenset(),
        held_out_script_files=frozenset(), text_cap=2, seed=42,
    )
    init_survey_worker(config)
    # Two shards, same ability, disjoint games.
    merged = merge_surveys([
        a_shard_survey(key="k", games={1}, records=1, hashes=(10,)),
        a_shard_survey(key="k", games={2}, records=1, hashes=(20,)),
    ])
    assert merged.key_records["k"] == 2
    assert len(merged.key_games["k"]) == 2
    assert sorted(merged.key_heaps["k"].values()) == [10, 20]
```

Write the `a_record` and `a_shard_survey` helpers at the top of the file. `a_record` must build a real `EffectRecord`; copy an existing factory from `tests/unit/effects/application/test_train_effect_model.py` or `tests/unit/effects/infrastructure/test_record_io.py` rather than inventing one, and give it `ability`, `game_id`, `record_id`, `kind` and `entity_names` keyword arguments.

- [ ] **Step 6: Run them to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus_survey.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'effects.application.build_corpus'`

- [ ] **Step 7: Write the survey**

```python
# src/effects/application/build_corpus.py
"""Build a fixed training corpus and two fixed validation strata (FR-135).

Two passes. The survey reads every shard and reports what the whole corpus
holds; the main process folds provenance keys to ability texts, decides the
split and the write targets, and the write pass re-reads each shard and keeps
what the decisions admit.

The survey keys on the provenance key rather than the ability text on purpose.
Resolving a key to its text needs a ``SidecarCache`` over tens of thousands of
sidecar files, and building one per worker process would cost more than the
scan it serves. Keys are in the record; the fold happens once, in the process
that already has a cache for the holdout.
"""

from __future__ import annotations

import logging
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.corpus_curation import CapHeap, record_hash
from effects.domain.corpus_manifest import SourceShard
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

_KEY_FIELD_SEPARATOR = "|"
_KEY_SEPARATOR = ";"


def ability_key(record: EffectRecord) -> str | None:
    """A record's whole acting-ability key tuple, as one hashable string.

    The whole tuple rather than its first element: ``ability_text_of`` resolves
    the first key that the sidecars can read, so a survey keyed on the first
    key alone would group records the trainer separates whenever the first key
    names a script with no sidecar.
    """
    if not record.ability:
        return None
    return _KEY_SEPARATOR.join(
        _KEY_FIELD_SEPARATOR.join(
            (key.script_file, str(key.face), key.trait_kind, str(key.index_within_kind))
        )
        for key in record.ability
    )


def parse_ability_key(rendered: str) -> tuple[ProvenanceKey, ...]:
    """The inverse of :func:`ability_key`."""
    keys = []
    for part in rendered.split(_KEY_SEPARATOR):
        script_file, face, trait_kind, index = part.split(_KEY_FIELD_SEPARATOR)
        keys.append(ProvenanceKey(script_file, int(face), trait_kind, int(index)))
    return tuple(keys)


def game_hash(game_id: str) -> int:
    """A stable 32-bit id for a game, for counting distinct games cheaply.

    Rarity needs the *number* of distinct games per text, not their names, and
    holding tens of millions of game-id strings across worker results costs
    hundreds of megabytes for a figure that is a cardinality. ``crc32`` rather
    than the built-in ``hash()``, which is salted per process (FR-088a).
    """
    return zlib.crc32(game_id.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """What every survey worker needs, sent once through the pool initializer."""

    records_dir: str
    held_out_names: frozenset[str]
    held_out_script_files: frozenset[str]
    text_cap: int
    seed: int


@dataclass(slots=True)
class ShardSurvey:
    """One shard's contribution. Plain data, so it pickles back cheaply."""

    name: str
    size: int
    records: int
    key_games: dict[str, set[int]] = field(default_factory=dict)
    key_records: Counter[str] = field(default_factory=Counter)
    key_hashes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    class_records: Counter[str] = field(default_factory=Counter)
    #: Records this shard's own per-key heap admitted, per class. An upper
    #: bound on what the text-level cap will keep — several keys can fold to
    #: one text, and only the main process knows which — and what
    #: ``class_targets`` divides the mixture over. The manifest's ``kept`` is
    #: the exact figure and comes from the write pass.
    class_capped_records: Counter[str] = field(default_factory=Counter)
    #: Each key's game ids as strings, for keys carried by a held-out game.
    #: The card-disjoint cap admits whole games (FR-142) and so needs the ids
    #: themselves, not the hashed cardinality rarity counts with.
    held_out_text_games: dict[str, set[str]] = field(default_factory=dict)
    held_out_games: set[str] = field(default_factory=set)
    games: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Survey:
    """What the whole corpus holds."""

    shards: tuple[SourceShard, ...]
    records: int
    key_games: dict[str, set[int]]
    key_records: Counter[str]
    key_heaps: dict[str, CapHeap]
    class_records: Counter[str]
    class_capped_records: Counter[str]
    held_out_text_games: dict[str, set[str]]
    held_out_games: frozenset[str]
    games: frozenset[str]


_CONFIG: SurveyConfig | None = None


def init_survey_worker(config: SurveyConfig) -> None:
    """Pool initializer: hand every worker the config once, not per shard."""
    global _CONFIG
    _CONFIG = config


def survey_shard(relative: str) -> ShardSurvey:
    """Survey one shard. Module-level so a process pool can pickle it."""
    from effects.application.train_effect_model import (
        HeldOutCards, record_names_held_out_card, sampling_class,
    )
    from effects.domain.corpus_curation import keeps
    from effects.infrastructure.record_io import read_shard

    config = _CONFIG
    assert config is not None, "init_survey_worker was not run"
    held_out = HeldOutCards(
        names=config.held_out_names, script_files=config.held_out_script_files,
    )
    path = Path(config.records_dir) / relative
    out = ShardSurvey(name=relative, size=path.stat().st_size, records=0)
    heaps: dict[str, CapHeap] = {}

    # Held for the second walk: a record's own heap admission is only known
    # once the shard's heaps are final, so the class tally cannot be inline.
    seen: list[tuple[str, str | None, int]] = []

    for record in read_shard(path):
        out.records += 1
        out.games.add(record.game_id)
        name = sampling_class(record)
        out.class_records[name] += 1
        held = record_names_held_out_card(record, held_out)
        if held:
            out.held_out_games.add(record.game_id)
        key = ability_key(record)
        value = record_hash(record.record_id, seed=config.seed)
        seen.append((name, key, value))
        if key is None:
            continue
        out.key_records[key] += 1
        out.key_games.setdefault(key, set()).add(game_hash(record.game_id))
        if held:
            out.held_out_text_games.setdefault(key, set()).add(record.game_id)
        heap = heaps.get(key)
        if heap is None:
            heap = heaps[key] = CapHeap(config.text_cap)
        heap.offer(value)

    out.key_hashes = {key: heap.values() for key, heap in heaps.items()}
    thresholds = {key: heap.threshold() for key, heap in heaps.items()}
    for name, key, value in seen:
        # A record with no acting line has no text and so no cap; it counts
        # toward its class in full (FR-087).
        if key is None or keeps(value, thresholds[key]):
            out.class_capped_records[name] += 1
    return out


def merge_surveys(parts) -> Survey:
    """Combine shard surveys into one corpus-wide picture."""
    config = _CONFIG
    cap = config.text_cap if config is not None else 0
    shards: list[SourceShard] = []
    records = 0
    key_games: dict[str, set[int]] = defaultdict(set)
    key_records: Counter[str] = Counter()
    key_heaps: dict[str, CapHeap] = {}
    class_records: Counter[str] = Counter()
    held_out_games: set[str] = set()
    games: set[str] = set()

    class_capped: Counter[str] = Counter()
    held_out_text_games: dict[str, set[str]] = defaultdict(set)

    for part in parts:
        shards.append(SourceShard(name=part.name, size=part.size))
        records += part.records
        key_records.update(part.key_records)
        class_records.update(part.class_records)
        class_capped.update(part.class_capped_records)
        held_out_games |= part.held_out_games
        games |= part.games
        for key, ids in part.held_out_text_games.items():
            held_out_text_games[key] |= ids
        for key, hashed in part.key_games.items():
            key_games[key] |= hashed
        for key, values in part.key_hashes.items():
            heap = key_heaps.get(key)
            if heap is None:
                heap = key_heaps[key] = CapHeap(cap)
            for value in values:
                heap.offer(value)

    return Survey(
        shards=tuple(sorted(shards, key=lambda s: s.name)),
        records=records,
        key_games=dict(key_games),
        key_records=key_records,
        key_heaps=key_heaps,
        class_records=class_records,
        class_capped_records=class_capped,
        held_out_text_games=dict(held_out_text_games),
        held_out_games=frozenset(held_out_games),
        games=frozenset(games),
    )


def shard_names(records_dir: Path) -> list[str]:
    """Every shard under the corpus root, as posix paths relative to it."""
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return [shard.relative_to(root).as_posix() for shard in iter_shards(root)]


def run_survey(
    records_dir,
    *,
    config: SurveyConfig,
    workers: int | None = None,
    progress_every: int = 50,
) -> Survey:
    """Survey every shard, in parallel, reporting progress as it goes."""
    root = Path(records_dir)
    names = shard_names(root)
    if not names:
        raise ValueError(f"{root}: no shards to build a corpus from")
    logger.info("Surveying %d shard(s) under %s", len(names), root)

    init_survey_worker(config)          # so a workers=1 run needs no pool
    if workers is not None and workers <= 1:
        parts = []
        for index, name in enumerate(names, start=1):
            parts.append(survey_shard(name))
            if index % progress_every == 0:
                logger.info("Surveyed %d of %d shard(s)", index, len(names))
        return merge_surveys(parts)

    parts = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_survey_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(pool.map(survey_shard, names, chunksize=1), 1):
            parts.append(part)
            if index % progress_every == 0:
                logger.info("Surveyed %d of %d shard(s)", index, len(names))
    return merge_surveys(parts)
```

- [ ] **Step 8: Run the survey tests**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus_survey.py -v`
Expected: PASS

- [ ] **Step 9: Run the import-boundary test**

Run: `python -m pytest tests/unit/effects/test_import_boundaries.py -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/effects/application/build_corpus.py src/effects/application/train_effect_model.py tests/unit/effects/application/
git commit -m "feat(effects): survey the whole corpus by provenance key"
```

---

### Task 5: The decisions and the dataset directory

**Files:**
- Modify: `src/effects/application/build_corpus.py`
- Create: `src/effects/infrastructure/corpus_store.py`
- Test: `tests/unit/effects/application/test_build_corpus_decide.py`
- Test: `tests/unit/effects/infrastructure/test_corpus_store.py`

**Interfaces:**
- Consumes: `Survey`, `parse_ability_key` (Task 4); `CapHeap`, `class_targets` (Task 2); `CorpusManifest`, `SourceShard` (Task 1); `text_for_key` (Task 4).
- Produces:
  - `BuildCorpusConfig` — the command's flags as a frozen dataclass, fields exactly: `records_dir: Path`, `output: Path = Path("output/effects/corpus")`, `cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")`, `vocab_path: str = "models/effects/vocab.txt"`, `holdout_permille: int = 20`, `holdout_max_carriers: int = 8`, `text_cap: int = 200`, `class_mix: dict[str, float] | None = None`, `training_records: int = 0`, `game_disjoint_target: int = 1000`, `card_disjoint_text_cap: int = 50`, `seed: int = 42`, `workers: int | None = None`, `verify: bool = False`.
  - `Decisions` and `decide(survey, *, sidecars, surface, config) -> Decisions`.
  - In `corpus_store.py`: `CorpusStore(directory)` with `save`/`load`/`manifest_path`/`training_dir`/`card_disjoint_dir`/`game_disjoint_dir`, and `current_sources(records_dir) -> tuple[SourceShard, ...]`.

**Two rulings already made — read before implementing.**

1. **The cap is per text, not per key.** Two functional reprints compile to the same script and so to one ability text; a threshold computed per provenance key would give them a cap each. `decide` therefore merges the survey's per-key heaps into per-text heaps, takes one threshold per text, and writes that threshold back onto every key that folds to it.
2. **The card-disjoint cap admits games, not records** (FR-142). Validation is selected whole-game (FR-136), so capping records inside a game would break the very joins that rule protects. `decide` walks held-out games in a seeded order, admitting a game while any held-out text it carries is still under `--card-disjoint-text-cap`, and stops once every held-out text is at its cap.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/effects/application/test_build_corpus_decide.py
from __future__ import annotations

import pytest

from effects.application.build_corpus import BuildCorpusConfig, decide


def test_two_keys_folding_to_one_text_share_a_threshold(fake_sidecars):
    survey = a_survey(key_records={"reprint-a": 300, "reprint-b": 300},
                      key_hashes={"reprint-a": range(300), "reprint-b": range(300)})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=100))

    assert decisions.thresholds["reprint-a"] == decisions.thresholds["reprint-b"]


def test_rarity_counts_games_over_the_whole_corpus(fake_sidecars):
    survey = a_survey(key_games={"reprint-a": {1, 2}, "reprint-b": {2, 3}})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config())

    # Both fold to one text, and game 2 is one game: three distinct games.
    assert decisions.rarity["deals 3 damage"] == 3


def test_a_text_under_the_cap_gets_no_threshold(fake_sidecars):
    survey = a_survey(key_records={"reprint-a": 5})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=100))

    assert decisions.thresholds["reprint-a"] is None


def test_a_key_no_sidecar_can_read_gets_no_threshold_and_no_rarity(fake_sidecars):
    survey = a_survey(key_records={"unknown-key": 500})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=10))

    assert "unknown-key" not in decisions.thresholds


def test_games_split_into_the_two_strata_and_training(fake_sidecars):
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(10)}),
        held_out_games=frozenset({"g0", "g1"}),
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(game_disjoint_target=3))

    assert decisions.card_disjoint <= frozenset({"g0", "g1"})
    assert len(decisions.game_disjoint) == 3
    assert not (decisions.card_disjoint & decisions.game_disjoint)
    assert not (decisions.game_disjoint & frozenset({"g0", "g1"}))


def test_the_game_disjoint_draw_is_seeded(fake_sidecars):
    survey = a_survey(games=frozenset({f"g{i}" for i in range(50)}))
    first = decide(survey, sidecars=fake_sidecars, surface="script",
                   config=a_config(game_disjoint_target=5, seed=7))
    again = decide(survey, sidecars=fake_sidecars, surface="script",
                   config=a_config(game_disjoint_target=5, seed=7))

    assert first.game_disjoint == again.game_disjoint


def test_a_game_disjoint_target_larger_than_the_corpus_takes_what_there_is(fake_sidecars):
    survey = a_survey(games=frozenset({"g0", "g1"}))
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(game_disjoint_target=100))

    assert decisions.game_disjoint == frozenset({"g0", "g1"})


def test_the_card_disjoint_cap_stops_admitting_games_once_texts_are_covered(fake_sidecars):
    # Twenty games all carrying the one held-out text; a cap of 2 admits few.
    survey = a_survey(
        games=frozenset({f"g{i}" for i in range(20)}),
        held_out_games=frozenset({f"g{i}" for i in range(20)}),
        held_out_text_games={"held-out-text": {f"g{i}" for i in range(20)}},
    )
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(card_disjoint_text_cap=2))

    assert 0 < len(decisions.card_disjoint) < 20


def test_capped_class_records_never_exceed_what_the_corpus_holds(fake_sidecars):
    survey = a_survey(class_records={"rewrite": 100, "combat": 50})
    decisions = decide(survey, sidecars=fake_sidecars, surface="script",
                       config=a_config(text_cap=10))

    for name, held in survey.class_records.items():
        assert decisions.capped_class_records[name] <= held
```

Write `a_survey(**overrides)` returning a real `Survey` with sensible empty defaults, and `a_config(**overrides)` returning a `BuildCorpusConfig(records_dir=Path("."), **overrides)`. Write `fake_sidecars` as a fixture whose `line_for`/`prose_for` resolve `reprint-a` and `reprint-b` to the text `"deals 3 damage"`, raise `KeyError` for `unknown-key`, and resolve `held-out-text` to `"held-out-text"`. Read `tests/unit/effects/application/test_train_effect_model.py` first and reuse its sidecar fake rather than writing a second one.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus_decide.py -v`
Expected: FAIL with `ImportError: cannot import name 'decide'`

- [ ] **Step 3: Write `decide`**

Append to `src/effects/application/build_corpus.py`:

```python
@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus)."""

    records_dir: Path
    output: Path = Path("output/effects/corpus")
    cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")
    vocab_path: str = "models/effects/vocab.txt"
    holdout_permille: int = 20
    holdout_max_carriers: int = 8
    text_cap: int = 200
    class_mix: dict[str, float] | None = None
    training_records: int = 0
    game_disjoint_target: int = 1000
    card_disjoint_text_cap: int = 50
    seed: int = 42
    workers: int | None = None
    verify: bool = False

    def mix(self) -> dict[str, float]:
        from effects.application.train_effect_model import DEFAULT_KIND_MIX

        return dict(self.class_mix or DEFAULT_KIND_MIX)


@dataclass(frozen=True, slots=True)
class Decisions:
    """Everything the write pass needs, and nothing it has to look up.

    ``thresholds`` is keyed by rendered provenance key rather than by ability
    text, so a worker applies it without building a ``SidecarCache``; the fold
    from key to text happened here, once.
    """

    thresholds: dict[str, int | None]
    class_targets: dict[str, int]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    rarity: dict[str, int]
    shortfall: dict[str, int]
    capped_class_records: dict[str, int]
    key_text: dict[str, str]


def _text_of_rendered_key(rendered: str, sidecars, surface: str) -> str | None:
    """The text a survey key folds to, through the trainer's own definition."""
    from effects.application.train_effect_model import text_for_key

    for key in parse_ability_key(rendered):
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


def _card_disjoint_games(
    survey: Survey, key_text: dict[str, str], *, cap: int, seed: int,
) -> frozenset[str]:
    """Held-out games admitted while some held-out text is under its cap.

    Games rather than records (FR-142, FR-136): a stratum built by dropping
    records inside a game would separate a probe from the combat record its
    ``mirror_of`` names, and the evaluator would score that keyword on nothing
    while reporting it as merely under-sampled.

    Walked in a seeded order so two builds of one corpus admit the same games,
    and greedy so a text carried by few games is never crowded out by one
    carried by thousands.
    """
    import random

    text_games: dict[str, set[str]] = defaultdict(set)
    for key, games in survey.held_out_text_games.items():
        text = key_text.get(key, key)
        text_games[text] |= games

    order = sorted(survey.held_out_games)
    random.Random(seed).shuffle(order)
    games_text: dict[str, set[str]] = defaultdict(set)
    for text, games in text_games.items():
        for game in games:
            games_text[game].add(text)

    taken: Counter[str] = Counter()
    admitted: set[str] = set()
    for game in order:
        texts = games_text.get(game, set())
        if not texts:
            continue
        if any(taken[text] < cap for text in texts):
            admitted.add(game)
            for text in texts:
                taken[text] += 1
    return frozenset(admitted)


def decide(survey: Survey, *, sidecars, surface: str, config) -> Decisions:
    """Fold keys to texts, cap, split the games, and set the class targets."""
    import random

    from effects.domain.corpus_curation import class_targets

    key_text: dict[str, str] = {}
    text_games: dict[str, set[int]] = defaultdict(set)
    text_heaps: dict[str, CapHeap] = {}
    for key in survey.key_records:
        text = _text_of_rendered_key(key, sidecars, surface)
        if text is None:
            # No sidecar can read this key, so it is not a text and cannot be
            # capped against one. Its records stay, uncapped: dropping them
            # would remove a card the converted corpus never held rather than
            # trimming a head.
            continue
        key_text[key] = text
        text_games[text] |= survey.key_games.get(key, set())
        heap = text_heaps.get(text)
        if heap is None:
            heap = text_heaps[text] = CapHeap(config.text_cap)
        heap.merge(survey.key_heaps[key])

    rarity = {text: len(games) for text, games in text_games.items()}
    thresholds = {key: text_heaps[text].threshold() for key, text in key_text.items()}

    card_disjoint = _card_disjoint_games(
        survey, key_text, cap=config.card_disjoint_text_cap, seed=config.seed,
    )
    remaining = sorted(survey.games - survey.held_out_games)
    take = min(config.game_disjoint_target, len(remaining))
    game_disjoint = frozenset(random.Random(config.seed).sample(remaining, take))

    capped = dict(survey.class_capped_records)
    targets, shortfall = class_targets(
        capped, config.mix(), ceiling=config.training_records,
    )
    return Decisions(
        thresholds=thresholds,
        class_targets=targets,
        card_disjoint=card_disjoint,
        game_disjoint=game_disjoint,
        rarity=rarity,
        shortfall=shortfall,
        capped_class_records=capped,
        key_text=key_text,
    )
```

Note the split excludes **every** held-out game from training, not only the admitted ones: `remaining` subtracts `survey.held_out_games`, not `card_disjoint`. A held-out game the cap declined is dropped entirely rather than trained on — FR-088 forbids training on it whatever the validation stratum does with it.

- [ ] **Step 4: Run them to verify they pass**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus_decide.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for the store**

```python
# tests/unit/effects/infrastructure/test_corpus_store.py
from __future__ import annotations

import pytest

from effects.domain.corpus_manifest import SourceShard
from effects.infrastructure.corpus_store import CorpusStore, current_sources


def test_a_manifest_round_trips_through_the_store(tmp_path, a_manifest):
    store = CorpusStore(tmp_path)
    store.save(a_manifest)
    assert store.load() == a_manifest


def test_load_fails_loudly_when_no_manifest_was_written(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        CorpusStore(tmp_path).load()


def test_the_store_names_the_three_output_directories(tmp_path):
    store = CorpusStore(tmp_path)
    assert store.training_dir == tmp_path / "training"
    assert store.card_disjoint_dir == tmp_path / "validation" / "card-disjoint"
    assert store.game_disjoint_dir == tmp_path / "validation" / "game-disjoint"


def test_current_sources_lists_shards_relative_to_the_corpus_root(tmp_path):
    (tmp_path / "depleted").mkdir()
    (tmp_path / "depleted" / "run.0-a.jsonl.gz").write_bytes(b"x" * 7)
    assert current_sources(tmp_path) == (
        SourceShard(name="depleted/run.0-a.jsonl.gz", size=7),
    )
```

Reuse the `manifest()` helper from `tests/unit/effects/domain/test_corpus_manifest.py` for the `a_manifest` fixture — import it rather than writing a second one.

- [ ] **Step 6: Run them to verify they fail, then write the store**

Run: `python -m pytest tests/unit/effects/infrastructure/test_corpus_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'effects.infrastructure.corpus_store'`

```python
# src/effects/infrastructure/corpus_store.py
"""Where a curated dataset lives on disk (FR-135).

The directory layout is the contract: ``training/``,
``validation/card-disjoint/`` and ``validation/game-disjoint/`` hold ordinary
shards, so every existing reader loads them unchanged, and ``manifest.json``
holds the decisions that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.corpus_manifest import CorpusManifest, SourceShard

MANIFEST_NAME = "manifest.json"


class CorpusStore:
    """Reads and writes one curated dataset directory."""

    def __init__(self, directory) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    @property
    def training_dir(self) -> Path:
        return self.directory / "training"

    @property
    def card_disjoint_dir(self) -> Path:
        return self.directory / "validation" / "card-disjoint"

    @property
    def game_disjoint_dir(self) -> Path:
        return self.directory / "validation" / "game-disjoint"

    def save(self, manifest: CorpusManifest) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.manifest_path

    def load(self) -> CorpusManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path}: not a curated corpus (no manifest.json). "
                "Build one with `python -m effects build-corpus`."
            )
        return CorpusManifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )


def current_sources(records_dir) -> tuple[SourceShard, ...]:
    """Every raw shard under ``records_dir``, by relative path and size."""
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return tuple(
        SourceShard(name=shard.relative_to(root).as_posix(), size=shard.stat().st_size)
        for shard in iter_shards(root)
    )
```

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus_decide.py tests/unit/effects/infrastructure/test_corpus_store.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/effects tests/unit/effects
git commit -m "feat(effects): decide a curated corpus's split, caps and targets"
```

---

### Task 6: The write pass and the command

**Files:**
- Modify: `src/effects/application/build_corpus.py`
- Modify: `src/effects/infrastructure/cli.py` — add the `build-corpus` subparser next to `collect-coverage` (around line 301)
- Test: `tests/unit/effects/application/test_build_corpus.py`

**Interfaces:**
- Consumes: `Decisions`, `BuildCorpusConfig` (Task 5); `CorpusStore`, `current_sources` (Task 5); `write_shard` (Task 3); `record_hash`, `keeps` (Task 2); `text_keyed_holdout`, `load_card_files`, `load_card_texts` (existing, `train_effect_model`); `SidecarCache` (existing); `surface_of` (existing, `surface_batching`).
- Produces: `WriteConfig`, `init_write_worker`, `write_shard_pass`, `WriteResult`, `build(config: BuildCorpusConfig) -> int`, and `run_build_corpus(args)` in the CLI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/effects/application/test_build_corpus.py
from __future__ import annotations

import pytest

from effects.application.build_corpus import BuildCorpusConfig, build
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.record_io import read_records


def test_build_writes_the_three_strata_and_a_manifest(tmp_path, a_corpus):
    out = tmp_path / "curated"
    assert build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    )) == 0

    store = CorpusStore(out)
    assert store.manifest_path.exists()
    assert list(store.training_dir.glob("*.jsonl.gz"))
    assert store.load().digest()


def test_no_training_record_belongs_to_a_withheld_game(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    store = CorpusStore(out)
    manifest = store.load()
    withheld = set(manifest.card_disjoint_games) | set(manifest.game_disjoint_games)
    assert withheld
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert not trained & withheld


def test_no_training_record_names_a_held_out_card(tmp_path, a_corpus):
    """FR-088: the exclusion is by game, and every held-out game is dropped."""
    from effects.application.train_effect_model import (
        HeldOutCards, record_names_held_out_card,
    )

    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    held_out = HeldOutCards(
        names=frozenset(manifest.held_out_cards), script_files=frozenset(),
    )
    for record in read_records(CorpusStore(out).training_dir):
        assert not record_names_held_out_card(record, held_out)


def test_a_validation_stratum_keeps_whole_games(tmp_path, a_corpus):
    """A probe and the combat record it mirrors land together (FR-136)."""
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    store = CorpusStore(out)
    mirrored = 0
    for directory in (store.card_disjoint_dir, store.game_disjoint_dir):
        records = list(read_records(directory))
        by_id = {r.record_id for r in records}
        for record in records:
            if record.mirror_of is not None:
                mirrored += 1
                assert record.mirror_of in by_id
    assert mirrored, "the fixture must contain a mirrored probe or this proves nothing"


def test_the_cap_trims_a_text_that_exceeds_it(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, text_cap=2, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) > 0


def test_two_builds_of_one_corpus_agree(tmp_path, a_corpus):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        build(BuildCorpusConfig(
            records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
            workers=1, game_disjoint_target=1,
        ))

    assert CorpusStore(first).load().digest() == CorpusStore(second).load().digest()
    assert (
        [r.record_id for r in read_records(CorpusStore(first).training_dir)]
        == [r.record_id for r in read_records(CorpusStore(second).training_dir)]
    )


def test_verify_reports_drift_and_writes_nothing(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    before = (out / "manifest.json").read_text(encoding="utf-8")

    (a_corpus.records / "extra.0-zz.jsonl.gz").write_bytes(b"")
    code = build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, verify=True,
    ))

    assert code == 1
    assert (out / "manifest.json").read_text(encoding="utf-8") == before


def test_verify_is_quiet_when_nothing_moved(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    assert build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, verify=True,
    )) == 0


def test_build_refuses_an_empty_corpus(tmp_path, a_corpus):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no shards"):
        build(BuildCorpusConfig(
            records_dir=empty, cards_folders=a_corpus.cards,
            output=tmp_path / "out", workers=1,
        ))
```

**The `a_corpus` fixture is the whole test's load-bearing part.** Build it as a small but *real* corpus, returning an object with `.records` (the raw shard root) and `.cards` (the converted-tree folders):

- A converted card tree with sidecars, so `text_keyed_holdout` finds texts and `SidecarCache` resolves keys. Look for an existing fixture under `tests/` that builds one (search for `provenance.json` in the test tree) and reuse it; write one only if none exists.
- At least one card whose text the holdout rule selects, so `held_out_cards` is non-empty. Pick the card by computing `crc32` of its normalized text and choosing text that lands under the permille, rather than by trial and error.
- Two raw shards, several games each: one game naming the held-out card, one game holding a `combat` record and a probe record whose `mirror_of` names it, and one ability text repeated enough times to exceed a `text_cap` of 2.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus.py -v`
Expected: FAIL with `ImportError: cannot import name 'build'`

- [ ] **Step 3: Write the write pass and the command**

Append to `src/effects/application/build_corpus.py`:

```python
#: Training admission is two independent thresholds on the record id: the
#: per-text cap, and the class's share of the mixture. Both are hashes rather
#: than counters so a worker needs no coordination with any other worker, and
#: the second is keyed differently from the first so a record is not judged
#: twice by the same number.
_CLASS_SEED_OFFSET = 0x9E3779B9


@dataclass(frozen=True, slots=True)
class WriteConfig:
    records_dir: str
    training_dir: str
    card_disjoint_dir: str
    game_disjoint_dir: str
    thresholds: dict[str, int | None]
    class_admit: dict[str, float]
    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    held_out_games: frozenset[str]
    seed: int


@dataclass(slots=True)
class WriteResult:
    kept: Counter[str] = field(default_factory=Counter)
    dropped_by_cap: Counter[str] = field(default_factory=Counter)
    read: Counter[str] = field(default_factory=Counter)
    kept_keys: dict[str, set[str]] = field(default_factory=dict)


_WRITE: WriteConfig | None = None


def init_write_worker(config: WriteConfig) -> None:
    global _WRITE
    _WRITE = config


def _output_name(relative: str) -> str:
    """A flat shard name for a source that may sit in a subdirectory.

    ``depleted/run.0-a.jsonl.gz`` and ``full-strength/run.0-a.jsonl.gz`` are
    different shards and must not write to one file.
    """
    return relative.replace("/", "__")


def write_shard_pass(relative: str) -> WriteResult:
    """Filter one shard into the three outputs. Module-level so it pickles."""
    from effects.application.train_effect_model import sampling_class
    from effects.domain.corpus_curation import keeps, record_hash
    from effects.infrastructure.record_io import read_shard, write_shard

    config = _WRITE
    assert config is not None, "init_write_worker was not run"
    out = WriteResult()
    training: list = []
    card_disjoint: list = []
    game_disjoint: list = []

    for record in read_shard(Path(config.records_dir) / relative):
        name = sampling_class(record)
        out.read[name] += 1
        if record.game_id in config.card_disjoint:
            card_disjoint.append(record)
            continue
        if record.game_id in config.game_disjoint:
            game_disjoint.append(record)
            continue
        if record.game_id in config.held_out_games:
            # Held out but not admitted to the stratum: dropped, never trained
            # on (FR-088).
            continue
        key = ability_key(record)
        value = record_hash(record.record_id, seed=config.seed)
        if key is not None and not keeps(value, config.thresholds.get(key)):
            out.dropped_by_cap[name] += 1
            continue
        share = config.class_admit.get(name, 0.0)
        if share < 1.0:
            draw = record_hash(record.record_id, seed=config.seed + _CLASS_SEED_OFFSET)
            if draw >= int(share * 2**64):
                continue
        training.append(record)
        out.kept[name] += 1
        if key is not None:
            out.kept_keys.setdefault(name, set()).add(key)

    shard_name = _output_name(relative)
    write_shard(Path(config.training_dir) / shard_name, training)
    write_shard(Path(config.card_disjoint_dir) / shard_name, card_disjoint)
    write_shard(Path(config.game_disjoint_dir) / shard_name, game_disjoint)
    return out


def run_write_pass(
    names, *, config: WriteConfig, workers: int | None, progress_every: int = 50,
) -> WriteResult:
    """Write every shard's three outputs, in parallel, reporting progress."""
    total = WriteResult()

    def absorb(part: WriteResult) -> None:
        total.kept.update(part.kept)
        total.dropped_by_cap.update(part.dropped_by_cap)
        total.read.update(part.read)
        for name, keys in part.kept_keys.items():
            total.kept_keys.setdefault(name, set()).update(keys)

    if workers is not None and workers <= 1:
        init_write_worker(config)
        for index, name in enumerate(names, start=1):
            absorb(write_shard_pass(name))
            if index % progress_every == 0:
                logger.info("Wrote %d of %d shard(s)", index, len(names))
        return total

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_write_worker, initargs=(config,),
    ) as pool:
        for index, part in enumerate(
            pool.map(write_shard_pass, names, chunksize=1), start=1,
        ):
            absorb(part)
            if index % progress_every == 0:
                logger.info("Wrote %d of %d shard(s)", index, len(names))
    return total


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.application.surface_batching import surface_of
    from effects.application.train_effect_model import (
        load_card_files, load_card_texts, text_keyed_holdout,
    )
    from effects.domain.corpus_manifest import ClassCounts, CorpusManifest
    from effects.infrastructure.corpus_store import CorpusStore, current_sources
    from effects.infrastructure.sidecar_io import SidecarCache

    store = CorpusStore(config.output)
    records_dir = Path(config.records_dir)

    if config.verify:
        manifest = store.load()
        added, removed, resized = manifest.drift(current_sources(records_dir))
        for label, names in (("added", added), ("removed", removed), ("resized", resized)):
            if names:
                logger.warning(
                    "%d shard(s) %s since this dataset was built: %s%s",
                    len(names), label, ", ".join(names[:5]),
                    ", …" if len(names) > 5 else "",
                )
        if added or removed or resized:
            logger.warning("Rebuild with `python -m effects build-corpus`.")
            return 1
        logger.info("Dataset is current against %s.", records_dir)
        return 0

    folders = {Path(f).name: Path(f) for f in config.cards_folders}
    cards_folder = folders.get("cardsfolder", Path(config.cards_folders[0]))
    sidecars = SidecarCache(folders)
    card_files = load_card_files(cards_folder)
    held_out = text_keyed_holdout(
        card_files,
        load_card_texts(card_files, sidecars),
        permille=config.holdout_permille,
        max_carriers=config.holdout_max_carriers,
    )
    surface = surface_of(config.vocab_path)
    logger.info(
        "Holdout: %d ability text(s) on %d card(s); encoding surface %r.",
        len(held_out.texts), len(held_out.names), surface,
    )

    survey = run_survey(
        records_dir,
        config=SurveyConfig(
            records_dir=str(records_dir),
            held_out_names=held_out.names,
            held_out_script_files=held_out.script_files,
            text_cap=config.text_cap,
            seed=config.seed,
        ),
        workers=config.workers,
    )
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card.",
        survey.records, len(survey.games), len(survey.held_out_games),
    )

    decisions = decide(survey, sidecars=sidecars, surface=surface, config=config)
    admit = {
        name: min(1.0, target / decisions.capped_class_records[name])
        for name, target in decisions.class_targets.items()
        if decisions.capped_class_records.get(name)
    }
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(
            records_dir=str(records_dir),
            training_dir=str(store.training_dir),
            card_disjoint_dir=str(store.card_disjoint_dir),
            game_disjoint_dir=str(store.game_disjoint_dir),
            thresholds=decisions.thresholds,
            class_admit=admit,
            card_disjoint=decisions.card_disjoint,
            game_disjoint=decisions.game_disjoint,
            held_out_games=survey.held_out_games,
            seed=config.seed,
        ),
        workers=config.workers,
    )

    per_class = {
        name: ClassCounts(
            read=written.read.get(name, 0),
            kept=written.kept.get(name, 0),
            dropped_by_cap=written.dropped_by_cap.get(name, 0),
            unique_texts=len({
                decisions.key_text.get(key, key)
                for key in written.kept_keys.get(name, ())
            }),
        )
        for name in sorted(written.read)
    }
    manifest = CorpusManifest(
        seed=config.seed,
        surface=surface,
        vocab_path=config.vocab_path,
        holdout_permille=config.holdout_permille,
        holdout_max_carriers=config.holdout_max_carriers,
        text_cap=config.text_cap,
        card_disjoint_text_cap=config.card_disjoint_text_cap,
        game_disjoint_target=config.game_disjoint_target,
        training_records=config.training_records,
        class_mix=config.mix(),
        held_out_cards=tuple(sorted(held_out.names)),
        card_disjoint_games=tuple(sorted(decisions.card_disjoint)),
        game_disjoint_games=tuple(sorted(decisions.game_disjoint)),
        rarity=decisions.rarity,
        sources=survey.shards,
        per_class=per_class,
        shortfall=decisions.shortfall,
    )
    store.save(manifest)

    for name, counts in per_class.items():
        logger.info(
            "%-22s read %9d  kept %9d  cap dropped %8d  unique texts %7d",
            name, counts.read, counts.kept, counts.dropped_by_cap, counts.unique_texts,
        )
    for name, missing in sorted(decisions.shortfall.items()):
        logger.warning(
            "%s is short %d record(s) of its share: --training-records asks for "
            "more than the corpus can supply at this mixture.", name, missing,
        )
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.kept.values()), store.directory, manifest.digest(),
    )
    return 0
```

Then add the subparser to `src/effects/infrastructure/cli.py`, following the shape of the `collect-coverage` parser above it, with every flag from the root spec's table and a `run_build_corpus(args)` that builds a `BuildCorpusConfig` and returns `build(config)`. `--class-mix` parses through the existing `parse_kind_mix`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole effects suite and the boundary test**

Run: `python -m pytest tests/unit/effects tests/integration -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/effects tests/unit/effects
git commit -m "feat(effects): write the curated corpus and wire up build-corpus"
```

---

### Task 7: Training against a curated corpus

**Files:**
- Modify: `src/effects/application/train_effect_model.py` — `TrainEffectModelConfig` (around line 841), `resolve_holdout` (around line 1060), and the corpus-resolution block (around line 1140)
- Modify: `src/effects/application/training_loop.py` — the `sample_weights` call (around line 299)
- Modify: `src/effects/infrastructure/cli.py` — the `train-effect-model` parser (around line 941)
- Test: `tests/unit/effects/application/test_train_effect_model.py`

**Interfaces:**
- Consumes: `CorpusStore`, `CorpusManifest` (Tasks 1 and 5).
- Produces: `TrainEffectModelConfig.corpus: str | None = None`; `CORPUS_EXCLUSIVE_FLAGS`; `validate_corpus_flags(config) -> None`; `sample_weights(records, text_of, *, rarity: Mapping[str, int] | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("field,value", [
    ("records_dir", "output/effects/records"),
    ("split_from", "models/effects/effect-model/latest.pt"),
    ("reserved_shards", 9),
    ("holdout_permille", 30),
    ("holdout_max_carriers", 4),
])
def test_corpus_refuses_the_flags_the_manifest_already_decides(field, value):
    from effects.application.train_effect_model import (
        TrainEffectModelConfig, validate_corpus_flags,
    )

    config = TrainEffectModelConfig(
        corpus="output/effects/corpus", **{field: value},
    )
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        validate_corpus_flags(config)


def test_corpus_alone_is_accepted():
    from effects.application.train_effect_model import (
        TrainEffectModelConfig, validate_corpus_flags,
    )

    validate_corpus_flags(TrainEffectModelConfig(corpus="output/effects/corpus"))


def test_sample_weights_prefer_a_supplied_corpus_wide_rarity_table():
    from effects.application.train_effect_model import sample_weights

    records = [a_record(record_id="r1", game_id="g1")]
    # The resident shard shows one game; corpus-wide the text was in a hundred.
    corpus_wide = sample_weights(records, lambda r: "t", rarity={"t": 100})
    shard_only = sample_weights(records, lambda r: "t")

    assert corpus_wide[0] < shard_only[0]


def test_a_text_the_table_does_not_name_falls_back_to_the_shard():
    """A shard collected after the dataset was built still weights sanely."""
    from effects.application.train_effect_model import sample_weights

    records = [a_record(record_id="r1", game_id="g1")]
    assert sample_weights(records, lambda r: "new", rarity={"t": 100}) == (
        sample_weights(records, lambda r: "new")
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_train_effect_model.py -k "corpus or rarity or table" -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'corpus'`

- [ ] **Step 3: Implement**

Add `corpus: str | None = None` to `TrainEffectModelConfig`, and:

```python
#: Flags a curated manifest already records. Passing one beside `--corpus` is
#: two spellings of one decision, and a disagreement nothing would report.
CORPUS_EXCLUSIVE_FLAGS: tuple[str, ...] = (
    "records_dir", "reserved_shards", "split_from",
    "holdout_permille", "holdout_max_carriers",
)


def validate_corpus_flags(config: TrainEffectModelConfig) -> None:
    """Refuse a flag the curated manifest decides (FR-146)."""
    if config.corpus is None:
        return
    defaults = TrainEffectModelConfig(corpus=config.corpus)
    for name in CORPUS_EXCLUSIVE_FLAGS:
        if getattr(config, name) != getattr(defaults, name):
            spelled = "--" + name.replace("_", "-")
            raise ValueError(
                f"{spelled} cannot be passed with --corpus: the dataset's "
                f"manifest at {config.corpus} already records it."
            )
```

Extend `sample_weights` with `rarity: Mapping[str, int] | None = None`. When given, build the weights from `{text: rarity.get(text, shard_count)}` — falling back per text to the resident-shard count, so a shard collected after the build still weights sanely rather than at zero.

In `run` (around line 1140): call `validate_corpus_flags` first; when `config.corpus` is set, load the manifest through `CorpusStore`, take `training_shards` from `iter_shards(store.training_dir)` and `validation_shards` from the two stratum directories, take `held_out`/`inherited` from the manifest instead of `resolve_holdout`, and pass `manifest.rarity` into `TrainingLoop`. Thread it to the `sample_weights` call in `training_loop.py`. Do not call `reserve_shards` on this path.

Add `--corpus` to the CLI parser with `default=None`.

- [ ] **Step 4: Run the suite**

Run: `python -m pytest tests/unit/effects -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/effects tests/unit/effects
git commit -m "feat(effects): train against a curated corpus and its rarity table"
```

---

### Task 8: Corpus provenance in the checkpoint

**Files:**
- Modify: `src/effects/infrastructure/effect_model_store.py` — `SplitProvenance` (around line 82)
- Modify: `src/effects/application/train_effect_model.py` — where the checkpoint's provenance is assembled
- Modify: `src/effects/application/evaluate_effect_model.py` — beside the vocabulary-hash guard
- Modify: `src/effects/CLAUDE.md`
- Test: `tests/unit/effects/infrastructure/test_effect_model_store.py`
- Test: `tests/unit/effects/application/test_evaluate_effect_model.py`

**Interfaces:**
- Consumes: `CorpusStore` (Task 5).
- Produces: `SplitProvenance.corpus_path: str = ""`, `SplitProvenance.corpus_digest: str = ""`; `CorpusMismatchError`; `check_corpus(provenance, *, actual_digest)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_provenance_round_trips_the_corpus_it_read():
    from effects.infrastructure.effect_model_store import SplitProvenance

    original = SplitProvenance(
        corpus_path="output/effects/corpus", corpus_digest="abc123",
    )
    assert SplitProvenance.from_dict(original.as_dict()) == original


def test_a_checkpoint_saved_before_curated_corpora_still_loads():
    from effects.infrastructure.effect_model_store import SplitProvenance

    assert SplitProvenance.from_dict({}).corpus_digest == ""


def test_evaluate_refuses_a_corpus_rebuilt_since_training(tmp_path):
    from effects.application.evaluate_effect_model import (
        CorpusMismatchError, check_corpus,
    )
    from effects.infrastructure.effect_model_store import SplitProvenance

    provenance = SplitProvenance(
        corpus_path=str(tmp_path), corpus_digest="trained-against-this",
    )
    with pytest.raises(CorpusMismatchError, match="rebuilt"):
        check_corpus(provenance, actual_digest="something-else")


def test_evaluate_accepts_a_checkpoint_that_read_no_curated_corpus():
    from effects.application.evaluate_effect_model import check_corpus
    from effects.infrastructure.effect_model_store import SplitProvenance

    check_corpus(SplitProvenance(), actual_digest="anything")   # no raise
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/effects -k "corpus_digest or check_corpus or rebuilt or no_curated or before_curated" -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'corpus_path'`

- [ ] **Step 3: Implement**

Add the two fields to `SplitProvenance` with `""` defaults, read in `from_dict` with `data.get(name, "")` so a checkpoint saved before this change still loads. Populate them in the trainer when `--corpus` is set. Add to `evaluate_effect_model.py`:

```python
class CorpusMismatchError(RuntimeError):
    """The curated corpus was rebuilt since the checkpoint trained on it."""


def check_corpus(provenance, *, actual_digest: str) -> None:
    """Refuse a dataset that is not the one the checkpoint read (FR-147).

    A rebuild is a different split, so scoring the gates against it would score
    them partly on games the model trained on — the failure the recorded
    ``game_id`` sets exist to prevent, arriving through the corpus instead.
    """
    if not provenance.corpus_digest:
        return
    if provenance.corpus_digest != actual_digest:
        raise CorpusMismatchError(
            f"{provenance.corpus_path} has been rebuilt since this checkpoint "
            f"trained on it (recorded {provenance.corpus_digest}, found "
            f"{actual_digest}). Evaluate against the dataset it read, or "
            "retrain against this one."
        )
```

Call it from `evaluate-effect-model` beside the existing vocabulary-hash check, and add `--corpus` to that parser, defaulting to the checkpoint's recorded path.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/unit/effects tests/integration -v`
Expected: PASS

- [ ] **Step 5: Update the package guide**

Add `build-corpus` to the subcommand list in `src/effects/CLAUDE.md`, between `train-effect-model` and `encode-abilities`, in that file's register — it describes what the code does now, not what changed. Note in "Performance invariants" that a `--corpus` run reads the rarity table from the manifest rather than counting the resident shard, which is the one place the count is right for coverage and variant shards.

- [ ] **Step 6: Commit**

```bash
git add src/effects tests
git commit -m "feat(effects): pin a checkpoint to the curated corpus it read"
```
