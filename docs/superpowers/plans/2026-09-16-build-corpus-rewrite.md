# Build-corpus Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m effects build-corpus` produce a corpus the trainer can consume without deriving anything: quality-filtered records, a gate-1 slice, fixed validation samples, uniform shards, and a per-source held-out report.

**Architecture:** The build keeps its two-pass shape (survey, then write). A pure domain module decides record quality; both passes skip defective records. The write pass writes per-source parts into a scratch directory and a new repack step streams them into fixed-size shards cut at game boundaries. A third, serial pass draws the validation samples from the written strata by smallest seeded record hash. Everything new is recorded in the manifest.

**Tech Stack:** Python 3.12, dataclasses, `ProcessPoolExecutor`, gzip JSONL shards, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-corpus-and-trainer-rework.md` (FR-135, FR-148, FR-149) over `specs/023-ability-effect-model/spec.md`.

## Global Constraints

- Every output shard directory must stay readable by `effects.infrastructure.record_io.iter_shards` / `read_shard` unchanged (FR-135).
- Two builds of one corpus at one seed must produce identical manifests (`test_two_builds_of_one_corpus_agree` must keep passing).
- Workers never hold more than one source shard; the repack and sample passes stream.
- Manifest readers must load a manifest written before this plan: every new field has a `.get` default in `CorpusManifest.from_dict`.
- Run tests from the repository root with `python -m pytest`.
- Before editing any file under `specs/` or `experiments/` (Tasks 6 and 7), load the `feature-workflow` skill: those files carry binding wording and structure rules.

---

## File map

| File | Responsibility |
|---|---|
| Create `src/effects/domain/record_quality.py` | `quality_defect(record)`: the three quality rules, pure |
| Modify `src/effects/domain/corpus_manifest.py` | new manifest fields with defaults |
| Modify `src/effects/infrastructure/corpus_store.py` | `gate_one_dir`, `samples_dir`, `sample_path`, `parts_dir`; clear them |
| Modify `src/effects/infrastructure/record_io.py` | `repack_shards(parts, out_dir, *, shard_records)` |
| Create `src/effects/application/validation_samples.py` | the serial sample pass: `draw_samples(...)` |
| Modify `src/effects/application/build_corpus.py` | thread quality, gate-one routing, parts + repack, per-source report, samples, manifest |
| Modify `src/effects/infrastructure/cli.py` | `--shard-records`, `--max-events-per-record`, `--validation-sample` |
| Modify `specs/023-ability-effect-model/quickstart.md`, `spec.md` | document outputs and flags |
| Tests | `tests/unit/effects/domain/test_record_quality.py`, `tests/unit/effects/infrastructure/test_repack_shards.py`, `tests/unit/effects/application/test_validation_samples.py`, additions to `tests/unit/effects/application/test_build_corpus.py` |

---

### Task 1: Record quality rules

**Files:**
- Create: `src/effects/domain/record_quality.py`
- Test: `tests/unit/effects/domain/test_record_quality.py`

**Interfaces:**
- Consumes: `effects.domain.effect_targets.events_of(record)`, `effects.domain.records.RecordKind`, `Event.attributed_to`.
- Produces: `quality_defect(record, *, max_events=MAX_EVENTS_PER_RECORD) -> str | None`; constants `NO_ABILITY = "no-ability"`, `UNATTRIBUTED = "unattributed-events"`, `EVENT_FLOOD = "event-flood"`, `QUALITY_REASONS`, `MAX_EVENTS_PER_RECORD = 64`, `UNRESOLVED = "unresolved"`.

- [ ] **Step 1: Write the failing tests**

```python
"""The three record-quality rules build-corpus applies (FR-148)."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.record_quality import (
    EVENT_FLOOD,
    NO_ABILITY,
    UNATTRIBUTED,
    quality_defect,
)
from effects.domain.records import CombatPayload, RecordKind, ResolutionPayload


def _event(attributed_to=None):
    return Event(type=EventType.DAMAGE_DEALT, subjects=("P0",), attributed_to=attributed_to)


def test_a_clean_resolution_record_has_no_defect(make_record):
    assert quality_defect(make_record()) is None


def test_a_resolution_record_with_no_ability_is_a_defect(make_record):
    assert quality_defect(make_record(ability=())) == NO_ABILITY
    assert quality_defect(make_record(ability=None)) == NO_ABILITY


def test_a_combat_record_needs_no_ability(make_record):
    record = make_record(kind=RecordKind.COMBAT, payload=CombatPayload(events=(_event(),)))
    assert quality_defect(record) is None


def test_an_unresolved_attribution_is_a_defect(make_record):
    payload = ResolutionPayload(events=(_event("root"), _event("unresolved")))
    assert quality_defect(make_record(payload=payload)) == UNATTRIBUTED


def test_an_unknown_attribution_is_not_a_defect(make_record):
    payload = ResolutionPayload(events=(_event(None),))
    assert quality_defect(make_record(payload=payload)) is None


def test_too_many_events_is_a_defect(make_record):
    payload = ResolutionPayload(events=tuple(_event("root") for _ in range(65)))
    assert quality_defect(make_record(payload=payload)) == EVENT_FLOOD
    assert quality_defect(make_record(payload=payload), max_events=100) is None


def test_no_ability_wins_over_the_other_reasons(make_record):
    payload = ResolutionPayload(events=tuple(_event("unresolved") for _ in range(65)))
    assert quality_defect(make_record(ability=(), payload=payload)) == NO_ABILITY
```

The `make_record` fixture lives in `tests/unit/effects/domain/conftest.py` and already accepts `ability=` and `payload=` overrides.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/domain/test_record_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'effects.domain.record_quality'`

- [ ] **Step 3: Write the module**

```python
"""Which records a curated corpus refuses on sight (FR-148).

Three shapes the first corpus held that no model should learn from. A
resolution record with no acting ability is an outcome with no cause. An
event attributed to ``unresolved`` was produced by a clause the collector
could not find on the acting chain, which in practice marks whole-game event
streams that landed on one record. And a record carrying more events than any
single ability resolves is the same dump seen from the other side: one such
record in the first corpus held 278 events, 88 card draws and the game's
``player_won``.

Pure over the record, so build-corpus applies it in a worker without a
sidecar, and a test can state each rule in one line.
"""

from __future__ import annotations

from effects.domain.effect_targets import events_of
from effects.domain.records import EffectRecord, RecordKind

NO_ABILITY = "no-ability"
UNATTRIBUTED = "unattributed-events"
EVENT_FLOOD = "event-flood"
QUALITY_REASONS: tuple[str, ...] = (NO_ABILITY, UNATTRIBUTED, EVENT_FLOOD)

#: The ``attributed_to`` sentinel the collector writes when the producing
#: clause was sought and nothing on the chain claimed the event.
UNRESOLVED = "unresolved"

#: More events than this on one record is a game, not an ability. The
#: heaviest legitimate resolutions in the first corpus (board wipes over
#: two full boards) sat under forty.
MAX_EVENTS_PER_RECORD = 64


def quality_defect(
    record: EffectRecord, *, max_events: int = MAX_EVENTS_PER_RECORD,
) -> str | None:
    """The first reason this record is refused, or None when it is clean.

    Ordered so the most specific reason wins: a record with no ability is
    reported as that, whatever its events look like.
    """
    if record.kind is RecordKind.RESOLUTION and not record.ability:
        return NO_ABILITY
    events = events_of(record)
    if any(event.attributed_to == UNRESOLVED for event in events):
        return UNATTRIBUTED
    if len(events) > max_events:
        return EVENT_FLOOD
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/domain/test_record_quality.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/record_quality.py tests/unit/effects/domain/test_record_quality.py
git commit -m "feat(effects): record-quality rules for build-corpus (FR-148)"
```

---

### Task 2: Manifest and store gain the new outputs

**Files:**
- Modify: `src/effects/domain/corpus_manifest.py`
- Modify: `src/effects/infrastructure/corpus_store.py`
- Test: `tests/unit/effects/domain/test_corpus_manifest.py` (append), `tests/unit/effects/application/test_build_corpus.py` (append)

**Interfaces:**
- Produces on `CorpusManifest`: `quality_dropped: dict[str, int]`, `games_by_source: dict[str, int]`, `held_out_games_by_source: dict[str, int]`, `shard_records: int`, `validation_sample: int`, `max_events_per_record: int`, all with defaults so old manifests load.
- Produces on `CorpusStore`: `gate_one_dir` (`validation/gate-one`), `samples_dir` (`validation/samples`), `sample_path(stratum) -> Path` (`validation/samples/<stratum>.jsonl.gz`), `parts_dir` (`.parts`), `parts_dir_for(stratum)` (`.parts/<stratum>`). `clear_outputs()` removes all of them.
- `STRATA` in `build_corpus.py` becomes `("training", "card-disjoint", "game-disjoint", "gate-one", "dropped-held-out")` and `OUTPUTS = STRATA[:-1]` (Task 4 makes that change; the manifest just stores whatever `per_stratum` it is given).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/effects/domain/test_corpus_manifest.py`:

```python
def test_a_manifest_written_before_the_rework_still_loads(manifest_dict):
    """Every field this rework added has a default (FR-143 compatibility)."""
    for key in (
        "quality_dropped", "games_by_source", "held_out_games_by_source",
        "shard_records", "validation_sample", "max_events_per_record",
    ):
        manifest_dict.pop(key, None)
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.quality_dropped == {}
    assert loaded.games_by_source == {}
    assert loaded.held_out_games_by_source == {}
    assert loaded.shard_records == 0
    assert loaded.validation_sample == 0
    assert loaded.max_events_per_record == 0


def test_the_new_fields_round_trip(manifest_dict):
    manifest_dict.update({
        "quality_dropped": {"no-ability": 3},
        "games_by_source": {"depleted": 10, "full-strength": 4},
        "held_out_games_by_source": {"depleted": 1, "full-strength": 4},
        "shard_records": 2000,
        "validation_sample": 2048,
        "max_events_per_record": 64,
    })
    loaded = CorpusManifest.from_dict(manifest_dict)
    assert loaded.as_dict()["quality_dropped"] == {"no-ability": 3}
    assert CorpusManifest.from_dict(loaded.as_dict()) == loaded
```

If `test_corpus_manifest.py` has no `manifest_dict` fixture, add one at the top of the file that builds a minimal manifest through the constructor and returns `.as_dict()`; every existing constructor call in that file shows the required fields.

Append to `tests/unit/effects/application/test_build_corpus.py`:

```python
def test_the_store_names_the_new_outputs(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    assert store.gate_one_dir == tmp_path / "corpus" / "validation" / "gate-one"
    assert store.samples_dir == tmp_path / "corpus" / "validation" / "samples"
    assert store.sample_path("card-disjoint") == (
        tmp_path / "corpus" / "validation" / "samples" / "card-disjoint.jsonl.gz"
    )
    assert store.parts_dir_for("training") == tmp_path / "corpus" / ".parts" / "training"


def test_clear_outputs_removes_the_new_outputs_too(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    for directory in (store.gate_one_dir, store.samples_dir, store.parts_dir_for("training")):
        directory.mkdir(parents=True)
        (directory / "x").write_text("x")
    store.clear_outputs()
    assert not store.gate_one_dir.exists()
    assert not store.samples_dir.exists()
    assert not store.parts_dir.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_manifest.py tests/unit/effects/application/test_build_corpus.py -k "rework or round_trip or new_outputs" -v`
Expected: FAIL with `TypeError` (unexpected keyword) and `AttributeError: 'CorpusStore' object has no attribute 'gate_one_dir'`

- [ ] **Step 3: Add the manifest fields**

In `CorpusManifest`, after `shortfall: dict[str, int]`, add:

```python
    #: Records refused by ``effects.domain.record_quality`` in both passes,
    #: by reason (FR-148). Empty on a manifest written before the rule existed.
    quality_dropped: dict[str, int] = field(default_factory=dict)
    #: Games per top-level source directory under ``--records-dir``
    #: ("depleted", "full-strength", …; "." for shards at the root), and how
    #: many of them name a held-out card. A depleted directory whose held-out
    #: count is not zero is a leak in collection, and this is where it shows
    #: (FR-149).
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: ``--shard-records``: the size output shards were repacked to. 0 on a
    #: manifest from before repacking, whose shards mirror their sources.
    shard_records: int = 0
    #: ``--validation-sample``: records per stratum in ``validation/samples/``.
    validation_sample: int = 0
    #: ``--max-events-per-record`` the quality rule used.
    max_events_per_record: int = 0
```

Add `field` to the dataclasses import. In `from_dict`, add:

```python
            quality_dropped={k: int(v) for k, v in data.get("quality_dropped", {}).items()},
            games_by_source={k: int(v) for k, v in data.get("games_by_source", {}).items()},
            held_out_games_by_source={
                k: int(v) for k, v in data.get("held_out_games_by_source", {}).items()
            },
            shard_records=int(data.get("shard_records", 0)),
            validation_sample=int(data.get("validation_sample", 0)),
            max_events_per_record=int(data.get("max_events_per_record", 0)),
```

- [ ] **Step 4: Add the store paths**

In `CorpusStore`:

```python
    @property
    def gate_one_dir(self) -> Path:
        """Resolution records of card-disjoint games whose acting text is held out."""
        return self.directory / "validation" / "gate-one"

    @property
    def samples_dir(self) -> Path:
        return self.directory / "validation" / "samples"

    def sample_path(self, stratum: str) -> Path:
        """The fixed validation sample the trainer reads for ``stratum``."""
        return self.samples_dir / f"{stratum}.jsonl.gz"

    @property
    def parts_dir(self) -> Path:
        """Per-source parts the write pass leaves for the repack step.

        Outside every stratum directory, because ``iter_shards`` is recursive
        and a part left under ``training/`` would be read as a shard.
        """
        return self.directory / ".parts"

    def parts_dir_for(self, stratum: str) -> Path:
        return self.parts_dir / stratum
```

Extend `clear_outputs` to remove `self.gate_one_dir`, `self.samples_dir` and `self.parts_dir` alongside the three existing directories.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/domain/test_corpus_manifest.py tests/unit/effects/application/test_build_corpus.py -v`
Expected: all pass, including the pre-existing manifest and build tests.

- [ ] **Step 6: Commit**

```bash
git add src/effects/domain/corpus_manifest.py src/effects/infrastructure/corpus_store.py tests/unit/effects/domain/test_corpus_manifest.py tests/unit/effects/application/test_build_corpus.py
git commit -m "feat(effects): manifest and store fields for the corpus rework"
```

---

### Task 3: Repack parts into fixed-size shards

**Files:**
- Modify: `src/effects/infrastructure/record_io.py` (append after `write_shard`)
- Test: `tests/unit/effects/infrastructure/test_repack_shards.py` (create; create `tests/unit/effects/infrastructure/__init__.py` if the directory has none)

**Interfaces:**
- Produces: `repack_shards(parts: Sequence[Path], out_dir: Path, *, shard_records: int, prefix: str = "shard") -> list[Path]`. Reads `parts` in the given order, writes `out_dir/<prefix>-00001.jsonl.gz` and so on, starts a new shard once the current one holds at least `shard_records` records **and** the next record's `game_id` differs from the last written one. `shard_records <= 0` means one shard per part (the old behaviour). Returns the written paths. Does not delete parts.

- [ ] **Step 1: Write the failing tests**

```python
"""Repacking per-source parts into uniform shards cut at game boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.record_io import read_records, read_shard, repack_shards, write_shard


@pytest.fixture
def parts(tmp_path, make_record) -> list[Path]:
    """Three parts: 3, 7 and 2 records; games of 2, 2, 3, 2, 2, 1, 1 records."""
    def game(gid, n):
        return [make_record(record_id=f"{gid}.{i}", game_id=gid) for i in range(n)]
    a = game("g1", 2) + game("g2", 1)
    b = game("g2", 1) + game("g3", 3) + game("g4", 2) + game("g5", 1)
    c = game("g6", 1) + game("g7", 1)
    out = []
    for name, records in (("a", a), ("b", b), ("c", c)):
        path = tmp_path / "parts" / f"{name}.jsonl.gz"
        write_shard(path, records)
        out.append(path)
    return out


def test_every_record_survives_in_order(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=4)
    before = [r.record_id for p in parts for r in read_shard(p)]
    after = [r.record_id for r in read_records(tmp_path / "out")]
    assert after == before
    assert [p.name for p in written] == sorted(p.name for p in written)


def test_a_shard_closes_at_a_game_boundary_once_full(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=4)
    games = [[r.game_id for r in read_shard(p)] for p in written]
    # 4 records fill the first shard mid-g3; it closes only when g3 ends.
    assert games[0] == ["g1", "g1", "g2", "g2", "g3", "g3", "g3"]
    for shard in games:
        for a, b in zip(shard, shard[1:]):
            assert a == b or shard.index(b) > shard.index(a)
    # No game spans two shards.
    seen = {}
    for index, shard in enumerate(games):
        for gid in shard:
            assert seen.setdefault(gid, index) == index


def test_zero_means_one_shard_per_part(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=0)
    assert len(written) == 3
    assert [sum(1 for _ in read_shard(p)) for p in written] == [3, 7, 2]


def test_names_are_zero_padded_and_sequential(tmp_path, parts):
    written = repack_shards(parts, tmp_path / "out", shard_records=2)
    assert written[0].name == "shard-00001.jsonl.gz"
    assert written[1].name == "shard-00002.jsonl.gz"


def test_no_parts_writes_nothing(tmp_path):
    assert repack_shards([], tmp_path / "out", shard_records=4) == []
    assert not (tmp_path / "out").exists()
```

`make_record` must accept `record_id=` and `game_id=` overrides; the domain fixture does (`defaults.update(overrides)`). If the infrastructure test directory has no `conftest.py`, add one that re-exports the domain fixtures:

```python
from tests.unit.effects.domain.conftest import ability_key, make_record, snapshot  # noqa: F401
```

(Check `tests/unit/effects/domain/conftest.py` is importable as a module path from the repo root; if `tests` is not a package, copy the three fixtures instead.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/infrastructure/test_repack_shards.py -v`
Expected: FAIL with `ImportError: cannot import name 'repack_shards'`

- [ ] **Step 3: Implement**

Append to `record_io.py` after `write_shard`:

```python
def repack_shards(
    parts: Sequence[Path], out_dir: Path, *, shard_records: int,
    prefix: str = "shard",
) -> list[Path]:
    """Stream ``parts`` into shards of about ``shard_records`` records each.

    A shard closes once it holds at least ``shard_records`` records *and* the
    next record belongs to a different game, so a game never spans two shards:
    the evaluator resolves a probe's ``mirror_of`` within its stratum, and the
    trainer's per-game text sharing wants a game's records together.

    ``shard_records <= 0`` writes one shard per part, which is what the write
    pass did before repacking existed. Parts are not deleted here.
    """
    import gzip

    parts = [Path(p) for p in parts]
    if not parts:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def open_next():
        path = out_dir / f"{prefix}-{len(written) + 1:05d}.jsonl.gz"
        written.append(path)
        return gzip.open(path, "wt", encoding="utf-8")

    if shard_records <= 0:
        for part in parts:
            with open_next() as handle:
                for record in read_shard(part):
                    handle.write(format_record_line(record))
                    handle.write("\n")
        return written

    handle = open_next()
    count = 0
    last_game: str | None = None
    try:
        for part in parts:
            for record in read_shard(part):
                if count >= shard_records and record.game_id != last_game:
                    handle.close()
                    handle = open_next()
                    count = 0
                handle.write(format_record_line(record))
                handle.write("\n")
                count += 1
                last_game = record.game_id
    finally:
        handle.close()
    if count == 0:
        # Only possible when every part was empty: drop the empty file.
        written.pop().unlink()
    return written
```

Add `Sequence` to the `collections.abc` import at the top of the module if it is not already there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/infrastructure/test_repack_shards.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/effects/infrastructure/record_io.py tests/unit/effects/infrastructure/
git commit -m "feat(effects): repack shards to a fixed size at game boundaries"
```

---

### Task 4: Survey and write pass apply quality, route gate-one, write parts, report per source

**Files:**
- Modify: `src/effects/application/build_corpus.py`
- Test: `tests/unit/effects/application/test_build_corpus.py` (append), `tests/unit/effects/application/test_build_corpus_survey.py` (append)

**Interfaces:**
- `SurveyConfig` gains `max_events: int`.
- `ShardSurvey` gains `quality_dropped: Counter[str]`.
- `Survey` gains `quality_dropped: Counter[str]`, `games_by_source: dict[str, int]`, `held_out_games_by_source: dict[str, int]`.
- `BuildCorpusConfig` gains `shard_records: int = 2000`, `max_events: int = 64`, `validation_sample: int = 2048`.
- `Decisions` gains `gate_one_keys: frozenset[str]` (rendered keys whose text is held out).
- `WriteConfig` gains `gate_one_dir: str`, `gate_one_keys: frozenset[str]`, `max_events: int`, and its four `*_dir` fields now point at **parts** directories.
- `WriteResult` gains `quality_dropped: Counter[str]`.
- `STRATA = ("training", "card-disjoint", "game-disjoint", "gate-one", "dropped-held-out")`; `OUTPUTS = STRATA[:-1]`.
- New helper `source_of(relative: str) -> str`: the first path component, or `"."`.

- [ ] **Step 1: Write the failing tests**

Append to `test_build_corpus_survey.py` (it already builds `ShardSurvey`s through `survey_shard`; follow its fixture for writing a raw shard):

```python
def test_a_defective_record_is_counted_and_otherwise_ignored(raw_shard_factory):
    """A record with no acting ability is not a key, a class or a game (FR-148)."""
    from effects.application.build_corpus import SurveyConfig, init_survey_worker, survey_shard
    from effects.domain.record_quality import NO_ABILITY

    shard = raw_shard_factory([
        make_resolution("clean", game="g1", ability=BOLT_KEY),
        make_resolution("junk", game="g2", ability=()),
    ])
    init_survey_worker(SurveyConfig(
        records_dir=str(shard.parent), held_out_names=frozenset(),
        held_out_script_files=frozenset(), text_cap=200, seed=1, max_events=64,
    ))
    out = survey_shard(shard.name)
    assert out.quality_dropped == {NO_ABILITY: 1}
    assert out.records == 2
    assert out.games == {"g1"}
    assert sum(out.class_records.values()) == 1
```

Adapt `raw_shard_factory`, `make_resolution` and `BOLT_KEY` to whatever helpers `test_build_corpus_survey.py` already defines for writing a shard and building a resolution record; the assertion set is what matters.

Append to `test_build_corpus.py`:

```python
def test_source_of_is_the_first_path_component():
    from effects.application.build_corpus import source_of
    assert source_of("depleted/run.0-a.jsonl.gz") == "depleted"
    assert source_of("run.0-a.jsonl.gz") == "."


def test_defective_records_reach_no_output_and_are_counted(tmp_path, a_corpus):
    """A junk resolution record in a training game and one in a held-out game both vanish."""
    from effects.domain.record_quality import NO_ABILITY, UNATTRIBUTED
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("junk-train", game="g-clean-1", ability=()),
        a_corpus.resolution(
            "junk-held", game="g-tainted",
            events=(Event(type=EventType.DAMAGE_DEALT, subjects=("P0",), attributed_to="unresolved"),),
        ),
    ])
    assert build(BuildCorpusConfig(records_dir=corpus.records_dir, output=tmp_path / "out",
                                   cards_folders=corpus.cards_folders, workers=1)) == 0
    store = CorpusStore(tmp_path / "out")
    ids = {r.record_id for d in (store.training_dir, store.card_disjoint_dir,
                                  store.game_disjoint_dir, store.gate_one_dir)
           for r in read_records(d)}
    assert "junk-train" not in ids and "junk-held" not in ids
    assert store.load().quality_dropped == {NO_ABILITY: 1, UNATTRIBUTED: 1}


def test_the_gate_one_slice_holds_held_out_resolutions_of_card_disjoint_games(tmp_path, a_corpus):
    assert build(BuildCorpusConfig(records_dir=a_corpus.records_dir, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards_folders, workers=1)) == 0
    store = CorpusStore(tmp_path / "out")
    manifest = store.load()
    slice_ = list(read_records(store.gate_one_dir))
    assert slice_, "g-tainted resolves the held-out text, so the slice is not empty"
    assert all(r.kind is RecordKind.RESOLUTION for r in slice_)
    assert all(r.game_id in manifest.card_disjoint_games for r in slice_)
    assert all(_HELD_OUT_KEY in (r.ability or ()) for r in slice_)
    card_disjoint_ids = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert {r.record_id for r in slice_} <= card_disjoint_ids
    assert manifest.per_stratum["gate-one"] == len(slice_)


def test_games_are_reported_per_source_directory(tmp_path, a_corpus):
    """Shards live under depleted/ and full-strength/ in the fixture."""
    assert build(BuildCorpusConfig(records_dir=a_corpus.records_dir, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards_folders, workers=1)) == 0
    manifest = CorpusStore(tmp_path / "out").load()
    assert set(manifest.games_by_source) == set(a_corpus.source_dirs)
    assert sum(manifest.games_by_source.values()) == a_corpus.game_count
    assert all(manifest.held_out_games_by_source.get(s, 0) <= n
               for s, n in manifest.games_by_source.items())


def test_output_shards_are_repacked_and_the_parts_removed(tmp_path, a_corpus):
    assert build(BuildCorpusConfig(records_dir=a_corpus.records_dir, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards_folders, workers=1,
                                   shard_records=2)) == 0
    store = CorpusStore(tmp_path / "out")
    assert not store.parts_dir.exists()
    names = sorted(p.name for p in store.training_dir.glob("*.jsonl.gz"))
    assert names and names[0] == "shard-00001.jsonl.gz"
    assert store.load().shard_records == 2
```

The `a_corpus` fixture does not yet expose `with_extra_records`, `resolution`, `source_dirs` or `game_count`. Add them to the fixture's dataclass in this task: `resolution(record_id, *, game, ability=..., events=...)` builds a resolution record the same way the fixture already builds `g-tainted`'s; `with_extra_records(records)` appends them to a new raw shard under the same `records_dir` and returns the fixture; `source_dirs` is the set of first path components of the raw shards; `game_count` the number of distinct games. If the fixture currently writes its raw shards flat, move them under `depleted/` and `full-strength/` (the tainted game's shard under `full-strength/`) so the per-source test has two sources; `test_no_training_record_belongs_to_a_withheld_game` and friends do not depend on the layout.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus.py tests/unit/effects/application/test_build_corpus_survey.py -v -k "defective or gate_one or per_source or repacked or source_of"`
Expected: FAIL (`TypeError` for `max_events`, `AttributeError` for `gate_one_dir` on the write config, missing `source_of`).

- [ ] **Step 3: Thread quality through the survey**

In `build_corpus.py`:

```python
from effects.domain.record_quality import quality_defect
```

`SurveyConfig`: add `max_events: int`. `ShardSurvey`: add `quality_dropped: Counter[str] = field(default_factory=Counter)`. In `survey_shard`, right after `out.records += 1`:

```python
        defect = quality_defect(record, max_events=config.max_events)
        if defect is not None:
            out.quality_dropped[defect] += 1
            continue
```

Add the helper:

```python
def source_of(relative: str) -> str:
    """The top-level directory a raw shard sits in: ``depleted``, ``full-strength``…

    ``"."`` for a shard at the root. What FR-149 reports held-out games
    against: a collection run is a directory, and a leak is a directory that
    should hold none.
    """
    return relative.split("/", 1)[0] if "/" in relative else "."
```

`Survey`: add `quality_dropped: Counter[str]`, `games_by_source: dict[str, int]`, `held_out_games_by_source: dict[str, int]`. In `merge_surveys`, accumulate:

```python
    quality_dropped: Counter[str] = Counter()
    games_by_source: dict[str, set[str]] = defaultdict(set)
    held_by_source: dict[str, set[str]] = defaultdict(set)
    ...
    for part in parts:
        ...
        quality_dropped.update(part.quality_dropped)
        source = source_of(part.name)
        games_by_source[source] |= part.games
        held_by_source[source] |= part.held_out_games
```

and pass `quality_dropped=quality_dropped`, `games_by_source={s: len(g) for s, g in games_by_source.items()}`, `held_out_games_by_source={s: len(g) for s, g in held_by_source.items()}` to the `Survey(...)` constructor.

- [ ] **Step 4: Config, decisions, and the gate-one key set**

`BuildCorpusConfig`: add `shard_records: int = 2000`, `max_events: int = 64`, `validation_sample: int = 2048`; add `("--shard-records", self.shard_records)`, `("--max-events-per-record", self.max_events)`, `("--validation-sample", self.validation_sample)` to the negative-value check in `__post_init__`.

`Decisions`: add `gate_one_keys: frozenset[str]`. In `decide`, after computing `key_text`:

```python
    gate_one_keys = frozenset(_held_out_text_of_key(survey, sidecars, held_out_texts))
```

Note `_held_out_text_of_key` only walks `survey.held_out_text_games`, which is keyed on records of held-out games; that is exactly the population the slice draws from. Pass `gate_one_keys=gate_one_keys` into the `Decisions(...)` constructor.

- [ ] **Step 5: The write pass writes parts, skips defects, routes gate-one**

`STRATA`/`OUTPUTS`:

```python
STRATA: tuple[str, ...] = (
    "training", "card-disjoint", "game-disjoint", "gate-one", "dropped-held-out",
)
OUTPUTS: tuple[str, ...] = STRATA[:-1]
```

`WriteConfig`: add `gate_one_dir: str`, `gate_one_keys: frozenset[str]`, `max_events: int`. `WriteResult`: add `quality_dropped: Counter[str] = field(default_factory=Counter)`.

In `write_shard_pass`, add `gate_one: list = []` beside the other three lists, and at the top of the loop body, right after `name = sampling_class(record)` and before `out.read[name] += 1`:

```python
        defect = quality_defect(record, max_events=config.max_events)
        if defect is not None:
            out.quality_dropped[defect] += 1
            continue
```

In the card-disjoint branch, before its `continue`:

```python
            if (
                record.kind is RecordKind.RESOLUTION
                and key is not None and key in config.gate_one_keys
            ):
                gate_one.append(record)
                out.stratum["gate-one"] += 1
                out.stratum_keys.setdefault("gate-one", set()).add(key)
```

Import `RecordKind` from `effects.domain.records`. Extend the final write loop with `(config.gate_one_dir, gate_one)`. The directories in `WriteConfig` are now parts directories; nothing in the function changes for that except the caller.

In `run_write_pass.absorb`, add `total.quality_dropped.update(part.quality_dropped)`.

- [ ] **Step 6: `build()` wires parts, repack, and the report**

In `build()`, replace the `WriteConfig(...)` directory arguments and add the repack step:

```python
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(
            records_dir=str(records_dir),
            training_dir=str(store.parts_dir_for("training")),
            card_disjoint_dir=str(store.parts_dir_for("card-disjoint")),
            game_disjoint_dir=str(store.parts_dir_for("game-disjoint")),
            gate_one_dir=str(store.parts_dir_for("gate-one")),
            gate_one_keys=decisions.gate_one_keys,
            max_events=config.max_events,
            thresholds=decisions.thresholds,
            class_admit=admit,
            card_disjoint=decisions.card_disjoint,
            game_disjoint=decisions.game_disjoint,
            held_out_games=survey.held_out_games,
            seed=config.seed,
        ),
        workers=config.workers,
    )
    _repack_outputs(store, shard_records=config.shard_records)
```

with the helper:

```python
def _repack_outputs(store, *, shard_records: int) -> None:
    """Stream each stratum's parts into uniform shards, then drop the parts."""
    import shutil

    from effects.infrastructure.record_io import iter_shards, repack_shards

    for stratum, target in (
        ("training", store.training_dir),
        ("card-disjoint", store.card_disjoint_dir),
        ("game-disjoint", store.game_disjoint_dir),
        ("gate-one", store.gate_one_dir),
    ):
        parts = iter_shards(store.parts_dir_for(stratum))
        paths = repack_shards(parts, target, shard_records=shard_records)
        logger.info("%-22s %d part(s) repacked into %d shard(s)", stratum, len(parts), len(paths))
    shutil.rmtree(store.parts_dir, ignore_errors=True)
```

Before `store.clear_outputs()` nothing changes. After the write, add the per-source report:

```python
    for source in sorted(survey.games_by_source):
        games = survey.games_by_source[source]
        held = survey.held_out_games_by_source.get(source, 0)
        share = held / games if games else 0.0
        report = logger.warning if 0 < share < 0.05 else logger.info
        report(
            "%-22s %7d game(s), %6d name a held-out card (%.1f%%)%s",
            source, games, held, 100.0 * share,
            " — a few held-out games in a directory that should hold none is "
            "a leak in collection, not a full-strength source" if 0 < share < 0.05 else "",
        )
    for reason, count in sorted(written.quality_dropped.items()):
        logger.info("%-22s %9d record(s) refused", reason, count)
```

Pass the new fields into `CorpusManifest(...)`: `quality_dropped=dict(written.quality_dropped)`, `games_by_source=survey.games_by_source`, `held_out_games_by_source=survey.held_out_games_by_source`, `shard_records=config.shard_records`, `validation_sample=config.validation_sample`, `max_events_per_record=config.max_events`. Also pass `max_events=config.max_events` into `SurveyConfig(...)`.

- [ ] **Step 7: Run the whole build test suite**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus.py tests/unit/effects/application/test_build_corpus_survey.py tests/unit/effects/application/test_build_corpus_decide.py -v`
Expected: all pass. `test_no_empty_output_shard_is_written` and `test_a_rebuild_removes_a_shard_the_new_build_does_not_write` may need their shard-name expectations updated from source-derived names to `shard-00001.jsonl.gz`; keep their intent.

- [ ] **Step 8: Commit**

```bash
git add src/effects/application/build_corpus.py tests/unit/effects/application/
git commit -m "feat(effects): build-corpus refuses defective records, writes the gate-one slice, repacks shards, reports per source"
```

---

### Task 5: Fixed validation samples

**Files:**
- Create: `src/effects/application/validation_samples.py`
- Modify: `src/effects/application/build_corpus.py` (call it from `build()`)
- Test: `tests/unit/effects/application/test_validation_samples.py`

**Interfaces:**
- Produces: `class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]` (`max(1, round(share * size))` per class, the same rule the trainer used).
- Produces: `draw_samples(*, card_disjoint: Path, game_disjoint: Path, gate_one: Path, mix, size, seed) -> dict[str, list[EffectRecord]]` returning `{"card-disjoint": [...], "game-disjoint": [...]}`, each shuffled by `random.Random(f"{seed}:{stratum}")`. Per stratum and class the sample is the `quota[class]` records with the smallest `record_hash(record_id, seed=seed + SAMPLE_SEED_OFFSET)`. For `card-disjoint`, classes `resolution-effect` and `resolution-cost` are drawn from `gate_one` first and topped up from `card_disjoint` only when the slice is short.
- Produces: `write_samples(store, samples) -> dict[str, int]` writing `store.sample_path(stratum)` via `write_shard` and returning counts.
- `SAMPLE_SEED_OFFSET = 0x2545F491`.

- [ ] **Step 1: Write the failing tests**

```python
"""The fixed validation samples build-corpus writes for the trainer."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.validation_samples import class_quota, draw_samples
from effects.domain.effect_model import (
    CLASS_COMBAT, CLASS_RESOLUTION_COST, CLASS_RESOLUTION_EFFECT,
)
from effects.domain.records import CombatPayload, Moment, RecordKind
from effects.infrastructure.record_io import write_shard


def test_class_quota_rounds_each_share_and_keeps_every_class():
    quota = class_quota({"a": 0.5, "b": 0.5, "c": 0.001}, 100)
    assert quota == {"a": 50, "b": 50, "c": 1}


@pytest.fixture
def strata(tmp_path, make_record):
    def res(i, game, moment=Moment.RESOLUTION):
        return make_record(record_id=f"r{i}", game_id=game, moment=moment)
    def combat(i, game):
        return make_record(record_id=f"c{i}", game_id=game, kind=RecordKind.COMBAT,
                           payload=CombatPayload())
    card = [res(i, f"cg{i % 3}") for i in range(30)] + [combat(i, f"cg{i % 3}") for i in range(30)]
    gate = card[:6]                       # six of the card-disjoint resolutions are gate-one
    game = [res(i, f"gg{i % 3}") for i in range(100, 130)] + [combat(i, f"gg{i % 3}") for i in range(100, 130)]
    for name, records in (("card", card), ("game", game), ("gate", gate)):
        write_shard(tmp_path / name / "shard-00001.jsonl.gz", records)
    return tmp_path


MIX = {CLASS_RESOLUTION_EFFECT: 0.5, CLASS_COMBAT: 0.5}


def test_each_stratum_holds_the_quota_per_class(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    for stratum in ("card-disjoint", "game-disjoint"):
        by_class = Counter(sampling_class(r) for r in samples[stratum])
        assert by_class == {CLASS_RESOLUTION_EFFECT: 10, CLASS_COMBAT: 10}


def test_card_disjoint_resolutions_come_from_the_gate_one_slice_first(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    resolutions = {r.record_id for r in samples["card-disjoint"] if r.kind is RecordKind.RESOLUTION}
    assert {f"r{i}" for i in range(6)} <= resolutions      # all six gate-one records
    assert len(resolutions) == 10                           # topped up from the stratum


def test_the_draw_is_seeded_and_shuffled(strata):
    a = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    b = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    c = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=2)
    ids = lambda s: [r.record_id for r in s["game-disjoint"]]
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    classes = [sampling_class(r) for r in a["game-disjoint"]]
    assert classes != sorted(classes), "a class-sorted sample scores one class per batch"


def test_a_short_class_takes_what_there_is(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix={CLASS_RESOLUTION_COST: 1.0}, size=5, seed=1)
    assert samples["card-disjoint"] == []


from effects.application.train_effect_model import sampling_class  # noqa: E402
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/effects/application/test_validation_samples.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""The fixed validation samples a curated corpus ships (FR-135, FR-089).

The trainer used to sweep every validation shard at startup to draw a
mixture-matched sample, and drew 20 gate-1 records into a 2,048-record
card-disjoint sample because it filled each class with whatever arrived
first. The sample is a property of the corpus, drawn once here by smallest
seeded record hash — the same device the per-text cap uses — and written
beside the strata so the trainer reads it and the evaluator can name it.

For the card-disjoint stratum the two resolution classes draw from the
gate-one slice first: that is the population the model ships on, and the
number that selects checkpoints should measure it.
"""

from __future__ import annotations

import heapq
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from effects.domain.corpus_curation import record_hash
from effects.domain.effect_model import CLASS_RESOLUTION_COST, CLASS_RESOLUTION_EFFECT
from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

SAMPLE_SEED_OFFSET = 0x2545F491
STRATA = ("card-disjoint", "game-disjoint")
#: Card-disjoint classes drawn from the gate-one slice before the stratum.
GATE_ONE_CLASSES = frozenset({CLASS_RESOLUTION_EFFECT, CLASS_RESOLUTION_COST})


def class_quota(mix: Mapping[str, float], size: int) -> dict[str, int]:
    """Records per class in a sample of ``size``: each share rounded, never zero."""
    return {name: max(1, round(share * size)) for name, share in mix.items()}


class _Smallest:
    """The ``cap`` records with the smallest hashes seen, by heap."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._heap: list[tuple[int, str, EffectRecord]] = []

    def offer(self, value: int, record: EffectRecord) -> None:
        item = (-value, record.record_id, record)
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:
            heapq.heappushpop(self._heap, item)

    def records(self) -> list[EffectRecord]:
        return [record for _, _, record in sorted(self._heap, reverse=True)]

    def __len__(self) -> int:
        return len(self._heap)


def _collect(
    directory: Path, quota: Mapping[str, int], *, seed: int,
    only: frozenset[str] | None = None,
) -> dict[str, _Smallest]:
    from effects.application.train_effect_model import sampling_class
    from effects.infrastructure.record_io import read_records

    heaps = {name: _Smallest(cap) for name, cap in quota.items()}
    for record in read_records(directory):
        name = sampling_class(record)
        if name not in heaps or (only is not None and name not in only):
            continue
        heaps[name].offer(record_hash(record.record_id, seed=seed + SAMPLE_SEED_OFFSET), record)
    return heaps


def draw_samples(
    *, card_disjoint: Path, game_disjoint: Path, gate_one: Path,
    mix: Mapping[str, float], size: int, seed: int,
) -> dict[str, list[EffectRecord]]:
    """One mixture-matched sample per stratum, seeded and shuffled."""
    quota = class_quota(mix, size)
    out: dict[str, list[EffectRecord]] = {}

    gate = _collect(gate_one, quota, seed=seed, only=GATE_ONE_CLASSES)
    for stratum, directory in (("card-disjoint", card_disjoint), ("game-disjoint", game_disjoint)):
        heaps = _collect(directory, quota, seed=seed)
        sample: list[EffectRecord] = []
        for name, cap in quota.items():
            chosen: list[EffectRecord] = []
            if stratum == "card-disjoint" and name in GATE_ONE_CLASSES:
                chosen = gate[name].records()[:cap]
            taken = {r.record_id for r in chosen}
            for record in heaps[name].records():
                if len(chosen) >= cap:
                    break
                if record.record_id not in taken:
                    chosen.append(record)
            if len(chosen) < cap:
                logger.warning(
                    "%s sample: %s holds %d record(s) of the %d its share asks for",
                    stratum, name, len(chosen), cap,
                )
            sample.extend(chosen)
        random.Random(f"{seed}:{stratum}").shuffle(sample)
        out[stratum] = sample
    return out


def write_samples(store, samples: Mapping[str, Iterable[EffectRecord]]) -> dict[str, int]:
    from effects.infrastructure.record_io import write_shard

    return {
        stratum: write_shard(store.sample_path(stratum), records)
        for stratum, records in samples.items()
    }
```

In `build()`, after `_repack_outputs(...)`:

```python
    from effects.application.validation_samples import draw_samples, write_samples

    samples = draw_samples(
        card_disjoint=store.card_disjoint_dir, game_disjoint=store.game_disjoint_dir,
        gate_one=store.gate_one_dir, mix=config.mix(), size=config.validation_sample,
        seed=config.seed,
    )
    for stratum, count in write_samples(store, samples).items():
        logger.info("%-22s %9d record(s) sampled for validation", stratum, count)
```

Append to `test_build_corpus.py`:

```python
def test_the_build_writes_a_validation_sample_per_stratum(tmp_path, a_corpus):
    assert build(BuildCorpusConfig(records_dir=a_corpus.records_dir, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards_folders, workers=1,
                                   validation_sample=4)) == 0
    store = CorpusStore(tmp_path / "out")
    for stratum in ("card-disjoint", "game-disjoint"):
        assert store.sample_path(stratum).exists()
        assert 0 < sum(1 for _ in read_shard(store.sample_path(stratum))) <= 4 * 8
    assert store.load().validation_sample == 4
```

(`read_shard` import from `effects.infrastructure.record_io`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/effects/application/test_validation_samples.py tests/unit/effects/application/test_build_corpus.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/effects/application/validation_samples.py src/effects/application/build_corpus.py tests/unit/effects/application/
git commit -m "feat(effects): build-corpus writes fixed validation samples drawn from the gate-one slice"
```

---

### Task 6: CLI flags and documentation

**Files:**
- Modify: `src/effects/infrastructure/cli.py` (`_build_corpus_parser`, `run_build_corpus`)
- Modify: `specs/023-ability-effect-model/quickstart.md` (§ build-corpus outputs), `specs/023-ability-effect-model/spec.md` (FR-135, add FR-148, FR-149)
- Test: `tests/unit/effects/application/test_build_corpus.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_the_cli_exposes_the_rework_flags():
    from effects.infrastructure.cli import build_parser
    args = build_parser().parse_args([
        "build-corpus", "--shard-records", "500", "--max-events-per-record", "32",
        "--validation-sample", "64",
    ])
    assert (args.shard_records, args.max_events_per_record, args.validation_sample) == (500, 32, 64)
    defaults = build_parser().parse_args(["build-corpus"])
    assert (defaults.shard_records, defaults.max_events_per_record, defaults.validation_sample) == (2000, 64, 2048)
```

Use whatever the module names its top-level parser factory (grep `def build_parser\|def main` in `cli.py`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_build_corpus.py -k rework_flags -v`
Expected: FAIL with `SystemExit: 2` (unrecognized arguments)

- [ ] **Step 3: Add the flags**

In `_build_corpus_parser`, after `--card-disjoint-text-cap`:

```python
    parser.add_argument(
        "--shard-records", type=int, default=2000,
        help=(
            "Records per output shard after repacking, cut at game boundaries; "
            "0 keeps one output shard per source shard (default: 2000)"
        ),
    )
    parser.add_argument(
        "--max-events-per-record", type=int, default=64,
        help=(
            "A record carrying more events than this is refused as a game "
            "dump rather than an ability (default: 64)"
        ),
    )
    parser.add_argument(
        "--validation-sample", type=int, default=2048,
        help=(
            "Records per stratum in validation/samples/, mixture-matched; the "
            "card-disjoint resolution classes draw from the gate-one slice "
            "(default: 2048)"
        ),
    )
```

In `run_build_corpus`, pass `shard_records=args.shard_records`, `max_events=args.max_events_per_record`, `validation_sample=args.validation_sample` into `BuildCorpusConfig(...)`.

- [ ] **Step 4: Document**

In `quickstart.md`, in the section that lists what `build-corpus` writes (search for `validation/game-disjoint/` and `manifest.json`), replace the output list with:

```
training/                          shard-00001.jsonl.gz … (--shard-records each, games never split)
validation/card-disjoint/          whole held-out games, repacked the same way
validation/game-disjoint/          whole clean games, repacked the same way
validation/gate-one/               resolution records of card-disjoint games whose acting text is held out
validation/samples/card-disjoint.jsonl.gz   the trainer's per-epoch validation set; its resolution classes come from gate-one
validation/samples/game-disjoint.jsonl.gz
manifest.json                      + quality_dropped, games_by_source, held_out_games_by_source, shard_records, validation_sample
```

and add a paragraph under it:

> Three kinds of record are refused on sight and counted in `quality_dropped`: a resolution record with no acting ability, a record with an event attributed to `unresolved`, and a record with more than `--max-events-per-record` events. The per-source report warns when a directory routes a few games to the card-disjoint stratum; a depleted directory should route none, and a small count there is a leak in collection (in the first corpus, The Hobbit and Marvel Super Heroes boosters).

In `spec.md`, extend FR-135's output list with the two new outputs and the repacking rule, and add after FR-147:

```
- **FR-148**: `build-corpus` MUST refuse a resolution record with no acting ability, any record
  carrying an event attributed to `unresolved`, and any record carrying more than
  `--max-events-per-record` events (default 64), in both passes, and MUST record the count per
  reason in the manifest as `quality_dropped`.
- **FR-149**: `build-corpus` MUST record, per top-level directory under `--records-dir`, the games
  it read and the games naming a held-out card (`games_by_source`, `held_out_games_by_source`),
  and MUST warn when a directory routes more than none and fewer than 5% of its games.
```

- [ ] **Step 5: Run the test and the whole effects unit suite**

Run: `python -m pytest tests/unit/effects -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/effects/infrastructure/cli.py specs/023-ability-effect-model/quickstart.md specs/023-ability-effect-model/spec.md tests/unit/effects/application/test_build_corpus.py
git commit -m "feat(effects): build-corpus flags and docs for repacking, quality and samples"
```

---

### Task 7: Rebuild the real corpus and check the manifest

**Files:** none (operational).

- [ ] **Step 1: Rebuild**

Do this after the running training job has stopped; the build is CPU-only but the repack and sample passes read tens of gigabytes.

```bash
python -m effects build-corpus --records-dir output/effects/records/ --output output/effects/corpus/ \
    --variant-scripts output/effects/variant-scripts/ --vocab-path models/effects/vocab-script.txt
```

- [ ] **Step 2: Read the manifest**

```bash
python -c "import json; m=json.load(open('output/effects/corpus/manifest.json')); print({k: m[k] for k in ('per_stratum','quality_dropped','games_by_source','held_out_games_by_source','shard_records','validation_sample')})"
```

Expected against the diagnosis:
- `per_stratum["gate-one"]` in the tens of thousands (the estimate was ~35,000);
- `quality_dropped` in the low percent of the resolution count: `no-ability` around 3%, `unattributed-events` around 10%, `event-flood` a few hundred;
- `held_out_games_by_source["depleted"]` about 65 against ~52,000 games, and a warning line for it in the build log;
- training shards named `shard-*.jsonl.gz` with about `per_stratum["training"] / 2000` files.

- [ ] **Step 3: Record the numbers**

Add a dated entry to `experiments/2026-09-04-ability-effect-model-design.md` (or the diary the project uses for runs) with the manifest figures and the build wall clock, so the next plan's training run has a baseline to cite.

---

## Self-review

- **Spec coverage.** FR-135 outputs: Tasks 2, 4, 5. FR-148 quality: Tasks 1, 4. FR-149 per-source report: Task 4. Repacking: Tasks 3, 4. Manifest compatibility: Task 2. Flags and docs: Task 6.
- **Type consistency.** `quality_defect(record, *, max_events)` in Task 1 is what Task 4 calls; `repack_shards(parts, out_dir, *, shard_records)` in Task 3 is what `_repack_outputs` calls; `draw_samples`/`write_samples`/`class_quota` in Task 5 match the `build()` call; `CorpusStore.sample_path`, `gate_one_dir`, `parts_dir_for` from Task 2 are used in Tasks 4 and 5.
- **Open question for the executor.** `test_build_corpus.py`'s `a_corpus` fixture needs the four helpers Task 4 names. If extending it proves harder than expected, build the extra records with the module-level helpers that fixture already uses rather than adding methods; the assertions are the deliverable.
