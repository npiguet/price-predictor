# Coverage and Variant Deck Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `collect-coverage` and `collect-variants` actually play the decks they compute, and stop when the round is done.

**Architecture:** Python already computes everything that decides *what* to collect — coverage weights, castability verdicts, variant scripts — and then discards it. This plan closes the gap in three places: Python builds 40-card decks from the weights and writes them to a decks file; the Java worker gains a decks-only mode that plays a pair from that file per match and never opens a sealed pool; and `play_round` bounds the round with `should_stop` against a real progress file. No new supervisor, no new worker — the existing `ForgeWorkerPool`, `GeneratedDecksIndex` and `MatchWorkerMain` carry it.

**Tech Stack:** Python 3.14 + pytest + ruff; Java 17 + JUnit 5 + Maven; Forge as a sibling checkout.

**Spec:** `specs/023-ability-effect-model/spec.md` (FR-044…FR-052 coverage, FR-053…FR-059 variants), root spec `specs/2026-09-05-ability-effect-model.md` § Coverage collector, § Synthetic script variants.

## Global Constraints

- `effects` imports from `sealed` and `price_predictor`, never the reverse, and never `sealed.application`. Asserted by `tests/unit/effects/test_import_boundaries.py`. `compute_basic_lands` is already on the declared surface.
- Coverage and variant matches write effect records only, and **never** `output/sealed/match-outcomes.txt` or `output/sealed/cards-played.txt` (FR-052, FR-059). A progress file under `output/effects/` is not the sealed corpus and is permitted; FR-052/FR-059 get a clarifying amendment in Task 7.
- A coverage deck is 40 cards: 23 nonlands plus basics from `compute_basic_lands` (FR-046).
- The castability consult ranks only; it never drops a card (FR-047).
- Card-name comparisons across the converted-tree/Forge boundary fold case through `fold_card_name` (`src/effects/application/train_effect_model.py`).
- Run `python -m ruff check src tests` before every commit; it must print `All checks passed!`.

---

### Task 1: Build coverage decks from the weights

**Files:**
- Modify: `src/effects/application/collect_coverage.py`
- Test: `tests/unit/effects/application/test_coverage.py`

**Interfaces:**
- Consumes: `deck_weights(coverage, target) -> dict[str, float]` and `rank_by_consult(weights, verdicts) -> dict[str, float]`, both already in `collect_coverage.py`.
- Produces: `build_coverage_decks(weights: dict[str, float], texts: dict[str, str], count: int, *, rng: random.Random) -> list[list[str]]` — each inner list is 40 card names, 23 nonlands (duplicates allowed) plus basics.

- [ ] **Step 1: Write the failing test**

```python
class TestCoverageDeckBuilding:
    """A coverage deck is 40 cards and favours the cards that need records."""

    def _texts(self, names):
        return {n: f"name: {n}\nmana cost: {{1}}{{G}}\ntypes: creature\n" for n in names}

    def test_a_deck_is_forty_cards_with_twenty_three_nonlands(self):
        from effects.application.collect_coverage import build_coverage_decks
        import random

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {n: 1.0 for n in names}, self._texts(names), count=3,
            rng=random.Random(1),
        )

        assert len(decks) == 3
        for deck in decks:
            assert len(deck) == 40

    def test_a_heavier_card_appears_more_often(self):
        from effects.application.collect_coverage import build_coverage_decks
        import random

        names = [f"card {i}" for i in range(40)]
        weights = {n: 1.0 for n in names}
        weights["card 0"] = 500.0
        decks = build_coverage_decks(
            weights, self._texts(names), count=40, rng=random.Random(1),
        )

        flat = [n for deck in decks for n in deck]
        assert flat.count("card 0") > flat.count("card 1")

    def test_a_card_with_no_converted_text_is_skipped(self):
        """A weight can name a card the tree has no text for; basics need text."""
        from effects.application.collect_coverage import build_coverage_decks
        import random

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {**{n: 1.0 for n in names}, "ghost": 5.0},
            self._texts(names), count=2, rng=random.Random(1),
        )

        assert all("ghost" not in deck for deck in decks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_coverage.py -q -k CoverageDeckBuilding`
Expected: FAIL — `ImportError: cannot import name 'build_coverage_decks'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/effects/application/collect_coverage.py`:

```python
def build_coverage_decks(
    weights: dict[str, float],
    texts: dict[str, str],
    count: int,
    *,
    rng: random.Random,
) -> list[list[str]]:
    """``count`` 40-card decks drawn from ``weights`` (FR-046).

    23 nonlands sampled with replacement in proportion to weight, then basics
    from ``compute_basic_lands`` over the chosen cards' converted text. With
    replacement because a weight of 500 against 1 should be able to fill a deck
    with one card: the point is to get that card into games, not to build a
    deck anyone would play.

    A weighted card with no converted text is skipped — the manabase heuristic
    reads the text, and a card without one cannot be placed.
    """
    from sealed.domain.manabase import compute_basic_lands

    pool = [(name, w) for name, w in weights.items() if name in texts and w > 0]
    if not pool:
        return []
    names = [name for name, _ in pool]
    ws = [w for _, w in pool]

    decks: list[list[str]] = []
    for _ in range(count):
        nonlands = rng.choices(names, weights=ws, k=NONLANDS_PER_DECK)
        basics = compute_basic_lands([texts[n] for n in nonlands])
        deck = list(nonlands)
        for land, n in basics.items():
            deck.extend([land] * n)
        decks.append(deck)
    return decks
```

Add `import random` to the module imports if absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/effects/application/test_coverage.py -q -k CoverageDeckBuilding`
Expected: PASS (3 passed)

- [ ] **Step 5: Delete the constant-only test it replaces**

Remove `test_a_deck_is_forty_cards_with_twenty_three_nonlands` from `class TestConfig` (it asserts `DECK_SIZE == 40` and `NONLANDS_PER_DECK == 23`, which tests two constants against themselves). The new test of the same name in `TestCoverageDeckBuilding` covers it against real behaviour.

- [ ] **Step 6: Run the suite and lint**

Run: `python -m pytest tests/unit/effects -q && python -m ruff check src tests`
Expected: all pass, `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/effects/application/collect_coverage.py tests/unit/effects/application/test_coverage.py
git commit -m "feat(effects): build coverage decks from the scarcity weights"
```

---

### Task 2: Write the decks to a file the worker can read

**Files:**
- Create: `src/effects/infrastructure/deck_file.py`
- Test: `tests/unit/effects/infrastructure/test_deck_file.py`

**Interfaces:**
- Consumes: `build_coverage_decks` from Task 1.
- Produces: `write_deck_file(decks: list[list[str]], path: Path, *, label: str, set_code: str) -> int` — writes `LABEL;SET_CODE;Card1|...|Card40` per line, the existing generated-decks format, and returns the line count. `COVERAGE_SET_CODE: str = "COVERAGE"`.

- [ ] **Step 1: Write the failing test**

```python
"""The decks file a coverage or variant round hands its workers.

The generated-decks format already read by `GeneratedDecksIndex`, so the worker
needs no new parser. The set code is a sentinel: a coverage deck is drawn from
the whole corpus and belongs to no set, and Task 3's decks-only mode is what
stops anything from trying to resolve it against Forge's set table.
"""

from __future__ import annotations

from pathlib import Path

from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file


class TestDeckFile:
    def test_one_line_per_deck_in_the_generated_decks_format(self, tmp_path):
        path = tmp_path / "decks.txt"

        written = write_deck_file(
            [["a", "b"], ["c"]], path, label="coverage", set_code=COVERAGE_SET_CODE,
        )

        assert written == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "coverage;COVERAGE;a|b"
        assert lines[1] == "coverage;COVERAGE;c"

    def test_it_overwrites_rather_than_appends(self, tmp_path):
        """Each round's decks replace the last round's, never join them."""
        path = tmp_path / "decks.txt"
        write_deck_file([["a"]], path, label="coverage", set_code=COVERAGE_SET_CODE)

        write_deck_file([["b"]], path, label="coverage", set_code=COVERAGE_SET_CODE)

        assert path.read_text(encoding="utf-8").splitlines() == ["coverage;COVERAGE;b"]

    def test_no_deck_writes_an_empty_file(self, tmp_path):
        path = tmp_path / "decks.txt"

        assert write_deck_file([], path, label="coverage", set_code=COVERAGE_SET_CODE) == 0
        assert path.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/effects/infrastructure/test_deck_file.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'effects.infrastructure.deck_file'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Writing a round's decks where the Forge worker can read them.

The generated-decks format (`LABEL;SET_CODE;Card1|...`) rather than a new one,
because `GeneratedDecksIndex` already parses it and the worker already knows how
to sample from it. What a coverage round needs beyond that is only that nothing
tries to resolve the set code — see the decks-only mode.
"""

from __future__ import annotations

from pathlib import Path

#: Sentinel set code for decks drawn from the whole corpus rather than a set.
#: Never resolved against Forge's set table: the decks-only mode plays both
#: sides from the file and opens no pool.
COVERAGE_SET_CODE = "COVERAGE"


def write_deck_file(
    decks: list[list[str]], path: Path, *, label: str, set_code: str,
) -> int:
    """Write one deck per line, replacing whatever was there. Returns the count.

    Replacing rather than appending: a round's decks are built for that round's
    weights, and last round's decks would pull play back toward cards that have
    since been satisfied.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{label};{set_code};{'|'.join(deck)}" for deck in decks]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/effects/infrastructure/test_deck_file.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/effects/infrastructure/deck_file.py tests/unit/effects/infrastructure/test_deck_file.py
git commit -m "feat(effects): write a round's decks in the generated-decks format"
```

---

### Task 3: A decks-only mode in the Java worker

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/MatchGenerator.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/MatchWorkerMain.java`
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorRoutingTest.java`

**Interfaces:**
- Consumes: the decks file from Task 2, via the existing `-Dside.a.decks.file` / `-Dside.b.decks.file` properties and `GeneratedDecksIndex`.
- Produces: system property `sealed.decks.only` (`"true"` enables); `MatchGenerator` package-private constructor gains a trailing `boolean decksOnly`.

**Note on the test:** `sideBIndex` must be non-null — `rollIsFileSample`
returns false early without an index, and decks-only does not remove that
precondition. Use the file's existing `emptyIndex()` helper, which builds a
`GeneratedDecksIndex` without Forge.

**Why a mode rather than a weight:** `pickDeckB` rolls between the four Forge methods and the file. When the roll picks a Forge method it calls `generatePool(a.setCode)`, which resolves the set code against Forge's booster and edition tables. `COVERAGE` is in neither, so the roll NPEs. Raising `sideBWeight` makes that rare, not impossible.

- [ ] **Step 1: Write the failing test**

Add to `MatchGeneratorRoutingTest` (Forge-free — it tests routing only):

```java
    @Test
    void decksOnlyAlwaysRollsTheFileForSideB() {
        // A coverage deck belongs to no set, so the Forge-method branch would
        // try to open a booster for a set code Forge has never heard of.
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, RUN_ID, null, emptyIndex(), 1,
                new Random(7), Set.of(), true);

        for (int i = 0; i < 100; i++) {
            assertTrue(generator.rollIsFileSample(),
                    "decks-only mode rolled a Forge method");
        }
    }

    @Test
    void withoutDecksOnlyTheRollStillMixes() {
        MatchGenerator generator = new MatchGenerator(
                List.of("RVR"), null, null, RUN_ID, null, emptyIndex(), 4,
                new Random(7), Set.of(), false);

        boolean sawForgeMethod = false;
        for (int i = 0; i < 200; i++) {
            if (!generator.rollIsFileSample()) {
                sawForgeMethod = true;
                break;
            }
        }
        assertTrue(sawForgeMethod, "the ordinary roll never picked a Forge method");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd forge-connector && mvn -o test -Dtest=MatchGeneratorRoutingTest`
Expected: FAIL — compilation error, constructor does not take 10 arguments

- [ ] **Step 3: Write minimal implementation**

In `MatchGenerator.java`, add the field and thread it through both constructors (the public one passes `false`, matching today's behaviour), then make the roll honour it:

```java
    private final boolean decksOnly;
```

```java
    boolean rollIsFileSample() {
        if (sideBIndex == null) {
            return false;
        }
        // A decks-only round has no set to open a booster for: both sides come
        // from the decks file or the match cannot be built at all.
        if (decksOnly) {
            return true;
        }
        double total = FORGE_METHODS_TOTAL_WEIGHT + sideBWeight;
        return random.nextDouble() < sideBWeight / total;
    }
```

In `MatchWorkerMain.java`, read the property and pass it to the generator:

```java
        boolean decksOnly = Boolean.parseBoolean(
                System.getProperty("sealed.decks.only", "false"));
```

and add `decksOnly` as the trailing argument where `MatchGenerator` is constructed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd forge-connector && mvn -o test -Dtest=MatchGeneratorRoutingTest`
Expected: PASS

- [ ] **Step 5: Run the whole Java suite**

Run: `cd forge-connector && mvn -o test`
Expected: `BUILD SUCCESS`, 751+ tests, 0 failures

- [ ] **Step 6: Commit**

```bash
git add forge-connector/src
git commit -m "feat(connector): decks-only mode, so a set-less deck never opens a pool"
```

---

### Task 4: Bound the round and count its progress

**Files:**
- Modify: `src/effects/infrastructure/collector_connector.py`
- Test: `tests/unit/effects/infrastructure/test_collector_connector.py` (create if absent)

**Interfaces:**
- Consumes: `write_deck_file`, `COVERAGE_SET_CODE` (Task 2); `ForgeWorkerPool(..., should_stop=Callable[[int], bool])`.
- Produces: `CollectorSupervisor.play_round(decks_file: Path, *, matches: int) -> None`; `CollectorSupervisor.progress_path` (a `Path` under the records directory).

**Why the signature changes:** `play_round(weights, cards_folder, *, decks)` took the weights and threw them away. The caller now builds the decks (Task 1) and writes them (Task 2), so the supervisor takes a file and a match budget — the two things it actually uses.

**Progress counting:** coverage workers run records-only and write no `match-outcomes.txt`, so there is nothing to count lines of; `output_path` was being handed the records *directory*, and `open()` on a directory throws and reads 0 forever. The worker now writes its match rows to `<records>/<run_id>.progress.txt` — under `output/effects/`, never the sealed corpus — and that file's line count is the round's progress.

- [ ] **Step 1: Write the failing test**

```python
"""The round supervisor: what it hands the workers and when it stops.

`play_round` used to take the weights and the deck budget and log both without
using either, so a round never ended and the decks were never the coverage
decks the caller had computed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from effects.infrastructure.collector_connector import CollectorSupervisor


class TestPlayRound:
    def _supervisor(self, tmp_path):
        return CollectorSupervisor(worker_count=2, effect_records=tmp_path)

    def test_the_round_stops_at_its_match_budget(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(tmp_path / "decks.txt", matches=500)

        stop = pool.call_args.kwargs["should_stop"]
        assert stop(499) is False
        assert stop(500) is True
        assert stop(501) is True

    def test_progress_is_counted_from_a_file_not_a_directory(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(tmp_path / "decks.txt", matches=10)

        output_path = pool.call_args.kwargs["output_path"]
        assert output_path.is_file() or not output_path.exists()
        assert output_path != Path(tmp_path)

    def test_the_decks_file_reaches_the_worker_on_both_sides(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        decks = tmp_path / "decks.txt"
        with patch.object(supervisor, "_connector") as connector:
            supervisor._decks_file = decks
            supervisor.start_worker(0)

        kwargs = connector.start.call_args.kwargs
        assert kwargs["side_a_decks_path"] == decks
        assert kwargs["side_b_decks_path"] == decks
        assert kwargs["decks_only"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/effects/infrastructure/test_collector_connector.py -q`
Expected: FAIL — `play_round() takes ... positional arguments` / `decks_only` not passed

- [ ] **Step 3: Write minimal implementation**

In `collector_connector.py`, add `self._decks_file: Path | None = None` to `__init__`, add the progress path, and replace `play_round`:

```python
    @property
    def progress_path(self) -> Path:
        """The file a round's completed matches are counted from.

        Under ``output/effects/``, never the sealed corpus: FR-052 and FR-059
        forbid a coverage or variant run reaching ``match-outcomes.txt``, and
        this is a per-run scratch file that nothing downstream reads.
        """
        return self._effect_records / f"{self._run_id}.progress.txt"

    def play_round(self, decks_file: Path, *, matches: int) -> None:
        """Play ``matches`` matches from ``decks_file``, then stop.

        The budget is what makes a round a round: without it the pool runs until
        it is signalled, the caller never recounts coverage, nothing retires,
        and the run cannot finish.
        """
        self._decks_file = Path(decks_file)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text("", encoding="utf-8")
        self._pool = ForgeWorkerPool(
            worker_count=self._worker_count,
            spawn_worker=self.start_worker,
            output_path=self.progress_path,
            should_stop=lambda completed: completed >= matches,
        )
        logger.info(
            "Playing %d matches from %d decks", matches,
            sum(1 for _ in self._decks_file.open(encoding="utf-8")),
        )
        self._pool.run()
```

and in `start_worker`, hand the worker the decks file and the progress file:

```python
        process = self._connector.start(
            self.progress_path,
            run_id=self._run_id,
            best_of=COVERAGE_BEST_OF,
            log_file=self._worker_logs.get(worker_id),
            effect_records_dir=self._effect_records,
            worker_index=worker_id,
            collection_caps=self._caps.as_system_properties(),
            side_a_decks_path=self._decks_file,
            side_b_decks_path=self._decks_file,
            decks_only=True,
        )
```

- [ ] **Step 4: Add `decks_only` to the worker connector**

In `src/sealed/infrastructure/match_worker_connector.py`, add the parameter and the property:

```python
        decks_only: bool = False,
```

```python
        if decks_only:
            system_properties["sealed.decks.only"] = "true"
```

Document it: *"Both sides sample from the decks file and no pool is opened. For decks that belong to no set — a coverage or variant deck drawn from the whole corpus — where resolving the set code would fail."*

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/effects tests/unit/sealed -q && python -m ruff check src tests`
Expected: PASS, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/effects/infrastructure/collector_connector.py src/sealed/infrastructure/match_worker_connector.py tests/unit/effects/infrastructure/test_collector_connector.py
git commit -m "feat(effects): bound a collection round and count its progress"
```

---

### Task 5: Wire the coverage round to the new pieces

**Files:**
- Modify: `src/effects/application/collect_coverage.py:318-390` (the `run` function)
- Test: `tests/unit/effects/application/test_coverage.py`

**Interfaces:**
- Consumes: `build_coverage_decks` (Task 1), `write_deck_file`/`COVERAGE_SET_CODE` (Task 2), `CollectorSupervisor.play_round(decks_file, matches=…)` (Task 4).
- Produces: no new public surface; `run` now plays the decks it computes.

- [ ] **Step 1: Write the failing test**

```python
class TestTheRoundPlaysTheComputedDecks:
    """The weights have to reach the table, which is what never happened.

    `play_round` received correctly-computed weights and discarded them, so the
    workers played ordinary sealed self-play: a slower duplicate of
    `match-outcomes` that reaches none of the cards coverage exists for.
    """

    def test_the_decks_played_are_built_from_the_weights(self, tmp_path):
        from effects.application import collect_coverage
        from effects.infrastructure import collector_connector

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        supervisor = MagicMock()
        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor), \
             patch.object(collect_coverage, "consult_castability", return_value={}):
            collect_coverage.run(
                collect_coverage.CollectCoverageConfig(
                    effect_records=tmp_path / "records",
                    cards_folders=(cards,),
                    target_records=1,
                    decks_per_round=4,
                )
            )

        decks_file = supervisor.play_round.call_args.args[0]
        rows = decks_file.read_text(encoding="utf-8").splitlines()
        assert rows, "no decks were written"
        assert all(row.split(";")[2].count("|") == 39 for row in rows)
        assert supervisor.play_round.call_args.kwargs["matches"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_coverage.py -q -k PlaysTheComputedDecks`
Expected: FAIL — `play_round` called with `(weights, cards_folder)`, not a decks file

- [ ] **Step 3: Write minimal implementation**

In `collect_coverage.py`, load the card texts once before the loop:

```python
    from effects.application.train_effect_model import load_card_files

    card_files = load_card_files(cards_folder)
    texts = {
        name: (cards_folder.parent / script).read_text(encoding="utf-8")
        for name, script in card_files.items()
        if (cards_folder.parent / script).exists()
    }
    decks_file = Path(config.effect_records) / "coverage-decks.txt"
    rng = random.Random(RANDOM_SEED)
```

and replace the `play_round` call inside the loop:

```python
            decks = build_coverage_decks(
                weights, texts, config.decks_per_round, rng=rng,
            )
            if not decks:
                logger.error(
                    "No card with a weight has converted text; nothing to deck."
                )
                break
            write_deck_file(
                decks, decks_file,
                label="coverage", set_code=COVERAGE_SET_CODE,
            )
            supervisor.play_round(decks_file, matches=config.decks_per_round)
```

Add `RANDOM_SEED = 42` beside the other module constants if absent, and import `build_coverage_decks`'s dependencies at module level.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/effects/application/test_coverage.py -q`
Expected: PASS

- [ ] **Step 5: Run the suite and lint**

Run: `python -m pytest tests/unit/effects -q && python -m ruff check src tests`
Expected: PASS, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/effects/application/collect_coverage.py tests/unit/effects/application/test_coverage.py
git commit -m "feat(effects): play the coverage decks the weights select"
```

---

### Task 6: Wire the variant round, and stage the scripts into Forge

**Files:**
- Modify: `src/effects/application/collect_variants.py:226-245`
- Modify: `src/effects/infrastructure/collector_connector.py`
- Modify: `src/sealed/infrastructure/match_worker_connector.py`
- Test: `tests/unit/effects/application/test_variants.py`

**Interfaces:**
- Consumes: everything from Tasks 2 and 4.
- Produces: `CollectorSupervisor(..., variant_scripts: Path | None = None)`; worker property `effect.variant.scripts`.

**Why the staging matters:** `ForgeEnvironmentInitializer.initialize(extraCardSource)` already stages a variant tree into Forge's custom-cards directory and registers each name with `VariantRegistry`, and `MatchWorkerMain` already reads `-Deffect.variant.scripts` to find it. **Nothing sets that property.** So the variant scripts are generated, sidecar'd, and never loaded — Forge has never seen one.

- [ ] **Step 1: Write the failing test**

```python
class TestTheVariantRoundPlaysTheVariants:
    def test_the_scripts_are_staged_and_decked(self, tmp_path):
        from effects.application import collect_variants
        from effects.infrastructure import collector_connector

        source = tmp_path / "cards"
        source.mkdir()
        (source / "bolt.txt").write_text(
            "Name:Bolt\nManaCost:R\nTypes:Instant\n"
            "A:SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any\n",
            encoding="utf-8",
        )
        records = tmp_path / "records"
        records.mkdir()
        (records / "seed.jsonl").write_text("{}\n" * 100, encoding="utf-8")

        supervisor = MagicMock()
        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor) as ctor, \
             patch.object(collect_variants, "VariantSidecarConnector") as sidecar:
            sidecar.return_value.run.return_value = 0
            collect_variants.run(
                collect_variants.CollectVariantsConfig(
                    effect_records=records,
                    forge_cards_path=source,
                    variant_scripts=tmp_path / "variants",
                    decks_per_round=3,
                )
            )

        assert ctor.call_args.kwargs["variant_scripts"] == tmp_path / "variants"
        decks_file = supervisor.play_round.call_args.args[0]
        assert decks_file.read_text(encoding="utf-8").strip(), "no variant decks"
        assert supervisor.play_round.call_args.kwargs["matches"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/effects/application/test_variants.py -q -k VariantRoundPlays`
Expected: FAIL — `CollectorSupervisor` got no `variant_scripts`, `play_round` called with weights

- [ ] **Step 3: Write minimal implementation**

`CollectorSupervisor.__init__` gains `variant_scripts: Path | None = None`, stored as `self._variant_scripts`, and `start_worker` passes it:

```python
            variant_scripts=self._variant_scripts,
```

`MatchWorkerConnector.start` gains `variant_scripts: Path | None = None` and:

```python
        if variant_scripts is not None:
            system_properties["effect.variant.scripts"] = str(variant_scripts)
```

In `collect_variants.py`, construct the supervisor with the tree and play the variant decks:

```python
    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records,
        caps=config.caps, variant_scripts=Path(config.variant_scripts),
    )
    decks_file = Path(config.effect_records) / "variant-decks.txt"
    texts = {v.name: _deck_text(v) for v in variants}
    decks = build_coverage_decks(
        {v.name: 1.0 for v in variants}, texts, config.decks_per_round,
        rng=random.Random(RANDOM_SEED),
    )
    write_deck_file(decks, decks_file, label="variant", set_code=COVERAGE_SET_CODE)
    try:
        supervisor.play_round(decks_file, matches=config.decks_per_round)
```

where `_deck_text` renders the minimum the manabase heuristic reads:

```python
def _deck_text(variant: GeneratedVariant) -> str:
    """The converted-shaped text `compute_basic_lands` needs for a variant.

    A variant has no converted prose by design (FR-056), and the manabase
    heuristic reads only the name, mana cost and type line, so those are
    rendered from the perturbed script rather than the tree.
    """
    lines = variant.path.read_text(encoding="utf-8").splitlines()
    cost = next((ln.split(":", 1)[1] for ln in lines if ln.startswith("ManaCost:")), "")
    types = next((ln.split(":", 1)[1] for ln in lines if ln.startswith("Types:")), "")
    return f"name: {variant.name}\nmana cost: {cost}\ntypes: {types}\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/effects/application/test_variants.py -q`
Expected: PASS

- [ ] **Step 5: Run both suites and lint**

Run: `python -m pytest tests/unit/effects tests/unit/sealed -q && python -m ruff check src tests`
Expected: PASS, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/effects/application/collect_variants.py src/effects/infrastructure/collector_connector.py src/sealed/infrastructure/match_worker_connector.py tests/unit/effects/application/test_variants.py
git commit -m "feat(effects): stage variant scripts into Forge and play them"
```

---

### Task 7: Prove it end to end, and correct the documents

**Files:**
- Modify: `tests/integration/test_effects_coverage_isolation.py`
- Modify: `specs/023-ability-effect-model/spec.md` (FR-052, FR-059)
- Modify: `src/effects/CLAUDE.md`
- Modify: `specs/023-ability-effect-model/tasks.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing integration test**

```python
def _deck_of(name: str, count: int = 23) -> list[str]:
    return [name] * count + ["Forest"] * 17


def test_a_named_deck_reaches_the_records(tmp_path: Path) -> None:
    """The whole point of the collector, against a real Forge.

    A coverage deck names the cards it is built to reach. If the worker plays
    that deck, those names appear in effect records; if it falls back to sealed
    self-play — which is what it did before this plan — they do not, and the
    run looks identical while collecting something else entirely.

    ``Llanowar Elves`` because it is in the converted tree, castable by the AI,
    and does nothing that needs another card present.
    """
    _require_jar()
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    decks_file = tmp_path / "decks.txt"
    write_deck_file(
        [_deck_of("Llanowar Elves")] * 2, decks_file,
        label="coverage", set_code=COVERAGE_SET_CODE,
    )

    process = MatchWorkerConnector().start(
        tmp_path / "progress.txt",
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
        side_a_decks_path=decks_file,
        side_b_decks_path=decks_file,
        decks_only=True,
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        names: set[str] = set()
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            names = {
                entity.name
                for record in read_records(records_dir)
                for entity in record.state.entities
            }
            if "Llanowar Elves" in names:
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    assert "Llanowar Elves" in names, (
        "the decked card never reached a record: the worker is not playing the "
        f"decks file. Saw {len(names)} distinct names."
    )
```

Mark it `@pytest.mark.integration` to match the rest of the file, and import
`write_deck_file` and `COVERAGE_SET_CODE` from `effects.infrastructure.deck_file`.

- [ ] **Step 2: Run it to verify it fails against the current code**

Run: `python -m pytest tests/integration/test_effects_coverage_isolation.py -q -m integration`
Expected: FAIL — the card appears in no record

- [ ] **Step 3: Confirm it passes on the new code**

Run the same command after Tasks 1–6 are merged.
Expected: PASS

- [ ] **Step 4: Amend FR-052 and FR-059 for the progress file**

Both currently say coverage and variant matches write effect records only. Add to each: *"A per-run progress file under `--effect-records` is not a sealed corpus and is permitted; it is what bounds a round, and nothing downstream reads it."*

Load the **feature-workflow** skill before editing anything under `specs/` or `experiments/`.

- [ ] **Step 5: Correct `src/effects/CLAUDE.md`**

The `collect-coverage` bullet describes deck building that did not exist. It is now true — verify each clause against the code and fix any that still are not.

- [ ] **Step 6: Re-open the tasks this plan completes**

`T119` and `T120` are marked `[X]` for work that was never done. Mark them `[ ]` with a note naming this plan, rather than silently leaving them checked.

- [ ] **Step 7: Full suite, both languages**

Run: `python -m pytest tests/unit -q && cd forge-connector && mvn -o test`
Expected: 2860+ Python pass, `BUILD SUCCESS` for Java

- [ ] **Step 8: Commit**

```bash
git add tests/integration specs src/effects/CLAUDE.md
git commit -m "test(effects): a coverage round reaches a card sealed play cannot"
```

---

## Notes for the executor

**The round recount is O(corpus) per round.** `collect_coverage.run` recounts by re-reading every shard under `--effect-records` after each round. At 1,502 shards that measured ~66 minutes per round, against ~35 minutes of play. This plan does not fix it; the loop already tracks `previous`, so an incremental recount over the shards written since the round began is the natural follow-up. Do not fold it into these tasks.

**`--decks-per-round` counts matches, not games.** Each match is `COVERAGE_BEST_OF` games. The name is the existing contract and this plan keeps it.

**The NPE in Forge's `EffectAi`** (`getValidCardsToTarget` with null target restrictions) fires on some cards and is caught inside Forge's own executor. It is noise, not a failure, and coverage decks will hit it far more often than sealed play does — an unusual-card path that sealed pools rarely reach. Do not add error handling for it; do not let its volume in the logs read as a broken run.
