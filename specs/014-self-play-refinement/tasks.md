# Tasks: Self-Play Refinement

**Input**: Design documents from `/specs/014-self-play-refinement/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: New shared module that multiple user stories depend on (eligible set enumeration).

- [ ] T001 [P] Create `eligible_sealed_sets()` function in `src/sealed/infrastructure/eligible_sets.py` — parse `AllPrintings.json` to return set codes matching `hasDraftBoosterTemplate && type != "funny"`, mirroring `MatchGenerator.computeEligibleSets()` (Java). Prior art: `src/sealed/infrastructure/pool_connector.py` for infrastructure-layer conventions.
- [ ] T002 [P] Create unit tests for `eligible_sealed_sets()` in `tests/unit/sealed/infrastructure/test_eligible_sets.py` — test with a trimmed `AllPrintings.json` fixture containing eligible sets, un-sets, and sets without draft templates. Prior art: `tests/unit/sealed/infrastructure/test_pool_connector.py`.

**Checkpoint**: Eligible set enumeration available for US1 (evaluate-scorer) and US4 (generate-pools random selection).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Pool file format change (`SET_CODE;Card1|Card2|...`) that ALL downstream stories depend on. Must be completed before any user story.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Modify `PoolMain.java` at `forge-connector/src/main/java/com/pricepredictor/connector/PoolMain.java` to prepend `setCode + ";"` to each pool line. Change the output format from `Card1|Card2|...|CardN` to `SET_CODE;Card1|Card2|...|CardN`. No backward compatibility needed (research.md Decision 7).
- [ ] T004 Update `_parse_pools()` in `src/sealed/application/evaluate_scorer.py:182` to handle the new `SET_CODE;Card1|Card2|...|CardN` format — split on first `";"` to extract set code, then split remainder on `"|"` for card names. Return a list of `(set_code, list[str])` tuples (or just the card names if the set code is not needed by the caller). Prior art: the existing `_parse_pools()` at line 182.
- [ ] T005 [P] Update pool-format-dependent tests: `tests/unit/sealed/application/test_evaluate_scorer.py` and `tests/unit/sealed/application/test_generate_pools.py` — update any pool file fixtures to use the new `SET_CODE;Card1|Card2|...` format. Update assertions that parse pool files.
- [ ] T006 [P] Update `tests/unit/sealed/infrastructure/test_pool_connector.py` — verify the `--set` argument is still passed correctly to `PoolMain` and adjust any mock assertions for the new format.
- [ ] T007 Rebuild the forge-connector JAR (`cd forge-connector && mvn package -DskipTests`) and verify that `PoolMain` with `--set RVR` produces lines prefixed with `RVR;`.

**Checkpoint**: Pool file format is `SET_CODE;Card1|Card2|...` everywhere. All existing tests pass with the new format.

---

## Phase 3: User Story 1 — Random-Set Pool Generation (Priority: P1)

**Goal**: `generate-pools` without `--set` generates pools from randomly selected sealed-legal sets, each line prefixed with its set code.

**Independent Test**: Run `generate-pools` without `--set` and verify output contains pools from multiple sets, each line prefixed with its set code.

### Tests for User Story 1

- [ ] T008 [P] [US1] Add unit tests in `tests/unit/sealed/application/test_generate_pools.py` for the `set_code=None` case — verify `PoolConnector.generate()` is called without `--set`, and the default output path is `output/sealed/pools/pools.txt` (not `output/sealed/pools/{set}/pools.txt`).
- [ ] T009 [P] [US1] Add unit tests in `tests/unit/sealed/infrastructure/test_pool_connector.py` for `set_code=None` — verify the `--set` argument is omitted from the Java command when `set_code` is `None`.

### Implementation for User Story 1

- [ ] T010 [US1] Modify `PoolMain.java` at `forge-connector/src/main/java/com/pricepredictor/connector/PoolMain.java` to support omitting `--set`: when `--set` is not provided, select a random eligible set per pool using `MatchGenerator.computeEligibleSets()`. Each pool line must still be prefixed with its (randomly chosen) set code.
- [ ] T011 [US1] Modify `PoolConnector.generate()` in `src/sealed/infrastructure/pool_connector.py` to accept `set_code: str | None`. When `None`, omit the `--set` argument from the Java command. Update the method signature and docstring.
- [ ] T012 [US1] Modify `GeneratePoolsUseCase.execute()` in `src/sealed/application/generate_pools.py` to accept `set_code: str | None` and pass it through to `PoolConnector.generate()`.
- [ ] T013 [US1] Modify the `generate-pools` CLI parser in `src/sealed/infrastructure/cli.py:66-89` — change `--set` default from `"RVR"` to `None`, update the `--pools-path` default logic (when `--set` is `None`, default to `output/sealed/pools/`; when `--set` is given, keep `output/sealed/pools/{set}/`), and update help text to document both defaults per FR-001.
- [ ] T014 [US1] Rebuild forge-connector JAR and verify random-set pool generation produces pools from multiple distinct sets.

**Checkpoint**: `generate-pools` without `--set` produces multi-set pools. With `--set MH3`, all pools are MH3. All tests pass.

---

## Phase 4: User Story 2 — Build Scorer Decks from Pools (Priority: P2)

**Goal**: New `build-decks` subcommand reads a pools file, builds scorer-guided 40-card decks, and writes a generated-decks file.

**Independent Test**: Run `build-decks` with a trained scorer checkpoint and a pools file, verify output has one 40-card deck per pool with correct set code prefixes.

### Tests for User Story 2

- [ ] T015 [P] [US2] Create unit tests in `tests/unit/sealed/application/test_build_decks.py` — test `BuildDecksUseCase.execute()` with mocked scorer model and `ConvertedCardLocator`. Verify: output file has one line per pool, each line has `SET_CODE;` prefix, each deck has exactly 40 pipe-separated card names. Test edge case: pool with fewer than 23 embeddable cards is skipped. Prior art: `tests/unit/sealed/application/test_evaluate_scorer.py`.

### Implementation for User Story 2

- [ ] T016 [US2] Create `BuildDecksConfig` dataclass and `BuildDecksUseCase` in `src/sealed/application/build_decks.py`. The use case loads the scorer checkpoint, iterates over pools from the input file (using the updated `_parse_pools()`), builds a deck per pool using `GreedyDeckBuilder` + `compute_basic_lands()` (same pattern as `_build_a_decks()` in `evaluate_scorer.py:191`), and writes the generated-decks file in `SET_CODE;Card1|Card2|...|Card40` format. Prior art: `src/sealed/application/evaluate_scorer.py:191-221`.
- [ ] T017 [US2] Add the `build-decks` subcommand to `src/sealed/infrastructure/cli.py` — add `_build_build_decks_parser(subparsers)` and `run_build_decks(args)` functions. Arguments: `--pools-path` (required), `--checkpoint` (default `models/sealed/scorer/latest.pt`), `--cards-path` (default `output/cardsfolder/`), `--output` (default `output/sealed/generated-decks.txt`). Register in `build_parser()`. Prior art: `_build_evaluate_scorer_parser()` at `cli.py:162`.
- [ ] T018 [US2] Add shared fixtures for generated-decks files in `tests/unit/sealed/conftest.py` — a pytest fixture that creates a temporary generated-decks file with known set codes and 40-card decks (will be needed by US3 tests).

**Checkpoint**: `build-decks` reads a pools file, builds scorer decks, and writes `generated-decks.txt`. All tests pass.

---

## Phase 5: User Story 3 — Self-Play Match Generation (Priority: P3)

**Goal**: `match-outcomes --generated-decks-path <file>` runs self-play matches — each match pits a scorer-built deck (from file) against an opponent built by one of 5 weighted methods, enforcing same-set pairing.

**Independent Test**: Run `match-outcomes --generated-decks-path <file>` and verify new match outcome lines are appended to `match-outcomes.txt`.

### Tests for User Story 3

- [ ] T019 [P] [US3] Create Java unit tests in `forge-connector/src/test/java/com/pricepredictor/connector/GeneratedDecksIndexTest.java` — test parsing of generated-decks file, `randomDeck()`, `randomDeckFromSet()`, and the exclude-self behavior (method 5 re-roll). Prior art: `forge-connector/src/test/java/com/pricepredictor/connector/DeckBuilderTest.java`.
- [ ] T020 [P] [US3] Create Java unit tests in `forge-connector/src/test/java/com/pricepredictor/connector/SelfPlayMatchGeneratorTest.java` — test method selection weights (4:3:2:1:4), same-set constraint for methods 1-4, method 5 picks from file with same set code. Prior art: `forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorTest.java`.
- [ ] T021 [P] [US3] Add Python unit tests in `tests/unit/sealed/infrastructure/test_match_worker_connector.py` for the `generated_decks_path` parameter — verify `-Dgenerated.decks.file=<path>` is added to the JVM command when the path is provided, and absent when `None`.
- [ ] T022 [P] [US3] Add Python unit tests in `tests/unit/sealed/application/test_match_outcomes.py` for `generated_decks_path` — verify `MatchOutcomeSupervisor` passes `generated_decks_path` through to `MatchWorkerConnector.start()`.

### Implementation for User Story 3

- [ ] T023 [US3] Create `GeneratedDecksIndex` Java class in `forge-connector/src/main/java/com/pricepredictor/connector/GeneratedDecksIndex.java` — parse a generated-decks file into an in-memory index. Fields: `allDecks` (list of parsed lines), `decksBySet` (map from set code to list of decks). Methods: `randomDeck(Random)`, `randomDeckFromSet(String setCode, GeneratedDeck exclude, Random)`. Use a `GeneratedDeck` record (inner or separate) with `setCode` and `cardNames` fields. Prior art: file format from `data-model.md`.
- [ ] T024 [US3] Create `SelfPlayMatchGenerator` Java class in `forge-connector/src/main/java/com/pricepredictor/connector/SelfPlayMatchGenerator.java` — generates one self-play match per call. Constructor takes `GeneratedDecksIndex`, `DeckBuilder`, `PoolGenerator`, `GamePlayer`, `List<String> eligibleSets`. Method `generateMatch()`: pick random deck A from index, roll for method 1-5 with weights 4:3:2:1:4, build deck B accordingly (methods 1-4: generate pool from deck A's set, build via DeckBuilder; method 5: pick from index with same set, exclude deck A), play best-of-3, return `MatchResult`. Prior art: `MatchGenerator.java:74-88`.
- [ ] T025 [US3] Modify `MatchWorkerMain.java` at `forge-connector/src/main/java/com/pricepredictor/connector/MatchWorkerMain.java` — read optional `-Dgenerated.decks.file` system property. When set, load the file into `GeneratedDecksIndex`, create `SelfPlayMatchGenerator`, and use it in the match loop. When absent, use existing `MatchGenerator` (Phase 0 behavior unchanged). Prior art: existing `MatchWorkerMain.java` structure.
- [ ] T026 [US3] Modify `MatchWorkerConnector.start()` in `src/sealed/infrastructure/match_worker_connector.py` — add optional `generated_decks_path: Path | None = None` parameter. When not `None`, add `"generated.decks.file": str(generated_decks_path)` to the `system_properties` dict passed to `build_jvm_command()`.
- [ ] T027 [US3] Modify `MatchOutcomeSupervisor.__init__()` in `src/sealed/application/match_outcomes.py` — add optional `generated_decks_path: Path | None = None` parameter, store it, and forward it to `self._connector.start()` in `_start_worker()`.
- [ ] T028 [US3] Modify the `match-outcomes` CLI parser in `src/sealed/infrastructure/cli.py:193-204` — add `--generated-decks-path` argument (default `None`). Pass it to `MatchOutcomeSupervisor` in `run_match_outcomes()`. Update help text.
- [ ] T029 [US3] Rebuild forge-connector JAR and verify self-play match generation appends valid match outcome lines when `--generated-decks-path` is provided.

**Checkpoint**: `match-outcomes --generated-decks-path <file>` generates self-play matches. Without the flag, Phase 0 behavior is unchanged. All tests pass.

---

## Phase 6: User Story 4 — Random-Set Evaluation (Priority: P4)

**Goal**: `evaluate-scorer` uses a randomly selected sealed-legal set by default instead of hardcoded RVR.

**Independent Test**: Run `evaluate-scorer` without `--set` and verify pools come from a randomly selected set. Run with `--set RVR` and verify all pools are RVR.

### Tests for User Story 4

- [ ] T030 [P] [US4] Add unit tests in `tests/unit/sealed/application/test_evaluate_scorer.py` for random set selection — test that when `set_code` is `None`, `eligible_sealed_sets()` is called and a random set is passed to `PoolConnector.generate()`. Test that when `set_code` is given, it is used directly.

### Implementation for User Story 4

- [ ] T031 [US4] Add `set_code: str | None = None` field to `EvaluateScorerConfig` in `src/sealed/application/evaluate_scorer.py:27`. Modify `_generate_pools()` at line 134 to use `config.set_code` when provided, or call `eligible_sealed_sets()` and `random.choice()` when `None`. Import `eligible_sealed_sets` from `src/sealed/infrastructure/eligible_sets.py` (created in T001).
- [ ] T032 [US4] Add `--set` argument to `_build_evaluate_scorer_parser()` in `src/sealed/infrastructure/cli.py:162-191` — optional, default `None`. Pass it to `EvaluateScorerConfig` as `set_code`. Update help text per FR-012/FR-013.

**Checkpoint**: `evaluate-scorer` without `--set` picks a random sealed-legal set. With `--set BLB`, all pools are BLB. All tests pass.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup.

- [ ] T033 Run full Python test suite (`pytest`) and verify all tests pass with no regressions.
- [ ] T034 Run full Java test suite (`cd forge-connector && mvn test`) and verify all tests pass.
- [ ] T035 Run Python linting (`ruff check src/ tests/`) and fix any issues.
- [ ] T036 Validate quickstart workflow end-to-end: `generate-pools` (no `--set`) -> `build-decks` -> `match-outcomes --generated-decks-path` -> verify match-outcomes.txt has valid lines.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Independent of Phase 1 — can start in parallel with Setup
- **US1 (Phase 3)**: Depends on Phase 2 (pool format change) being complete
- **US2 (Phase 4)**: Depends on Phase 2 (pool format change for `_parse_pools()`)
- **US3 (Phase 5)**: Depends on Phase 2 (pool format). Java implementation (T023-T025) can start after Phase 2. Python wiring (T026-T028) is independent of US1/US2.
- **US4 (Phase 6)**: Depends on Phase 1 (`eligible_sealed_sets()`) and Phase 2 (pool format)
- **Polish (Phase 7)**: Depends on all story phases being complete

### User Story Dependencies

- **US1 (P1)**: Depends on Phase 2 only. No dependencies on other stories.
- **US2 (P2)**: Depends on Phase 2 only. No dependencies on other stories.
- **US3 (P3)**: Depends on Phase 2 only. No dependencies on US1/US2 for implementation, but end-to-end usage requires US1 (pool generation) and US2 (build-decks) to produce input files.
- **US4 (P4)**: Depends on Phase 1 (`eligible_sealed_sets`). No dependencies on other stories.

### Within Each User Story

- Tests written first (when marked [P], they can run in parallel)
- Infrastructure/domain changes before application layer
- Application layer before CLI wiring
- JAR rebuild and manual verification last

### Parallel Opportunities

- Phase 1 (T001-T002) can run in parallel with Phase 2 (T003-T007)
- Within Phase 2: T005 and T006 can run in parallel (different test files)
- After Phase 2: US1, US2, US3, US4 can all start in parallel
  - US1: T008-T009 tests in parallel, then T010-T014 sequentially
  - US2: T015 test, then T016-T018 sequentially
  - US3: T019-T022 tests in parallel, then T023-T029 sequentially (Java before Python)
  - US4: T030 test, then T031-T032 sequentially
- Phase 7: T033-T035 can run in parallel

---

## Parallel Example: After Phase 2 Completes

```
# All four user stories can start simultaneously:

Stream A (US1): T008 → T009 → T010 → T011 → T012 → T013 → T14
Stream B (US2): T015 → T016 → T017 → T018
Stream C (US3): T019 + T020 + T021 + T022 → T023 → T024 → T025 → T026 → T027 → T028 → T029
Stream D (US4): T030 → T031 → T032
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (eligible sets)
2. Complete Phase 2: Foundational (pool format change)
3. Complete Phase 3: US1 (random-set pool generation)
4. **STOP and VALIDATE**: Generate pools without `--set`, verify multi-set output

### Incremental Delivery

1. Setup + Foundational -> Pool format updated everywhere
2. Add US1 -> Random-set pools work -> Validate independently
3. Add US2 -> `build-decks` produces generated-decks file -> Validate independently
4. Add US3 -> Self-play match generation works end-to-end -> Validate with quickstart workflow
5. Add US4 -> Evaluation uses random sets -> Validate independently
6. Polish -> Full test suite green, linting clean, quickstart validated

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- **Principle VII (Codebase-Aware Planning)**: Every new class/function references its nearest prior art in the task description
