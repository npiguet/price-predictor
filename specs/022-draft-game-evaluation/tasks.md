---
description: "Task list for 022-draft-game-evaluation"
---

# Tasks: Draft agent game-played evaluation

**Input**: Design documents from `/specs/022-draft-game-evaluation/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), test tasks are
mandatory and are listed before the implementation they cover.

**Organization**: Tasks are grouped by user story so each story can be implemented, tested
and delivered independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel — different files, no dependency on an incomplete task
- **[Story]**: The user story the task serves (US1–US4)

## Path Conventions

Python lives under `src/draft/`, `src/sealed/`, `src/price_predictor/` in the three-layer
`domain` → `application` → `infrastructure` split. Java lives under
`forge-connector/src/main/java/com/pricepredictor/connector/`. Python tests mirror the
source tree under `tests/unit/`; Java tests sit beside their peers in
`forge-connector/src/test/java/com/pricepredictor/connector/`.

---

## Phase 1: Setup

**Purpose**: Give the feature's flags and JVM entry point a home before any behaviour lands.

- [X] T001 [P] Add a `play-draft-games` subparser to `src/draft/infrastructure/cli.py` carrying every flag in `specs/022-draft-game-evaluation/contracts/cli.md`, dispatching to a `run_play_draft_games` stub that raises `NotImplementedError`
- [X] T002 [P] Add `DraftGameWorkerMain.java` to `forge-connector/src/main/java/com/pricepredictor/connector/` with system-property parsing only — `seats.file`, `output.file`, `run.id`, `best.of`, `include.mirrors` — exiting 2 when a required property is missing and defaulting `best.of` to 3 with a warning, as `ValidationWorkerMain` does

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extract the Forge worker supervisor that both this feature and the sealed
match-generation command will use. Required by the constitution's third-instance rule
(see [plan.md](plan.md) Codebase Survey) and by the spec's requirement that this command's
progress output stay identical to `match-outcomes`.

**⚠️ This phase modifies a working sealed command. Complete and verify it before Phase 3.**

- [X] T003 Extract a `ForgeWorkerPool` class into `src/price_predictor/infrastructure/forge_jvm.py`, taking a worker-command factory, a worker count, a status-line callback and a recycle interval, and providing: monitor-thread-per-worker with restart on exit, the 60-second status line, oldest-worker recycling, SIGINT/SIGTERM shutdown, and output-line counting — lifted from `MatchOutcomeSupervisor` in `src/sealed/application/match_outcomes.py`
- [X] T004 Migrate `MatchOutcomeSupervisor` in `src/sealed/application/match_outcomes.py` onto `ForgeWorkerPool`, leaving every line it prints byte-identical
- [X] T005 [P] Add unit tests for `ForgeWorkerPool` status-line formatting, line counting, oldest-worker selection and the stop condition in `tests/unit/infrastructure/test_forge_worker_pool.py` (the repo puts price_predictor tests directly under `tests/unit/`)
- [X] T006 [P] Add a regression test in `tests/unit/sealed/application/test_match_outcomes.py` asserting the migrated supervisor's status line matches the format `[{elapsed}s] {n} matches completed | {rate} matches/min | {alive}/{workers} workers alive`

**Checkpoint**: `python -m sealed match-outcomes` behaves and prints exactly as before.

---

## Phase 3: User Story 1 — Measure an agent by games won (P1) 🎯 MVP

**Goal**: Sample same-pod pairs, play best-of-N matches, and append rows that
`scripts/analyze_winrates.py` reads unchanged.

**Independent test**: Run against a corpus holding two or more agent labels with a small
`--n-pairings`, then run the tally over the output and read per-agent win rates.

### Tests

- [X] T007 [P] [US1] Seat-table line tests in `tests/unit/draft/domain/test_seat_table.py`: a record projects to one line per seat; fields are `draft_id;set_code;label;kind;cards`; card names containing commas and apostrophes survive a round trip; `kind` is `deck` for every seat; a corpus ending in a trailing partial line projects the complete records and ignores the partial one (FR-016)
- [X] T008 [P] [US1] `SeatTableIndexTest.java` in `forge-connector/src/test/java/com/pricepredictor/connector/`: parses lines, groups seats by `draft_id`, draws two distinct seats of one pod, rejects mirror pairs when `include.mirrors` is false, and skips a pod whose seats all share a label rather than looping
- [X] T009 [P] [US1] `DraftGamePlayerTest.java` in `forge-connector/src/test/java/com/pricepredictor/connector/`: given a played match, builds a `MatchResult` carrying the two seat labels in `methodA`/`methodB`, the pod's set code, and the decks as played

### Implementation

- [X] T010 [US1] Implement the corpus projection in `src/draft/domain/seat_table.py`: a pure function from a draft record to seat rows, plus line formatting, streaming one record at a time
- [X] T011 [US1] Implement `SeatTableIndex.java` in `forge-connector/src/main/java/com/pricepredictor/connector/`: load the table, group by `draft_id`, and expose a random non-mirror pair draw, mirroring `GeneratedDecksIndex`'s shape
- [X] T012 [US1] Implement `DraftGamePlayer.java` in `forge-connector/src/main/java/com/pricepredictor/connector/`: resolve a pair's cards through `FModel.getMagicDb()`, play via `GamePlayer`, build a `MatchResult`, and append it with `MatchResultWriter`
- [X] T013 [US1] Complete `DraftGameWorkerMain.java` in `forge-connector/src/main/java/com/pricepredictor/connector/`: initialise Forge, load the index, and loop drawing and playing until killed, logging and continuing on a per-pairing exception without writing a row
- [X] T014 [US1] Implement `src/draft/infrastructure/draft_game_connector.py`: build the worker command via `build_jvm_command` and `build_forge_classpath` with `-Xmx1200m` and the five system properties from `contracts/seat-table.md`
- [X] T015 [US1] Implement `src/draft/application/play_draft_games.py`: write the seat table to a file in the system temporary directory, drive `ForgeWorkerPool`, stop once the output file has gained `--n-pairings` rows since the run began, print the exit summary, and delete the seat table on exit including on interrupt
- [X] T016 [US1] Replace the `run_play_draft_games` stub in `src/draft/infrastructure/cli.py` with the real wiring, including the startup validation and startup echo from `contracts/cli.md` and exit codes 0/2/130
- [X] T017 [P] [US1] Supervisor tests in `tests/unit/draft/application/test_play_draft_games.py`: the stopping condition counts only rows added since the run began and ignores pre-existing rows; the summary reports matches played, elapsed and the output path; the status line and worker-lifecycle lines match the shapes required by FR-023 and FR-024; the seat table is deleted on exit

**Checkpoint**: `python -m draft play-draft-games --n-pairings N` produces rows that
`python scripts/analyze_winrates.py` tallies by agent label.

---

## Phase 4: User Story 2 — Evaluate one corpus among many (P2)

**Goal**: Restrict sampling to chosen run identifiers so a shared corpus can be evaluated
one generation at a time.

**Independent test**: Run against a corpus holding two run identifiers, restricted to one,
and confirm no output row's decks come from the other.

### Tests

- [X] T018 [P] [US2] Projection filter tests in `tests/unit/draft/domain/test_seat_table.py`: a single `--run-id` emits seats only from matching records; several are unioned; none emits every record

### Implementation

- [X] T019 [US2] Add repeatable `--run-id` filtering to the projection in `src/draft/domain/seat_table.py` and thread it through `src/draft/application/play_draft_games.py`
- [X] T020 [US2] Extend the startup echo in `src/draft/infrastructure/cli.py` to name the run identifiers in scope, or state that the whole corpus is, and to fail validation with exit 2 when no pair survives the filters

**Checkpoint**: Rows trace only to records carrying a selected run identifier.

---

## Phase 5: User Story 3 — Accumulate matches over a long session (P3)

**Goal**: Run without a pairing target until interrupted, and top up an existing output file
on a later invocation.

**Independent test**: Start with no `--n-pairings`, interrupt after some matches, confirm the
output parses completely, then re-invoke and confirm rows are appended rather than replaced.

### Tests

- [X] T021 [P] [US3] Stopping-condition tests in `tests/unit/draft/application/test_play_draft_games.py`: an absent `--n-pairings` never satisfies the stop check; an interrupt before the first recorded match yields exit 130 and one after yields exit 0

### Implementation

- [X] T022 [US3] Make `--n-pairings` optional in `src/draft/application/play_draft_games.py` and `src/draft/infrastructure/cli.py`, running until interrupted when absent, and ensure the exit summary prints on the interrupt path

**Checkpoint**: An unbounded run stops cleanly on Ctrl-C with a complete output file.

---

## Phase 6: User Story 4 — Compare Forge's builder against the recorded one (P3)

**Goal**: Divert a share of `forge-full` seats to Forge's own sealed deck builder, working
from their drafted pools and reporting under the label `forge-native`.

**Independent test**: Run with the fraction at a half and confirm the tally reports the
recorded and rebuilt variants as two separate rows with comparable sample sizes.

### Tests

- [X] T023 [P] [US4] Diversion tests in `tests/unit/draft/domain/test_seat_table.py`: only `forge-full` seats are eligible; the diverted share approaches the fraction over many seats; a diverted seat emits `kind = pool` with the label `forge-native` and its drafted pool; a fraction of 0 diverts nothing and emits no pool
- [X] T024 [P] [US4] Pool-side tests in `DraftGamePlayerTest.java`: a `pool` side is built through `DeckBuilder.buildStandard` rather than played as given, and the resulting `MatchResult` records the built deck rather than the pool

### Implementation

- [X] T025 [US4] Add diversion to `src/draft/domain/seat_table.py`: select `forge-full` seats with the given probability, emit their pool from `DraftGeometry.drafted_pool` with `kind = pool`, and rewrite their label to `forge-native`
- [X] T026 [US4] Add the `pool` branch to `DraftGamePlayer.java` in `forge-connector/src/main/java/com/pricepredictor/connector/`, calling `DeckBuilder.buildStandard` and recording the built deck on the row
- [X] T027 [US4] Add `--forge-native-fraction` to `src/draft/infrastructure/cli.py` with range validation on [0, 1] and a startup-echo line reporting how many seats were diverted out of how many

**Checkpoint**: With the fraction between 0 and 1, `forge-full` and `forge-native` appear as
separate rows in the tally and meet head-to-head in the pairwise matrix.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T028 [P] Document the `play-draft-games` subcommand in `src/draft/CLAUDE.md`, including its flags, the `forge-native` label, and that reporting is `scripts/analyze_winrates.py`
- [X] T029 [P] Add the `draft-games.txt` output to the corpus-file-formats section of the root `CLAUDE.md`, noting it reuses the sealed match-outcome format with agent labels as method tags
- [X] T030 [P] Add the evaluation workflow to `README.md` per constitution Principle VI
- [X] T031 [P] Correct the `_kill_oldest_worker` docstring, which attributed recycling to JVM slowdown rather than to workers hanging in near-infinite loops — done as part of T003, the method having moved to `forge_jvm.py`
- [X] T032 [P] Correct the root `CLAUDE.md` description of `forge-connector` as "zero-dependency (stdlib-only)" — it declares `minlog` and ships a `jar-with-dependencies` assembly
- [X] T033 Run [quickstart.md](quickstart.md) end to end against `output/draft/yardstick-drafts.jsonl` and confirm `scripts/analyze_winrates.py` reads the output with no modification

---

## Dependencies

**Phase order**: Setup → Foundational → US1 → (US2, US3, US4 in any order) → Polish.

**Blocking**:

- T003 blocks T004, T005, T015.
- T004 blocks the Phase 2 checkpoint; nothing in Phase 3 depends on it, but it must land
  before the sealed command is trusted again.
- T010 blocks T017, T019, T025.
- T011, T012 block T013.
- T013, T014 block T015; T015 blocks T016.
- T012 blocks T026.

**Story independence**: US2, US3 and US4 each touch US1's files but not each other's
behaviour, so they may be implemented in any order once US1 is complete. US4 is the only one
that touches Java.

## Parallel execution examples

Phase 2 tests, once T003 and T004 are in:

```text
T005  tests/unit/price_predictor/infrastructure/test_forge_jvm.py
T006  tests/unit/sealed/application/test_match_outcomes.py
```

US1 tests, all before their implementation and in different files:

```text
T007  tests/unit/draft/domain/test_seat_table.py
T008  forge-connector/.../SeatTableIndexTest.java
T009  forge-connector/.../DraftGamePlayerTest.java
```

Polish, all independent files:

```text
T028  src/draft/CLAUDE.md
T029  CLAUDE.md
T030  README.md
T031  src/sealed/application/match_outcomes.py
T032  CLAUDE.md          # sequence after T029, same file
```

## Implementation strategy

**MVP**: Phases 1–3. That delivers the measurement the feature exists for — same-pod matches
tallied by agent label — and every later story is an option on top of it.

**Incremental delivery**: US2 next if the shared corpus already mixes generations, since
without it an evaluation cannot be scoped. US4 next if the question is how much of the Forge
reference's result comes from drafting versus building. US3 last; it is a convenience over
repeated bounded runs.

**Risk note**: Phase 2 is the only work that changes a command already in use. Keep T004 in
its own commit so it can be reverted without losing the extraction in T003.
