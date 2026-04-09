# Tasks: Sealed Training Data Generation

**Input**: Design documents from `/specs/012-sealed-training-data/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add required Forge dependencies for game playing and deck building

- [X] T001 Add forge-gui and forge-ai as system-scope dependencies in forge-connector/pom.xml, and include forge-gui/target/dependency/* on the runtime classpath (per research R2)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core Java infrastructure that MUST be complete before ANY user story can be implemented. These classes are shared across all match generation components.

- [X] T002 [P] Create GuiHeadless.java (IGuiBase stub for headless Forge operation, adapted from jumpstart-tierlist) in forge-connector/src/main/java/com/pricepredictor/connector/GuiHeadless.java
- [X] T003 [P] Create MatchResult.java (value object: deckA card list, deckB card list, winsA, winsB) in forge-connector/src/main/java/com/pricepredictor/connector/MatchResult.java
- [X] T004 Rewrite ForgeEnvironmentInitializer.java to use FModel.initialize() + GuiHeadless instead of manual StaticData setup (per research R1) in forge-connector/src/main/java/com/pricepredictor/connector/ForgeEnvironmentInitializer.java
- [X] T005 [P] Create MatchResultWriter.java (open file in APPEND mode, format MatchResult as semicolon-separated line with pipe-separated card names, validate wins sum to 2 or 3) in forge-connector/src/main/java/com/pricepredictor/connector/MatchResultWriter.java
- [X] T006 Unit test for MatchResultWriter (format validation, wins sum constraint, pipe/semicolon encoding) in forge-connector/src/test/java/com/pricepredictor/connector/MatchResultWriterTest.java

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Generate Match Outcome Data (Priority: P1) MVP

**Goal**: Complete end-to-end pipeline: Python supervisor spawns Java workers that generate sealed match outcomes and append results to a shared flat file. Workers auto-restart on crash, supervisor handles clean shutdown.

**Independent Test**: Run `python -m sealed match-outcomes --workers 2`, let it produce a few dozen outcomes, verify `./output/sealed/match-outcomes.txt` contains well-formed records with valid card names and plausible win counts.

### Tests for User Story 1 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T007 [P] [US1] Unit test for MatchGenerator (mock DeckBuilder, GamePlayer, PoolGenerator; verify match flow: set selection, pool generation, deck building, game playing, result assembly) in forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorTest.java
- [X] T008 [P] [US1] Unit test for supervisor logic (mock subprocess: verify spawn count, crash detection and restart, shutdown signal handling, status reporting line counting) in tests/unit/sealed/application/test_match_outcomes.py
- [X] T009 [P] [US1] Unit test for match worker connector (verify Java subprocess command construction: classpath, main class, -Xmx1200m flag) in tests/unit/sealed/infrastructure/test_match_worker_connector.py

### Implementation for User Story 1

- [X] T010 [P] [US1] Implement GamePlayer (create RegisteredPlayer + LobbyPlayerAi for each deck, GameRules with GameType.Sealed and gamesPerMatch=3, loop createGame/startGame until match.isMatchOver, return [winsA, winsB]) in forge-connector/src/main/java/com/pricepredictor/connector/GamePlayer.java
- [X] T011 [P] [US1] Implement DeckBuilder with method 1 only (standard SealedDeckBuilder: new SealedDeckBuilder(pool).buildDeck()) in forge-connector/src/main/java/com/pricepredictor/connector/DeckBuilder.java
- [X] T012 [US1] Implement MatchGenerator (pick random set from all sets with boosters, generate 2 pools of 6 boosters each via PoolGenerator, build deck from each pool via DeckBuilder, play best-of-3 via GamePlayer, return MatchResult) in forge-connector/src/main/java/com/pricepredictor/connector/MatchGenerator.java
- [X] T013 [US1] Implement MatchWorkerMain (initialize Forge via ForgeEnvironmentInitializer, create MatchGenerator + MatchResultWriter with output file path from system property, loop forever: generateMatch + write result) in forge-connector/src/main/java/com/pricepredictor/connector/MatchWorkerMain.java
- [X] T014 [P] [US1] Implement MatchWorkerConnector (build Java subprocess command with classpath including all 5 JARs + dependency dir, -Xmx1200m, pass output file path as system property, return Popen handle) in src/sealed/infrastructure/match_worker_connector.py
- [X] T015 [US1] Implement MatchOutcomeSupervisor (spawn worker_count monitor threads, each thread loops: start worker via MatchWorkerConnector, waitFor, restart if not shutting down; register SIGINT/SIGTERM handlers to set shutdown Event and terminate all workers; status reporter prints line count, rate, alive workers every 60s) in src/sealed/application/match_outcomes.py
- [X] T016 [US1] Extend CLI with match-outcomes subcommand (add subparser with --workers int default 12, wire to MatchOutcomeSupervisor, ensure output dir ./output/sealed/ is created) in src/sealed/infrastructure/cli.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Running `python -m sealed match-outcomes --workers 2` should produce match outcomes using the standard Forge deck builder.

---

## Phase 4: User Story 2 - Configurable Parallelism (Priority: P2)

**Goal**: The --workers argument (already wired in US1) correctly controls the exact number of spawned Java worker processes.

**Independent Test**: Run with `--workers 2`, verify exactly 2 worker processes are spawned. Run without `--workers`, verify 12 workers are spawned by default.

### Tests for User Story 2 (MANDATORY per Constitution)

- [X] T017 [US2] Add tests verifying --workers argument: explicit value spawns exact count, omitted value spawns default 12, in tests/unit/sealed/application/test_match_outcomes.py

**Checkpoint**: Worker count is configurable and verified

---

## Phase 5: User Story 3 - Varied Deck Quality via Multiple Construction Methods (Priority: P2)

**Goal**: DeckBuilder supports four construction methods with weighted random selection (40/30/20/10), producing decks of varying quality from competent to random. Methods 2-4 include land rebalancing via Forge's LimitedDeckBuilder.

**Independent Test**: Collect a sample of generated decks and verify the four methods appear at approximately their expected proportions (within 5 percentage points per SC-003).

### Tests for User Story 3 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T018 [P] [US3] Unit test for DeckBuilder (verify weighted method selection distribution over many calls; verify method 2 swaps exactly 3 nonland cards; verify method 3 swaps exactly 8 nonland cards; verify method 4 picks 23 random nonland cards; verify rebalanceLands produces exactly 40-card decks; verify FR-010: for methods 2-4, each card instance in the resulting deck appears at most once from the original pool — no pool card reuse after swaps) in forge-connector/src/test/java/com/pricepredictor/connector/DeckBuilderTest.java

### Implementation for User Story 3

- [X] T019 [US3] Add deck construction methods 2-4 and weighted selection to DeckBuilder: method 2 (buildWithSwaps 3 nonland cards), method 3 (buildWithSwaps 8 nonland cards), method 4 (buildRandom: pick 23 random nonland cards); add rebalanceLands (remove basic lands, keep non-basic lands, use SealedDeckBuilder/LimitedDeckBuilder addLands to fill to 40 cards); add selectMethod with weights [0.4, 0.3, 0.2, 0.1] in forge-connector/src/main/java/com/pricepredictor/connector/DeckBuilder.java

**Checkpoint**: Decks are now built using all four methods with correct weight distribution and proper land rebalancing

---

## Phase 6: User Story 4 - Expansion Set Diversity (Priority: P3)

**Goal**: MatchGenerator filters eligible sets to only those with draft/play boosters and excludes un-sets, ensuring matches use a wide variety of valid sealed-legal sets.

**Independent Test**: Run generation for several hundred outcomes and verify that multiple distinct sets appear and no un-sets or non-draft sets are used.

### Tests for User Story 4 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T020 [P] [US4] Unit test for eligible set filtering (verify sets with draft booster template are included; verify sets without draft booster template are excluded; verify Type.FUNNY sets are excluded even if they have boosters) in forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorTest.java

### Implementation for User Story 4

- [X] T021 [US4] Refine eligible set selection in MatchGenerator: iterate StaticData.instance().getEditions(), include only sets where edition.getBoosterTemplate("Draft") != null AND edition.getType() != CardEdition.Type.FUNNY; compute filtered list once at construction time, pick random set from it per match in forge-connector/src/main/java/com/pricepredictor/connector/MatchGenerator.java

**Checkpoint**: All user stories are now independently functional. Match generation uses diverse sealed-legal sets with varied deck quality.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Verify nothing is broken and the full system works end-to-end

- [ ] T022 [P] Verify existing forge-connector entry points (PoolMain, ConvertMain) still function correctly after ForgeEnvironmentInitializer rewrite — run existing commands and confirm unchanged behavior
- [ ] T023 Run quickstart.md end-to-end validation: build forge-connector (mvn package -DskipTests), run python -m sealed match-outcomes --workers 2, verify output file contains well-formed records per contracts/cli.md format spec; note observed throughput rate against SC-001 target (≥500 matches/hour at 12 workers)
- [X] T024 [P] Update project root README.md with the new `match-outcomes` command: add workflow description (supervisor/worker architecture, inputs, outputs), document the output format, and describe how to run and stop the generation process (per Constitution VI)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-6)**: All depend on Foundational phase completion
  - US1 (Phase 3): Must complete first — provides the end-to-end pipeline
  - US2 (Phase 4): Depends on US1 (verifies --workers behavior wired in US1)
  - US3 (Phase 5): Depends on US1 (extends DeckBuilder from Phase 3)
  - US4 (Phase 6): Depends on US1 (refines MatchGenerator from Phase 3)
  - US3 and US4 can proceed in parallel after US1
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 (tests the --workers arg added in US1 CLI)
- **User Story 3 (P2)**: Depends on US1 (extends DeckBuilder created in US1) - independently testable
- **User Story 4 (P3)**: Depends on US1 (refines MatchGenerator created in US1) - independently testable

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Value objects/models before services
- Domain logic (DeckBuilder, GamePlayer) before orchestration (MatchGenerator)
- Java worker (MatchWorkerMain) before Python supervisor (MatchOutcomeSupervisor)
- Supervisor before CLI extension
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 2**: T002 + T003 in parallel; then T004 + T005 in parallel; T006 after T005
- **Phase 3 tests**: T007 + T008 + T009 all in parallel (different languages/files)
- **Phase 3 impl**: T010 + T011 + T014 in parallel (GamePlayer, DeckBuilder, MatchWorkerConnector are independent files); then T012 → T013 → T015 → T016 sequentially
- **Phase 5 + 6**: Can proceed in parallel after Phase 3 (US3 modifies DeckBuilder, US4 modifies MatchGenerator — different files)
- **Phase 7**: T022 + T023 in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (TDD: write first, watch fail):
Task T007: "Unit test for MatchGenerator in MatchGeneratorTest.java"
Task T008: "Unit test for supervisor in test_match_outcomes.py"
Task T009: "Unit test for connector in test_match_worker_connector.py"

# Launch independent Java domain classes together:
Task T010: "Implement GamePlayer in GamePlayer.java"
Task T011: "Implement DeckBuilder in DeckBuilder.java"

# Launch Python connector in parallel with Java orchestration:
Task T014: "Implement MatchWorkerConnector in match_worker_connector.py"
Task T012: "Implement MatchGenerator in MatchGenerator.java"  (after T010, T011)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (add dependencies)
2. Complete Phase 2: Foundational (GuiHeadless, ForgeEnvironmentInitializer rewrite, MatchResult, MatchResultWriter)
3. Complete Phase 3: User Story 1 (full pipeline with method-1 deck building)
4. **STOP and VALIDATE**: Run `python -m sealed match-outcomes --workers 2`, verify output
5. This is already a functional data generator producing training data

### Incremental Delivery

1. Setup + Foundational = build infrastructure
2. Add US1 = functional match generator with standard decks (MVP!)
3. Add US2 = verified configurable parallelism
4. Add US3 = diverse deck quality (richer training signal)
5. Add US4 = proper set filtering (broader generalization)
6. Each story improves data quality without breaking the pipeline

### Parallel Team Strategy

With two developers after US1 is complete:

- Developer A: User Story 3 (DeckBuilder methods 2-4)
- Developer B: User Story 4 (eligible set filtering in MatchGenerator)
- Both modify different Java files, no conflicts

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The jumpstart-tierlist project (`../jumpstart-tierlist`) provides reference implementations for GuiHeadless, FModel initialization, and the supervisor/worker pattern
