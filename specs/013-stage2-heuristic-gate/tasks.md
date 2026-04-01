# Tasks: Stage 2 Training — Heuristic Gate

**Input**: Design documents from `/specs/013-stage2-heuristic-gate/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Foundational (Protocol Extension + Adapter Extraction)

**Purpose**: Extract the shared embedding adapter, extend the domain protocol, and fix test fixtures — blocking prerequisites for ALL user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T001 Add `get_card_text(card_name: str) -> str` method to `CardEmbeddingPort` protocol in src/sealed/domain/card_embedding_port.py
- [X] T002 Write unit tests for `EmbeddingAdapter` (including `get_card_text()` with file caching, `get_embedding()`, and `is_land()`) in tests/unit/sealed/infrastructure/test_embedding_adapter.py
- [X] T003 Extract `_EmbeddingAdapter` from src/sealed/application/train_stage1.py to src/sealed/infrastructure/embedding_adapter.py as public `EmbeddingAdapter`, adding `get_card_text()` with file-read caching
- [X] T004 [P] Fix test fixtures in tests/fixtures/converted_cards/ to match production format — replace any wrong formats (e.g. `mana[N]:`) with exact copies of the corresponding cards from output/cardsfolder/ per research.md §1
- [X] T005 [P] Update src/sealed/application/train_stage1.py — remove `_EmbeddingAdapter` class, import `EmbeddingAdapter` from `sealed.infrastructure.embedding_adapter`
- [X] T006 [P] Update src/sealed/application/sample_stage1.py — import `EmbeddingAdapter` from `sealed.infrastructure.embedding_adapter` instead of `train_stage1._EmbeddingAdapter`
- [X] T007 Run existing Stage 1 test suite (`pytest tests/unit/sealed/ tests/integration/sealed/`) to verify adapter extraction is non-breaking

**Checkpoint**: Protocol extended, adapter extracted, fixtures corrected, all existing tests pass.

---

## Phase 2: User Story 2 — Heuristic Mana Score Computation (Priority: P1) 🎯 MVP

**Goal**: Pure domain module that counts pips, computes ideal mana distributions, counts actual sources, and produces a score in [0, 1] — the reward signal for Stage 2 training.

**Independent Test**: Construct known decks with predictable mana distributions and verify scores match hand-calculated expected values.

### Tests for User Story 2 ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation. Any new test card fixtures MUST be copied verbatim from output/cardsfolder/.**

- [X] T008 [US2] Write unit tests for `count_pips()` covering single-color ({W}, {U}), Phyrexian ({W/P} → +0.5), hybrid ({G/R} → +0.5 each), generic ({2}, {X} → ignored), colorless ({C} → +1.0), multi-face cards (all faces counted), and no-mana-cost cards in tests/unit/sealed/domain/test_mana_scorer.py
- [X] T009 [US2] Write unit tests for `compute_ideal_distribution()` covering single-color deck (all 17 to one color), multi-color (proportional + 2-source minimum floor), and zero-pip edge case in tests/unit/sealed/domain/test_mana_scorer.py
- [X] T010 [US2] Write unit tests for `count_actual_sources()` covering basic lands (`{T}: add {W}`), dual lands (`add {G} or {U}` → +1 each), tri-lands (`add {R}, {G}, or {W}`), Sol Ring (`add {C}{C}` → +1 C not +2), non-mana activated abilities (filtered out), and `add one mana of any color` (→ +0) in tests/unit/sealed/domain/test_mana_scorer.py
- [X] T011 [US2] Write unit tests for `compute_mana_score()` covering perfect match (score=1.0), land-count deviation penalty, distribution mismatch, combined errors, score floor at 0.0, reward mapping (2*score−1), and edge cases (all lands, all spells, colorless-only) in tests/unit/sealed/domain/test_mana_scorer.py

### Implementation for User Story 2

- [X] T012 [US2] Define `PipCounts`, `IdealDistribution`, `ActualSourceCounts`, `ManaScore` value objects and implement `count_pips()` — parse `mana cost:` lines, handle all pip types per FR-006, multi-face `ALTERNATE` separator in src/sealed/domain/mana_scorer.py
- [X] T013 [US2] Implement `compute_ideal_distribution()` — 2-source minimum per color present, proportional allocation of remaining from 17 total per FR-007 in src/sealed/domain/mana_scorer.py
- [X] T014 [US2] Implement `count_actual_sources()` — match `activated[N]: {T}: add ...` pattern, extract distinct color symbols `{W/U/B/R/G/C}` per ability line per FR-008 and research section 7 in src/sealed/domain/mana_scorer.py
- [X] T015 [US2] Implement `compute_mana_score()` — L1 error + land-count penalty, `score = max(0.0, 1.0 - (l1 + |n_lands - 17|) / 17.0)`, `reward = 2 * score - 1` per FR-009/FR-010 in src/sealed/domain/mana_scorer.py
- [X] T016 [US2] Verify all mana scorer unit tests pass via `pytest tests/unit/sealed/domain/test_mana_scorer.py -v`

**Checkpoint**: Mana scorer is fully functional and independently tested with hand-calculated values.

---

## Phase 3: User Story 1 — Launch Stage 2 Training (Priority: P1)

**Goal**: The core Stage 2 training loop — load Stage 1 weights, run episodes with mana-score reward for completed decks (Stage 1 fallback for duplicates), PPO update, save checkpoints, halt when all 32 episodes score > 0.90.

**Independent Test**: Run `python -m sealed train --stage 2` against a small dataset and observe episodes completing, mana scores computed, checkpoint written.

**Depends on**: Phase 2 (US2 mana scorer).

### Tests for User Story 1 ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T017 [P] [US1] Write unit tests for `TrainStage2UseCase` — mock episode runner, mana scorer, PPO trainer; test reward override for completed episodes (uniform mana score), Stage 1 fallback for duplicate episodes, completion criterion (all 32 > 0.90), batch logging format, and verify card_encoder parameters receive no gradients during training step (FR-002) in tests/unit/sealed/application/test_train_stage2.py
- [X] T018 [P] [US1] Write unit tests for CLI `--stage 2` and `--init-from` argument parsing, stage-dependent `--model-path` defaults, and error when neither checkpoint exists in tests/unit/sealed/infrastructure/test_cli_sealed_train_stage2.py

### Implementation for User Story 1

- [X] T019 [US1] Implement `TrainStage2UseCase` in src/sealed/application/train_stage2.py — init from Stage 1 weights (model only, fresh optimizer, episode_count=0), run 32-episode batches via EpisodeRunner, overwrite step_rewards with uniform mana-score reward for completed episodes (FR-005), keep Stage 1 per-step rewards for duplicate episodes (FR-003), PPO update with advantage normalization (FR-011), save latest.pt after each batch (FR-013 partial, also covers US4-AC2), halt when all 32 score > 0.90 (FR-012), stdout format per CLI contract
- [X] T020 [US1] Add `--init-from` argument (default `models/sealed/stage1/{set}/latest.pt`) and `--stage 2` routing to train subcommand in src/sealed/infrastructure/cli.py — stage-dependent `--model-path` default (`models/sealed/stage2/{set}/latest.pt` for stage 2)
- [X] T021 [US1] Write integration test for Stage 2 training — small card set, few batches, verify episodes run, pool is reshuffled before each pick (FR-016), mana scores computed, checkpoint saved in tests/integration/sealed/test_train_stage2_integration.py
- [X] T022 [US1] Run all Stage 2 training tests via `pytest tests/unit/sealed/application/test_train_stage2.py tests/unit/sealed/infrastructure/test_cli_sealed_train_stage2.py tests/integration/sealed/test_train_stage2_integration.py -v`

**Checkpoint**: Stage 2 training is fully functional — can start from Stage 1 checkpoint, run training loop, compute mana scores, and halt on convergence.

---

## Phase 4: User Story 3 — Inspect Stage 2 Sample Picks (Priority: P2)

**Goal**: A researcher can view what the model picks during Stage 2 and see mana analysis (ideal vs actual distributions + score) to qualitatively assess training progress.

**Independent Test**: Run `python -m sealed sample --stage 2` after any Stage 2 checkpoint exists, verify human-readable output with mana analysis.

### Tests for User Story 3 ✅

- [X] T023 [P] [US3] Write unit tests for `SampleStage2UseCase` — mock dependencies, verify output includes 40 picks, per-color ideal vs actual mana distributions, and mana score per CLI contract output format in tests/unit/sealed/application/test_sample_stage2.py
- [X] T024 [P] [US3] Write unit tests for CLI `--stage 2` routing on sample subcommand and stage-dependent `--model-path` default in tests/unit/sealed/infrastructure/test_cli_sealed_sample_stage2.py

### Implementation for User Story 3

- [X] T025 [US3] Implement `SampleStage2UseCase` in src/sealed/application/sample_stage2.py — run N episodes from random pools, display 40 picks per deck, compute and show ideal vs actual mana source distributions per color (W/U/B/R/G/C), land count, and mana score per CLI contract format
- [X] T026 [US3] Add `--stage 2` routing to sample subcommand in src/sealed/infrastructure/cli.py — stage-dependent `--model-path` default
- [X] T027 [US3] Verify sample tests pass via `pytest tests/unit/sealed/application/test_sample_stage2.py tests/unit/sealed/infrastructure/test_cli_sealed_sample_stage2.py -v`

**Checkpoint**: Sample command shows human-readable deck picks with mana analysis.

---

## Phase 5: User Story 4 — Resume Stage 2 from Checkpoint (Priority: P2)

**Goal**: A researcher whose Stage 2 training was interrupted can resume from the most recent checkpoint, preserving episode count and optimizer state. Timestamped snapshots are saved every 1000 episodes.

**Independent Test**: Run Stage 2 briefly, stop, re-launch, verify training resumes from saved state.

**Depends on**: Phase 3 (US1 TrainStage2UseCase).

### Tests for User Story 4 ✅

- [X] T028 [US4] Write unit tests for checkpoint resume behavior — model-path takes priority over init-from, episode count preserved on resume, optimizer state preserved, fresh optimizer on init-from, error when neither checkpoint exists — in tests/unit/sealed/application/test_train_stage2.py
- [X] T029 [US4] Write unit test for timestamped checkpoint saving every 1000 episodes in tests/unit/sealed/application/test_train_stage2.py

### Implementation for User Story 4

- [X] T030 [US4] Add checkpoint priority logic to `TrainStage2UseCase` — if model-path exists load full checkpoint (model + optimizer + training state), else fall through to init-from path, else error — in src/sealed/application/train_stage2.py
- [X] T031 [US4] Add timestamped checkpoint saving every 1000 episodes to `checkpoints/` subfolder in src/sealed/application/train_stage2.py
- [X] T032 [US4] Verify resume and checkpoint tests pass via `pytest tests/unit/sealed/application/test_train_stage2.py -v -k "resume or checkpoint"`

**Checkpoint**: Training can be interrupted and restarted without loss of progress, with periodic timestamped snapshots.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all stories.

- [X] T033 Run full test suite via `cd src && pytest` and fix any failures
- [X] T034 Run linter via `cd src && ruff check .` and fix any issues
- [X] T035 Run quickstart.md validation — verify all documented commands in specs/013-stage2-heuristic-gate/quickstart.md work as described
- [X] T036 Update project-level documentation with `--stage 2` workflow, `--init-from` flag, and sample output format per constitution principle VI

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — can start immediately. BLOCKS all user stories.
- **US2 Mana Score (Phase 2)**: Depends on Phase 1 (adapter provides card text access). No dependency on other stories.
- **US1 Training Loop (Phase 3)**: Depends on Phase 2 (needs mana scorer). This is the core integration phase.
- **US3 Sample (Phase 4)**: Depends on Phase 2 (needs mana scorer) and Phase 3 (reuses similar infrastructure). Can start after Phase 2 if developed independently of CLI routing.
- **US4 Resume (Phase 5)**: Depends on Phase 3 (adds to TrainStage2UseCase).
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US2 (P1)**: Can start after Foundational — pure domain, no dependency on other stories
- **US1 (P1)**: Must follow US2 — training loop requires the mana scorer
- **US3 (P2)**: Can start after US2 (needs mana scorer) — independent of US1 for domain logic, shares CLI file for routing
- **US4 (P2)**: Must follow US1 — extends TrainStage2UseCase with resume logic

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Value objects / domain logic before application use cases
- Use cases before CLI routing
- Unit tests before integration tests
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1**: T004, T005, and T006 can run in parallel (different files, all depend on T003)
- **Phase 2**: T008–T011 can be written as a single test-writing session (same file)
- **Phase 3**: T017 and T018 can run in parallel (different test files)
- **Phase 4**: T023 and T024 can run in parallel (different test files)
- **Phase 2 + Phase 4 partial**: US3 test writing (T023, T024) can begin after Phase 2 since SampleStage2UseCase shares domain dependencies with US1 but not the use case itself

---

## Parallel Example: Phase 3 (User Story 1)

```bash
# Write both test files in parallel:
Task T017: "Unit tests for TrainStage2UseCase in tests/unit/sealed/application/test_train_stage2.py"
Task T018: "CLI train tests for --stage 2 in tests/unit/sealed/infrastructure/test_cli_sealed_train.py"

# Then implement sequentially (use case before CLI routing):
Task T019: "Implement TrainStage2UseCase in src/sealed/application/train_stage2.py"
Task T020: "Add --stage 2 routing in src/sealed/infrastructure/cli.py"
```

---

## Implementation Strategy

### MVP First (US2 + US1)

1. Complete Phase 1: Foundational (adapter extraction + fixture fix)
2. Complete Phase 2: US2 Mana Score (domain module)
3. Complete Phase 3: US1 Training Loop (end-to-end training)
4. **STOP and VALIDATE**: Run `python -m sealed train --stage 2` against a small dataset
5. At this point, Stage 2 training is fully functional

### Incremental Delivery

1. Phase 1 → Adapter extracted, fixtures corrected, existing tests pass
2. Phase 2 (US2) → Mana scorer tested independently with hand-calculated values
3. Phase 3 (US1) → Training loop works end-to-end (MVP complete!)
4. Phase 4 (US3) → Sample inspection for qualitative assessment
5. Phase 5 (US4) → Interrupt/resume for long training runs
6. Phase 6 → Full validation + documentation

---

## Notes

- [P] tasks = different files, no dependencies within the phase
- [US#] label maps task to specific user story for traceability
- US2 before US1 because the mana scorer is a prerequisite for the training loop
- Adapter extraction (Phase 1) is the only modification to existing Stage 1 code
- Test fixtures in tests/fixtures/converted_cards/ MUST be exact copies of corresponding cards from output/cardsfolder/ (research.md §1)
- Research section 7 documents the regex patterns for mana source parsing
- Domain mana_scorer module is pure functions (no I/O) — card text comes via CardEmbeddingPort
