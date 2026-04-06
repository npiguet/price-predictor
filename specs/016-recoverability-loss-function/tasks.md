# Tasks: Recoverability-Based Per-Step Stage 2 Loss

**Input**: Design documents from `/specs/016-recoverability-loss-function/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No new project structure needed — all changes modify existing files. This phase ensures the branch is ready.

- [ ] T001 Verify existing tests pass by running `pytest` from `src/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No foundational blocking tasks. All existing infrastructure (mana_scorer.py, episode_runner.py, replay_buffer.py, PPO trainer) is already in place from features 013/015.

**Checkpoint**: Foundation ready — user story implementation can begin.

---

## Phase 3: User Story 2 — Recoverability Ratio Computation (Priority: P1)

**Goal**: Implement `compute_recoverability_ratio()` — the pure-math function that measures how critical the current mana imbalance is given remaining picks. This is the mathematical foundation for the shaping signal.

**Independent Test**: Construct known deck states at various episode points and verify the ratio matches hand-calculated expected values for early picks (low urgency), late picks (high urgency), and boundary conditions (no spells, perfect balance, terminal step).

### Tests for User Story 2 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T002 [P] [US2] Write unit tests for `compute_recoverability_ratio()` in tests/unit/sealed/domain/test_mana_scorer.py — cover: step 5 with imbalance 4.0 and exponent 2 → 4.0/35²≈0.00327; step 35 with same imbalance → 4.0/5²=0.16; no spells picked → ratio 0; perfect balance → ratio 0; remaining_picks=0 → ratio equals raw imbalance; custom exponent values

### Implementation for User Story 2

- [ ] T003 [US2] Implement `compute_recoverability_ratio(pip_counts, actual_sources, remaining_picks, exponent)` in src/sealed/domain/mana_scorer.py — reuse existing `compute_ideal_distribution()`, compute L1 imbalance across all 6 colors (W/U/B/R/G/C), divide by `max(remaining_picks, 1) ** exponent` per data-model.md

**Checkpoint**: `compute_recoverability_ratio()` passes all unit tests. Can be verified independently by constructing PipCounts/actual_sources dicts and checking return values.

---

## Phase 4: User Story 1 — Per-Step Recoverability Reward in Stage 2 Training (Priority: P1)

**Goal**: Replace the uniform end-of-episode mana score reward with a per-step reward that combines the Stage 1 budget signal (+1/-1) with a recoverability-based shaping signal bounded via `tanh`. Each of the 40 picks receives its own reward reflecting whether it improved or worsened the deck's mana recoverability.

**Independent Test**: Run Stage 2 training on a small dataset and verify each step receives a distinct reward value (not uniform), rewards fall within (-2, 2), and picks that improve mana balance receive higher rewards than picks that worsen it.

### Tests for User Story 1 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T004 [P] [US1] Write unit tests for `compute_per_step_rewards()` in tests/unit/sealed/domain/test_mana_scorer.py — cover: returns float32[40] array; values in (-2, 2); Plains pick into under-sourced white deck → positive shaping; Plains pick into over-sourced white deck → negative shaping; spell pick that shifts ideal → correct delta; shaping signals are not all identical (per-step credit assignment); mean_shaping and final_imbalance diagnostic values returned correctly
- [ ] T005 [P] [US1] Write unit tests for per-step reward integration in tests/unit/sealed/application/test_train_stage2.py — cover: `step_rewards` on Episode objects are overridden with per-step values (not uniform); `ep.reward` still holds the end-of-episode mana score for convergence logging

### Implementation for User Story 1

- [ ] T006 [US1] Implement `compute_per_step_rewards()` in src/sealed/domain/mana_scorer.py — signature per data-model.md: takes actions, pool_names, card_port, budget_rewards, urgency_exponent, temperature; replays 40 picks maintaining running pip_counts and actual_sources; computes ratio_before/ratio_after per step; delta = ratio_before - ratio_after; shaping = tanh(delta / temperature); step_rewards[t] = budget_rewards[t] + shaping; returns PerStepRewardResult (step_rewards, mean_shaping, final_imbalance)
- [ ] T007 [US1] Modify `TrainStage2UseCase.execute()` in src/sealed/application/train_stage2.py — replace the uniform `np.full(len(ep.actions), ms.reward)` reward assignment (lines 184-185) with a call to `compute_per_step_rewards()` using `ep.step_rewards` (budget rewards from episode runner) and the episode's actions/pool_names; keep `ep.reward = ms.reward` for convergence checking

**Checkpoint**: Stage 2 training produces per-step rewards. Each step's reward is distinct. `ep.reward` still holds the mana score for convergence.

---

## Phase 5: User Story 3 — Stage 1 Budget Signal Preserved at Full Strength (Priority: P1)

**Goal**: Verify that the Stage 1 budget reward (+1/-1) is preserved at full strength — no scaling, weighting, or replacement. The shaping signal is purely additive.

**Independent Test**: Verify that the Stage 1 component of the total reward is always exactly +1 or -1 (unchanged from Stage 1 behaviour), and that the total reward is the arithmetic sum of budget + shaping.

### Tests for User Story 3 (MANDATORY per Constitution)

- [ ] T008 [US3] Write unit tests verifying budget signal preservation in tests/unit/sealed/domain/test_mana_scorer.py — cover: for every step, step_rewards[t] == budget_rewards[t] + shaping[t] exactly; budget_rewards are always ±1 (unchanged from episode runner output); no scaling factor applied to either component; total reward bounded to (-2, 2)

### Implementation for User Story 3

No separate implementation — the budget signal preservation is enforced by the additive composition in `compute_per_step_rewards()` (T006) and the fact that `budget_rewards` from the episode runner are passed through unmodified. T008 tests serve as the verification gate.

**Checkpoint**: All US3 tests pass, confirming budget signal is untouched.

---

## Phase 6: User Story 4 — Configurable Hyperparameters (Priority: P2)

**Goal**: Expose `urgency_exponent` and `temperature` as CLI arguments on `sealed train --stage 2`, passing them through to `compute_per_step_rewards()`.

**Independent Test**: Run training with `--urgency-exponent 3 --temperature 0.5` and verify reward values differ from default-hyperparameter runs.

### Tests for User Story 4 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T009 [P] [US4] Write unit tests for CLI args in tests/unit/sealed/infrastructure/test_cli_sealed_train_stage2.py — cover: `--urgency-exponent 3` parsed correctly; `--temperature 0.5` parsed correctly; defaults are exponent=2.0 and temperature=1.0 when not specified; help text includes `(default: ...)` suffix matching CLI conventions

### Implementation for User Story 4

- [ ] T010 [US4] Add `--urgency-exponent` and `--temperature` arguments to `train_parser` in src/sealed/infrastructure/cli.py — type=float, defaults 2.0 and 1.0, help text per data-model.md conventions
- [ ] T011 [US4] Thread `urgency_exponent` and `temperature` through `run_train()` → `TrainStage2UseCase.execute()` → `compute_per_step_rewards()` in src/sealed/infrastructure/cli.py and src/sealed/application/train_stage2.py — add parameters to `execute()` signature with defaults matching CLI defaults

**Checkpoint**: `sealed train --stage 2 --urgency-exponent 3 --temperature 0.5` passes args all the way to reward computation.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Sample output enhancement (FR-011), batch logging (FR-012), and final validation.

- [ ] T012 [P] Write unit tests for mana cost prefix display in tests/unit/sealed/application/test_sample_stage2.py — cover: non-land card shows `{mana_cost} CardName` format; land card shows name only (no cost prefix); multi-face card shows first-face cost
- [ ] T013 [P] Write unit tests for batch log format in tests/unit/sealed/application/test_train_stage2.py — cover: batch print line includes `shaping=X.XX` and `imbalance=X.X` fields alongside existing `mean_score` and timing fields
- [ ] T014 Modify sample output in src/sealed/application/sample_stage2.py to prefix each non-land pick with its mana cost string (FR-011) — extract the `mana cost:` line from card text; format as `{cost} CardName`; lands print name only
- [ ] T015 Modify batch logging in src/sealed/application/train_stage2.py to include `shaping=` (batch-mean shaping signal) and `imbalance=` (batch-mean final imbalance) on the existing print line (FR-012) — values come from `PerStepRewardResult.mean_shaping` and `final_imbalance` accumulated across the batch
- [ ] T016 Run `ruff check .` from `src/` and fix any lint warnings
- [ ] T017 Run full test suite (`pytest` from `src/`) and verify all tests pass
- [ ] T018 Validate quickstart.md scenarios manually — verify CLI commands from quickstart.md parse correctly (arg parsing, not full training run)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: N/A — no blocking tasks
- **US2 (Phase 3)**: Can start immediately after Phase 1 — no dependencies on other stories
- **US1 (Phase 4)**: Depends on US2 (Phase 3) — `compute_per_step_rewards()` calls `compute_recoverability_ratio()`
- **US3 (Phase 5)**: Depends on US1 (Phase 4) — verification tests exercise the reward composition
- **US4 (Phase 6)**: Can start after Phase 1 (CLI args are independent of reward logic), but threading through to `execute()` depends on US1 (Phase 4)
- **Polish (Phase 7)**: FR-011 (T012/T014) can start any time after Phase 1. FR-012 (T013/T015) depends on US1 (Phase 4). Final validation (T016-T018) depends on all prior phases.

### User Story Dependencies

- **User Story 2 (P1)**: No dependencies — pure domain math
- **User Story 1 (P1)**: Depends on User Story 2 (uses `compute_recoverability_ratio()`)
- **User Story 3 (P1)**: Depends on User Story 1 (verifies reward composition)
- **User Story 4 (P2)**: CLI parsing is independent; threading depends on User Story 1

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Domain functions before application-layer integration
- Core implementation before cross-cutting concerns

### Parallel Opportunities

- T002 (US2 tests) and T009 (US4 CLI tests) and T012/T013 (polish tests) can all run in parallel — different files
- T004 and T005 (US1 tests) can run in parallel — different test files
- T012 and T013 (polish tests) can run in parallel — different test files
- T014 and T015 (polish implementation) can run in parallel — different source files

---

## Parallel Example: User Story 1

```
# Write tests in parallel (different files):
Task T004: "Unit tests for compute_per_step_rewards() in tests/unit/sealed/domain/test_mana_scorer.py"
Task T005: "Unit tests for per-step reward integration in tests/unit/sealed/application/test_train_stage2.py"

# Then implement sequentially (T006 before T007 — domain before application):
Task T006: "Implement compute_per_step_rewards() in src/sealed/domain/mana_scorer.py"
Task T007: "Modify TrainStage2UseCase.execute() in src/sealed/application/train_stage2.py"
```

---

## Implementation Strategy

### MVP First (User Story 2 + User Story 1)

1. Complete Phase 1: Setup (verify green baseline)
2. Complete Phase 3: User Story 2 — `compute_recoverability_ratio()` with tests
3. Complete Phase 4: User Story 1 — `compute_per_step_rewards()` + training integration
4. **STOP and VALIDATE**: Run Stage 2 training briefly and verify per-step rewards appear in episode data
5. Complete Phase 5: User Story 3 — Verify budget signal preservation

### Incremental Delivery

1. US2 → ratio computation works → test independently
2. US1 → per-step reward works → training produces non-uniform rewards (MVP!)
3. US3 → budget preservation verified → confidence in correctness
4. US4 → hyperparameters tunable from CLI → researcher can experiment
5. Polish → sample output, batch logging, final validation → feature complete

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- All new functions go in `mana_scorer.py` (domain layer) — no infrastructure imports allowed
- `compute_per_step_rewards()` reuses existing `count_pips()`, `compute_ideal_distribution()`, `count_actual_sources()` — do not duplicate this logic
- The episode runner's `step_rewards` already contains the ±1 budget signal — these become `budget_rewards` input to `compute_per_step_rewards()`
- `ep.reward` (scalar mana score) is kept for convergence checking and logging — only `ep.step_rewards` changes
- Research decision R1: post-episode replay approach (not inline in EpisodeRunner)
- Research decision R5: no GAE needed initially; current PPO hyperparameters are adequate
