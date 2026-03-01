# Tasks: CardMarket EUR Pricing

**Input**: Design documents from `/specs/004-cardmarket-eur-pricing/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Test Fixtures)

**Purpose**: Add a fixture card that passes name-to-UUID filters but has no price entry, enabling FR-006 and FR-008 testing.

- [ ] T001 Add a "No Price Card" to test fixtures: add a card entry (`name: "No Price Card"`, `uuid: "ffffffff-aaaa-aaaa-aaaa-aaaaaaaaaaaa"`) to `tests/fixtures/allprintings_sample.json` in a new set block `"NPC"` with `availability: ["paper"]`, `isFunny: false`, `isOnlineOnly: false`, `language: "English"`. Do NOT add a corresponding entry to `tests/fixtures/allprices_sample.json` — this card intentionally has no price data.

**Checkpoint**: Fixtures updated — both JSON files parse correctly, existing tests still pass.

---

## Phase 2: User Story 1 — Training Data Uses CardMarket EUR Prices (Priority: P1) — Formalization

**Goal**: Verify that the existing codebase already sources all training prices from CardMarket EUR. No code changes needed — this story formalizes existing behavior.

**Independent Test**: Run existing test suite and confirm all price extraction tests pass against CardMarket EUR fixture data.

**Covers**: FR-001, FR-002, FR-003, FR-004, FR-005

- [ ] T002 [US1] Run existing test suite (`cd src && pytest ../tests/unit/infrastructure/test_mtgjson_loader.py -v`) and verify that `TestGetCheapestPrice` and `TestBuildPriceMapFloor` tests pass — these tests exercise the `paper → cardmarket → retail` extraction path against EUR fixture data, confirming FR-001 through FR-005 are already implemented. No new tests or code changes needed.

**Checkpoint**: Existing CardMarket EUR behavior verified by existing tests.

---

## Phase 3: User Story 2 — Predictions Output in EUR (Priority: P2) — Formalization

**Goal**: Verify that prediction and evaluation output is denominated in EUR. No code changes needed — this is an inherent consequence of training on EUR data.

**Independent Test**: Confirm EUR field names in predict.py and evaluate.py, and that existing tests validate EUR output.

**Covers**: FR-009, FR-010

- [ ] T003 [US2] Verify EUR output fields: confirm `src/price_predictor/application/predict.py` returns `predicted_price_eur` and `src/price_predictor/application/evaluate.py` uses `mean_absolute_error_eur`. Run `cd src && pytest` to confirm all existing tests pass. No new tests or code changes needed.

**Checkpoint**: EUR prediction output verified by existing field names and tests.

---

## Phase 4: User Story 3 — Cards Without CardMarket Prices Are Excluded (Priority: P3)

**Goal**: Verify exclusion of cards without CardMarket prices (FR-006) and add exclusion count logging (FR-008). FR-008 is the only new code in this feature.

**Independent Test**: Build a price map from fixtures including the "No Price Card" (which has no price entry), verify the card is excluded and the exclusion count is logged.

**Covers**: FR-006, FR-007, FR-008

### Tests for User Story 3 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T004 [P] [US3] Write test for missing card exclusion (FR-006) in `tests/unit/infrastructure/test_mtgjson_loader.py`: call `build_name_to_uuids()` then `build_price_map()` with fixtures, verify "No Price Card" is NOT in the returned price map (it has no price data)
- [ ] T005 [P] [US3] Write test for exclusion count logging (FR-008) in `tests/unit/infrastructure/test_mtgjson_loader.py`: using `caplog` at `logging.INFO` level for logger `price_predictor.infrastructure.mtgjson_loader`, call `build_price_map()` with fixtures, verify (a) an INFO log message containing "exclusion" (case-insensitive) is emitted, (b) the message contains the correct count of excluded cards (cards in `name_to_uuids` with no price in the price map)

### Implementation for User Story 3

- [ ] T006 [US3] Add exclusion count log line in `build_price_map()` in `src/price_predictor/infrastructure/mtgjson_loader.py`: after the existing `for name, uuids` loop and after the "Price selection summary" log, compute `excluded_count = len(name_to_uuids) - len(result)` and add `logger.info("Price exclusion: %d of %d cards excluded (no CardMarket price available)", excluded_count, len(name_to_uuids))` before the existing "Loaded price data" log line
- [ ] T007 [US3] Run full test suite (`cd src && pytest`) — verify T004/T005 tests pass and all existing tests in `test_mtgjson_loader.py`, `test_train.py`, `test_train_logging.py`, and `test_end_to_end.py` still pass with no regressions

**Checkpoint**: Exclusion count logging working. Cards without CardMarket prices excluded and counted.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup across all changes

- [ ] T008 [P] Run ruff linting (`cd src && ruff check .`) and fix any style or lint issues introduced by this feature
- [ ] T009 [P] Run end-to-end integration tests (`cd src && pytest tests/integration/`) to verify the full train → predict → evaluate pipeline works with updated logging
- [ ] T010 [P] Validate that `specs/004-cardmarket-eur-pricing/quickstart.md` log format examples match the actual log output format implemented in T006

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **User Story 1 (Phase 2)**: No dependencies — verification only (can run in parallel with Phase 1)
- **User Story 2 (Phase 3)**: No dependencies — verification only (can run in parallel with Phase 1)
- **User Story 3 (Phase 4)**: Depends on Phase 1 (needs "No Price Card" fixture for tests)
- **Polish (Phase 5)**: Depends on Phase 4

### User Story Dependencies

- **User Story 1 (P1)**: Independent — verification only, no code changes
- **User Story 2 (P2)**: Independent — verification only, no code changes
- **User Story 3 (P3)**: Depends on Phase 1 fixture setup. This is the only story with new code.

### Within User Story 3

- Tests MUST be written FIRST and verified to FAIL before implementation
- Implementation modifies `src/price_predictor/infrastructure/mtgjson_loader.py`
- Verification runs full test suite after implementation

### Parallel Opportunities

- **Phase 1 + Phase 2 + Phase 3**: T001, T002, and T003 can all run in parallel (independent tasks)
- **Phase 4**: T004 and T005 can run in parallel (different test classes)
- **Phase 5**: T008, T009, and T010 are independent and can run in parallel

---

## Implementation Strategy

### MVP First (User Story 3 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 4: User Story 3 (T004–T007)
3. **STOP and VALIDATE**: Run `cd src && pytest` — all tests pass, exclusion count logged
4. Feature is functional with the single new behavior

### Incremental Delivery

1. T001 → Fixture ready
2. T002–T003 → US1/US2 verified: existing CardMarket EUR behavior confirmed
3. T004–T007 → US3 complete: exclusion count logging (the only new code!)
4. T008–T010 → Polish: lint, integration, docs validation

---

## Notes

- Primary file modified: `src/price_predictor/infrastructure/mtgjson_loader.py` (US3 only — one log line)
- Test file modified: `tests/unit/infrastructure/test_mtgjson_loader.py` (US3 only — 2 new tests)
- Fixture file modified: `tests/fixtures/allprintings_sample.json` (setup only — one new card entry)
- No new modules, dependencies, or domain entity changes
- FR-001–FR-006, FR-009, FR-010 are formalization of existing behavior — no code changes
- FR-008 (exclusion count logging) is the sole new code change in this feature
- Feature 003's €0.01 price floor applies to zero-priced cards (FR-007) — already implemented, no change needed
