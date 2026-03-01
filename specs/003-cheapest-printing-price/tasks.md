# Tasks: Cheapest Printing Price for Training

**Input**: Design documents from `/specs/003-cheapest-printing-price/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Test Fixtures)

**Purpose**: Add edge-case test data to existing fixtures so US1/US2 tests have the data they need.

- [ ] T001 Add edge-case price entries to test fixtures: add a zero-price UUID (`ffffffff-0000-0000-0000-000000000000` with price `0.00`) and a sub-cent UUID (`ffffffff-0001-0001-0001-000000000001` with price `0.005`) to `tests/fixtures/allprices_sample.json`, and add corresponding card entries ("Zero Price Card" and "Sub Cent Card") to `tests/fixtures/allprintings_sample.json` in new set blocks with `availability: ["paper"]`, `isFunny: false`, `isOnlineOnly: false`, `language: "English"`

**Checkpoint**: Fixtures updated — both JSON files parse correctly, existing tests still pass.

---

## Phase 2: User Story 1 — Use Cheapest Printing Price for Training (Priority: P1) 🎯 MVP

**Goal**: When a card has multiple printings, select the cheapest price. Apply a €0.01 floor to prevent log-domain errors. Allow zero-priced printings into comparison.

**Independent Test**: Provide a card with a €0.00 price printing, run price map building, verify the returned price is €0.01. Provide a card with multiple printings at different prices, verify the cheapest is selected.

**Covers**: FR-001, FR-002, FR-003, FR-004, FR-005, FR-006

### Tests for User Story 1 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T002 [P] [US1] Write tests for price floor in `get_cheapest_price()` in `tests/unit/infrastructure/test_mtgjson_loader.py`: (a) zero-price UUID returns 0.01, (b) sub-cent price UUID returns 0.01, (c) normal price above 0.01 is unchanged, (d) existing `test_cheapest_across_printings` still returns 0.10 (unchanged — already above floor)
- [ ] T003 [P] [US1] Write tests for price floor in `build_price_map()` in `tests/unit/infrastructure/test_mtgjson_loader.py`: (a) card with only a zero-price printing gets 0.01 in price map, (b) card with sub-cent printing gets 0.01 in price map, (c) card with mix of zero and normal prices gets the normal price (if it is the minimum above floor), (d) existing multi-printing cards (Grizzly Bears, Lightning Bolt) retain their current cheapest prices

### Implementation for User Story 1

- [ ] T004 [US1] Update `get_cheapest_price()` and `build_price_map()` in `src/price_predictor/infrastructure/mtgjson_loader.py`: change the price filter from `if price and price > 0` to `if price is not None and price >= 0` in both functions' inner loops, and apply `max(min(all_prices), 0.01)` as the price floor before returning/storing the result. In `get_cheapest_price()` apply floor on line returning `min(all_prices)`. In `build_price_map()` apply floor on the line `result[name] = min(all_prices)`.
- [ ] T005 [US1] Run full test suite (`cd src && pytest`) — verify T002/T003 tests pass and all existing tests in `test_mtgjson_loader.py`, `test_train.py`, `test_train_logging.py`, and `test_end_to_end.py` still pass with no regressions

**Checkpoint**: Price floor working. Zero-price printings included in comparison and clamped. All existing behavior for normal prices preserved.

---

## Phase 3: User Story 2 — Training Data Transparency for Price Selection (Priority: P2)

**Goal**: Log per-card price selection details and a summary count of multi-printing cards to stderr at INFO level during `build_price_map()`.

**Independent Test**: Build a price map from test fixtures (which include multi-printing cards like Grizzly Bears and Lightning Bolt), capture log output, verify per-card lines and summary line are present with correct counts.

**Covers**: FR-007, FR-008

### Tests for User Story 2 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T006 [P] [US2] Write tests for per-card price selection logging (FR-008) in `tests/unit/infrastructure/test_mtgjson_loader.py`: using `caplog` at `logging.INFO` level for logger `price_predictor.infrastructure.mtgjson_loader`, call `build_price_map()` with fixtures, verify (a) a log message containing "Grizzly Bears" is emitted (multi-printing card), (b) the message includes the selected price and total price count, (c) no per-card log is emitted for single-printing cards like "Serra Angel"
- [ ] T007 [P] [US2] Write tests for summary logging (FR-007) in `tests/unit/infrastructure/test_mtgjson_loader.py`: using `caplog`, call `build_price_map()` with fixtures, verify (a) a summary log message is emitted containing the word "summary" (case-insensitive) and the count of cards that had multiple price points, (b) the count matches the actual number of multi-printing cards in the fixture data

### Implementation for User Story 2

- [ ] T008 [US2] Add per-card and summary logging in `build_price_map()` in `src/price_predictor/infrastructure/mtgjson_loader.py`: inside the `for name, uuids` loop, after computing `all_prices`, when `len(all_prices) > 1` log `logger.info("  %s: selected €%.2f from %d prices", name, selected_price, len(all_prices))` and increment a `multi_printing_count` counter. After the loop, before the existing "Loaded price data" log, add `logger.info("Price selection summary: %d of %d cards had multiple price points", multi_printing_count, len(result))`
- [ ] T009 [US2] Run full test suite (`cd src && pytest`) — verify T006/T007 tests pass and all existing tests (including `test_train_logging.py` and `test_end_to_end.py`) still pass with no regressions

**Checkpoint**: Multi-printing transparency logging working. Per-card detail lines and summary line appear on stderr during training.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup across all changes

- [ ] T010 Run ruff linting (`cd src && ruff check .`) and fix any style or lint issues introduced by this feature
- [ ] T011 Run end-to-end integration tests (`cd src && pytest tests/integration/`) to verify the full train → predict → evaluate pipeline works with updated price logic
- [ ] T012 Validate that `specs/003-cheapest-printing-price/quickstart.md` log format examples match the actual log output format implemented in T008

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **User Story 1 (Phase 2)**: Depends on Phase 1 (needs fixture data for tests)
- **User Story 2 (Phase 3)**: Depends on Phase 2 (logging references the price floor logic and `all_prices` collection)
- **Polish (Phase 4)**: Depends on Phase 2 and Phase 3

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Setup (Phase 1) — no dependency on US2
- **User Story 2 (P2)**: Depends on US1 being complete — US2 adds logging to the same `build_price_map()` function modified by US1. The logging references `all_prices` and the floor-clamped price, both introduced by US1.

### Within Each User Story

- Tests MUST be written FIRST and verified to FAIL before implementation
- Implementation modifies `src/price_predictor/infrastructure/mtgjson_loader.py`
- Verification runs full test suite after implementation

### Parallel Opportunities

- **Phase 1**: Single task — no parallelism
- **Phase 2**: T002 and T003 can run in parallel (test different functions)
- **Phase 3**: T006 and T007 can run in parallel (test different log types)
- **Phase 4**: T010, T011, and T012 are independent and can run in parallel

---

## Parallel Example: User Story 1

```text
# Launch test tasks in parallel (different test classes, no file conflicts):
Task T002: "Write tests for price floor in get_cheapest_price()"
Task T003: "Write tests for price floor in build_price_map()"

# Then implementation (sequential — same file):
Task T004: "Update both functions with new filter and floor"
Task T005: "Run full test suite"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: User Story 1 (T002–T005)
3. **STOP and VALIDATE**: Run `cd src && pytest` — all tests pass, price floor works
4. Feature is functional without transparency logging

### Incremental Delivery

1. T001 → Fixtures ready
2. T002–T005 → US1 complete: cheapest price with €0.01 floor (MVP!)
3. T006–T009 → US2 complete: multi-printing transparency logging
4. T010–T012 → Polish: lint, integration, docs validation

---

## Notes

- Primary file modified: `src/price_predictor/infrastructure/mtgjson_loader.py` (both user stories)
- Test file modified: `tests/unit/infrastructure/test_mtgjson_loader.py` (both user stories)
- Fixture files modified: `tests/fixtures/allprices_sample.json`, `tests/fixtures/allprintings_sample.json` (setup only)
- No new modules, dependencies, or domain entity changes
- Existing `TrainingExample` entity validation (`actual_price_eur > 0`) is compatible with the €0.01 floor — no change needed
- The `get_cheapest_price()` standalone function gets the floor but NOT the logging (per research R-004)
