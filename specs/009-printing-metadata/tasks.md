# Tasks: Printing Data Fields in Training & Prediction

**Input**: Design documents from `/specs/009-printing-metadata/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/rest-api.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Test fixture preparation and shared constants

- [ ] T001 Update test fixture `tests/fixtures/allprintings_sample.json` to include `isReserved`, `rarity`, `legalities`, `printings`, and `setCode` fields for all sample cards
- [ ] T002 [P] Add enriched card text fixtures in `tests/fixtures/converted_cards_training/` — create at least 2 sample `.txt` files that include the 5 printing data lines appended at the end

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Domain model, metadata extraction, parser, and enrichment module that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T003 [P] Create `PrintingData` frozen dataclass in `src/price_predictor/domain/value_objects.py` with fields: `is_reserved` (bool), `rarity` (str), `printings_count` (int), `set_code` (str), `legalities` (list[str]). Add validation, `defaults()` classmethod, and `RECOGNIZED_FORMATS` constant tuple. See data-model.md for field specs.
- [ ] T004 [P] Add unit tests for `PrintingData` in `tests/unit/domain/test_printing_data.py` — test construction, validation (invalid rarity, negative printings_count), `defaults()` factory, and `RECOGNIZED_FORMATS` contents
- [ ] T005 Add optional `printing_data: PrintingData | None = None` field to `Card` entity in `src/price_predictor/domain/entities.py`
- [ ] T006 Extend `build_name_to_uuids` in `src/price_predictor/infrastructure/mtgjson_loader.py` to also return a `dict[str, dict]` mapping UUID to card metadata (`rarity`, `setCode`, `isReserved`, `legalities`, `printings` list). Return as a second value from an updated function or a new `build_name_to_uuids_with_metadata` function.
- [ ] T007 Modify `build_price_map` in `src/price_predictor/infrastructure/mtgjson_loader.py` to also track which UUID produced the cheapest price per card. Return `dict[str, tuple[float, str]]` (price, cheapest_uuid) — or add a new `build_price_map_with_uuids` function to avoid breaking existing callers.
- [ ] T008 Add new `build_metadata_map` function in `src/price_predictor/infrastructure/mtgjson_loader.py` that combines name-to-UUID mapping, UUID metadata, and cheapest-UUID tracking to produce `dict[str, PrintingData]` — one `PrintingData` per card name, derived from the cheapest printing's data. Handle `isReserved` absent = False, legalities filtering to `RECOGNIZED_FORMATS`, rarity fallback to "rare" for missing/unknown values.
- [ ] T009 [P] Add unit tests for metadata map building in `tests/unit/infrastructure/test_mtgjson_loader.py` — test `build_metadata_map` with sample fixture data: reserved card, multi-printing card, card with no rarity, card with online-only format legalities, card banned in all formats
- [ ] T010 Verify that `parse_converted_text` in `src/price_predictor/infrastructure/converted_card_parser.py` handles enriched card text (with 5 printing data lines appended) without errors — no code changes needed, the parser already ignores unrecognized key:value lines. The parser does NOT populate `Card.printing_data`; all printing data extraction is handled by `card_enrichment.py` (T012).
- [ ] T011 [P] Add backward-compatibility test in `tests/unit/infrastructure/test_converted_card_parser.py` — verify parsing card text with printing data lines present produces the same Card entity (same name, types, mana cost, oracle text, ability count) as parsing without them
- [ ] T012 Create `src/price_predictor/application/card_enrichment.py` with: (a) `enrich_card_text(text: str, printing_data: PrintingData) -> str` that appends the 5 printing data lines at the end of card text; (b) `extract_printing_fields_from_text(text: str) -> dict[str, str]` that detects which printing data keys are already present in the text and returns their values; (c) `extract_printing_data_from_text(text: str) -> PrintingData | None` that parses all 5 fields from the text into a `PrintingData` instance (returns None if no fields present); (d) `enrich_or_default(text: str, metadata_map: dict[str, PrintingData]) -> str` that parses the card name from text, looks it up in the metadata map, merges client-provided fields with auto-filled or default values, and returns enriched text. Note: the card text parser (`converted_card_parser.py`) does NOT handle printing data — all extraction and enrichment is centralized here.
- [ ] T013 [P] Add unit tests for card enrichment in `tests/unit/application/test_card_enrichment.py` — test `enrich_card_text` produces correct format, `extract_printing_fields_from_text` finds existing fields, `enrich_or_default` auto-fills for known card, applies defaults for unknown card, and respects client-provided overrides

**Checkpoint**: Foundation ready — all domain objects, metadata extraction, parsing, and enrichment logic are in place. User story implementation can begin.

---

## Phase 3: User Story 1 — Enrich Training Data with Printing Metadata (Priority: P1) MVP

**Goal**: Both sklearn and transformer training pipelines enrich card text with the 5 printing data fields during data loading. All training cards have metadata appended. Models are retrained on enriched format.

**Independent Test**: Run data loading pipeline and verify that output card text for known cards (e.g., Black Lotus, Lightning Bolt) includes correct metadata fields matching AllPrintings.json.

### Tests for User Story 1 (MANDATORY per Constitution)

- [ ] T014 [P] [US1] Add unit tests for printing data features in `tests/unit/application/test_feature_engineering.py` — test that a Card with `printing_data` produces 17 additional dense features (is_reserved, rarity one-hot, printings_count, legalities_count, 10 format multi-hot), and that a Card with `printing_data=None` produces 17 zeros
- [ ] T015 [P] [US1] Add unit tests for enriched sklearn training in `tests/unit/application/test_train.py` — test that `TrainModelUseCase.execute` produces Cards with `printing_data` populated, and that the feature matrix has 17 additional columns
- [ ] T016 [P] [US1] Add unit tests for enriched transformer training in `tests/unit/application/test_train_transformer.py` — test that `_match_cards_to_texts` returns enriched text strings containing metadata lines
- [ ] T017 [P] [US1] Add integration test for enriched training pipeline in `tests/integration/test_end_to_end.py` — train sklearn model on fixture data with metadata, verify model artifact loads and predicts without error

### Implementation for User Story 1

- [ ] T018 [US1] Add 17 printing data features to `FeatureEngineering._transform_single` in `src/price_predictor/application/feature_engineering.py` — is_reserved (1 binary), rarity one-hot (4: common/uncommon/rare/mythic), printings_count (1 numeric), legalities_count (1 numeric), per-format legality multi-hot (10 binary). Update `get_feature_count`. When `card.printing_data` is None, output 17 zeros. See data-model.md for feature spec.
- [ ] T019 [US1] Update `TrainModelUseCase.execute` in `src/price_predictor/application/train.py` — after joining cards to prices, call `build_metadata_map` and attach `PrintingData` to each training Card by constructing a new Card instance with `printing_data` set (Card is frozen). The parser does not set printing_data; this is the only place it gets attached for sklearn training.
- [ ] T020 [US1] Update `_match_cards_to_texts` in `src/price_predictor/application/train_transformer.py` — after matching card name to price, also look up metadata from `build_metadata_map` and call `enrich_card_text` to append metadata lines to the raw text before returning it
- [ ] T021 [US1] Update `EvaluateModelUseCase.execute` in `src/price_predictor/application/evaluate.py` — same enrichment as training: build metadata map, attach `PrintingData` to each eval Card before feature transformation
- [ ] T022 [US1] Update `evaluate_transformer` in `src/price_predictor/application/evaluate_transformer.py` — enrich text with metadata before tokenization, same as training pipeline

**Checkpoint**: Training and evaluation pipelines produce enriched card text with metadata. Models can be retrained. US1 acceptance scenarios 1–7 are testable.

---

## Phase 4: User Story 2 — Predict Known Cards with Auto-Filled Metadata (Priority: P2)

**Goal**: The prediction API (and CLI predict commands) automatically look up known cards in AllPrintings.json and fill in all 5 metadata fields from the cheapest printing before running prediction. No client changes required.

**Independent Test**: Call prediction API with a known card name (no metadata in text) and verify the prediction uses auto-filled metadata from AllPrintings.

### Tests for User Story 2 (MANDATORY per Constitution)

- [ ] T023 [P] [US2] Add integration test for server auto-fill in `tests/integration/test_server_integration.py` — POST known card text (no metadata lines) to `/api/v1/predict`, verify 200 response. Verify that the server internally enriched the text (can test via a mock or by checking that the model received enriched input).
- [ ] T024 [P] [US2] Add unit test for `enrich_or_default` known-card path in `tests/unit/application/test_card_enrichment.py` — verify that for a card name present in the metadata map, all 5 fields are filled from the map values

### Implementation for User Story 2

- [ ] T025 [US2] Update `serve` subparser in `src/price_predictor/infrastructure/cli.py` — add `--printings-path` (default: `resources/AllPrintings.json`) and `--prices-path` (default: `resources/AllPricesToday.json`) arguments
- [ ] T026 [US2] Update `run_serve` in `src/price_predictor/infrastructure/cli.py` — call `build_metadata_map` at startup using the new CLI args, pass the resulting metadata map to `create_app`
- [ ] T027 [US2] Update `create_app` and the `predict` endpoint in `src/price_predictor/infrastructure/server.py` — accept `metadata_map` parameter, store in `app.state.metadata_map`. In the predict handler, after reading the body: (1) call `enrich_or_default(body, metadata_map)` to produce enriched text, (2) parse enriched text to Card via `parse_converted_text`, (3) call `extract_printing_data_from_text(enriched_text)` to get PrintingData and construct a new Card with printing_data set for sklearn, (4) pass enriched text to transformer tokenizer.

**Checkpoint**: Server auto-fills metadata for known cards. Existing API clients work unchanged. US2 acceptance scenarios 1–3 are testable.

---

## Phase 5: User Story 3 — Predict Unknown Cards with Optional Metadata (Priority: P3)

**Goal**: Unknown cards (not in AllPrintings) receive sensible defaults for missing metadata fields. Clients may provide some or all fields inline in the card text, and provided values override defaults.

**Independent Test**: Call prediction API with unknown card text (no AllPrintings match), with and without metadata fields, and verify defaults are applied correctly for missing fields.

### Tests for User Story 3 (MANDATORY per Constitution)

- [ ] T028 [P] [US3] Add integration test for server defaults in `tests/integration/test_server_integration.py` — POST unknown card text (no metadata) to `/api/v1/predict`, verify 200 response with default metadata applied
- [ ] T029 [P] [US3] Add integration test for partial metadata override in `tests/integration/test_server_integration.py` — POST unknown card text with `rarity: mythic` only, verify mythic is used and other 4 fields get defaults
- [ ] T030 [P] [US3] Add unit test for `enrich_or_default` unknown-card path in `tests/unit/application/test_card_enrichment.py` — verify defaults applied when card name not in metadata map
- [ ] T031 [P] [US3] Add unit test for client override in `tests/unit/application/test_card_enrichment.py` — verify that client-provided fields in the text are preserved and not overwritten by auto-fill or defaults

### Implementation for User Story 3

- [ ] T032 [US3] Verify `enrich_or_default` in `src/price_predictor/application/card_enrichment.py` handles the unknown-card path correctly — if card name not found in metadata map, apply `PrintingData.defaults()` for all missing fields. If some fields are client-provided in the text, merge: keep client values, fill remaining from defaults. (This should already be implemented in T012; this task verifies and fixes edge cases.)
- [ ] T033 [US3] Verify known-card override behavior in `src/price_predictor/application/card_enrichment.py` — if a known card has client-provided fields in the text, those override auto-filled values (e.g., `rarity: mythic` overrides the actual rarity). (Same as T032 — verify edge case from spec.)

**Checkpoint**: All three user stories are independently functional and testable. Default values and partial overrides work correctly.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Cleanup, validation, and documentation

- [ ] T034 [P] Update CLI predict commands (`run_predict_sklearn`, `run_predict_transformer`) in `src/price_predictor/infrastructure/cli.py` to support metadata enrichment for file-based and inline card text prediction (auto-fill if AllPrintings available, otherwise pass through)
- [ ] T035 [P] Run `ruff check .` from `src/` and fix any linting issues introduced by this feature
- [ ] T036 Run full test suite (`cd src && pytest`) and fix any failures
- [ ] T037 Run quickstart.md validation — manually verify the training, evaluation, and prediction examples from `specs/009-printing-metadata/quickstart.md` work end-to-end
- [ ] T038 [P] SC-004 accuracy comparison: run `evaluate sklearn` and `evaluate transformer` on pre-metadata models to capture baseline metrics, then retrain both models on enriched data, run evaluate again, and log the before/after comparison. This is informational only — no pass/fail gate.
- [ ] T039 [P] Update project documentation to reflect feature 009 changes: new `--printings-path` and `--prices-path` serve CLI options, enriched card text format with 5 printing data fields, and updated workflow descriptions per Constitution Principle VI

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001, T002) for fixtures — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — no dependency on US2 or US3
- **US2 (Phase 4)**: Depends on Foundational (Phase 2) + US1 (needs enriched-format models to be meaningful, though server code is independent)
- **US3 (Phase 5)**: Depends on Foundational (Phase 2) + US2 (extends the same server enrichment path)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2). No dependency on other stories. This is the MVP — training pipeline enrichment.
- **User Story 2 (P2)**: Can start after Foundational (Phase 2). Server auto-fill is independent of training, but meaningful end-to-end testing requires retrained models from US1.
- **User Story 3 (P3)**: Builds on US2's server enrichment path. The default logic is part of `enrich_or_default` (Foundational), but server integration testing extends US2's work.

### Within Each User Story

- Tests MUST be written first and FAIL before implementation
- Domain/infrastructure changes before application-layer orchestration
- Core implementation before integration testing
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 2 (Foundational)**:
- T003 + T004 (PrintingData + tests) can run in parallel with T001 + T002 (fixtures)
- T006 + T007 (mtgjson_loader) are sequential (T007 depends on T006)
- T009 (loader tests) can run in parallel with T010 + T011 (parser + tests)
- T012 + T013 (enrichment + tests) can run after T003 + T005

**Phase 3 (US1)**:
- T014, T015, T016, T017 (all tests) can run in parallel
- T018 (feature engineering) can run in parallel with T019 (train.py) since they touch different files
- T020 (train_transformer) can run in parallel with T021 (evaluate.py)

**Phase 4 (US2)**:
- T023 + T024 (tests) can run in parallel
- T025 + T026 are sequential (CLI args → startup logic)

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (write-first, should fail):
Task: "T014 — test_feature_engineering.py printing data features"
Task: "T015 — test_train.py enriched training"
Task: "T016 — test_train_transformer.py enriched transformer training"
Task: "T017 — test_end_to_end.py enriched pipeline integration"

# Then launch implementation in parallel where possible:
Task: "T018 — feature_engineering.py (17 new features)"
Task: "T019 — train.py (metadata enrichment)"  # parallel with T018
Task: "T020 — train_transformer.py (text enrichment)"  # after T019 pattern established
Task: "T021 — evaluate.py"  # parallel with T020
Task: "T022 — evaluate_transformer.py"  # parallel with T021
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (fixtures)
2. Complete Phase 2: Foundational (PrintingData, metadata map, parser, enrichment)
3. Complete Phase 3: User Story 1 (training + evaluation enrichment)
4. **STOP and VALIDATE**: Retrain both models, evaluate, compare metrics
5. Commit and verify — enriched training is independently valuable

### Incremental Delivery

1. Setup + Foundational → Domain model and infrastructure ready
2. Add US1 → Retrain models → Evaluate (MVP!)
3. Add US2 → Server auto-fills known cards → Test with API
4. Add US3 → Server defaults unknown cards → Full feature complete
5. Polish → Lint, docs, quickstart validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Card entity is frozen — creating enriched Cards requires constructing new instances with `printing_data` set
- The transformer pipeline uses raw text (not Card entities) — enrichment appends text lines
- The sklearn pipeline uses Card entities — enrichment sets `Card.printing_data` field
- `enrich_or_default` handles both US2 (auto-fill) and US3 (defaults) — the logic is unified
- Models trained without metadata MUST NOT be used with metadata-enriched input (FR-008)
