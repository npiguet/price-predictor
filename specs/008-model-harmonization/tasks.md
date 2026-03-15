# Tasks: Model Harmonization

**Input**: Design documents from `/specs/008-model-harmonization/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No setup needed — project already exists with all dependencies installed.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Converted Card Text Parser

- [ ] T001 [P] Create `parse_converted_text()` and `parse_converted_cards()` functions in `src/price_predictor/infrastructure/converted_card_parser.py`. Parse the converted card text format (lowercase `name:`, `mana cost:`, `types:`, `power toughness:`, `loyalty:`, ability lines) into `Card` entities. Use existing `ManaCost.parse()` for mana cost and existing `_classify_types()` logic from `forge_parser.py` for type splitting. Concatenate ability lines into `oracle_text`, count them for `ability_count`. See research.md R1 field mapping table for details.
- [ ] T002 [P] Write unit tests for converted card parser in `tests/unit/infrastructure/test_converted_card_parser.py`. Cover: creature with power/toughness, instant with spell abilities, planeswalker with loyalty, card with multiple ability types, missing optional fields (no mana cost, no P/T), malformed input (missing `name:`, missing `types:`). Reuse existing fixtures from `tests/fixtures/converted_cards/`.

### Model Store Changes

- [ ] T003 [P] Update `transformer_store.py` in `src/price_predictor/infrastructure/transformer_store.py` to use versioned filenames: `save_model()` generates a timestamp version string (reuse `generate_model_version()` from `model_store.py`), saves to `<version>.pt`, and creates a `latest.pt` copy. `load_model()` defaults to loading `latest.pt`. Return `(version, path)` tuple like sklearn's `save_model()`.
- [ ] T004 [P] Update unit tests for transformer store versioning in `tests/unit/infrastructure/test_transformer_store.py`. Cover: save creates versioned file + latest.pt copy, multiple saves create multiple versioned files, load defaults to latest.pt.
- [ ] T005 [P] Verify `model_store.py` in `src/price_predictor/infrastructure/model_store.py` is path-agnostic (no hardcoded `models/` default). If it is, no changes needed — the path change to `models/sklearn/` is handled by callers (train.py, predict.py, evaluate.py, cli.py) in later tasks. If any hardcoded defaults exist, update them.
- [ ] T006 [P] Update unit tests for model store in `tests/unit/infrastructure/test_model_store.py` to verify save/load works correctly with a `models/sklearn/` style path.

### Transformer Predict Use Case

- [ ] T007 [P] Extract transformer inference logic from `src/price_predictor/infrastructure/server.py` (lines ~83-123) into a new `PredictTransformerUseCase` in `src/price_predictor/application/predict_transformer.py`. Accept raw converted card text string, tokenize with BERT, run through model, convert shifted-log output to EUR price. Return `PriceEstimate`.
- [ ] T008 [P] Write unit tests for `PredictTransformerUseCase` in `tests/unit/application/test_predict_transformer.py`. Cover: prediction returns positive EUR price, handles short/long text, model not found raises error.

### CLI Restructure

- [ ] T009 Rewrite `build_parser()` in `src/price_predictor/infrastructure/cli.py` to use nested subparsers: top-level commands `train`, `predict`, `evaluate` (each with `sklearn`/`transformer` sub-subparsers), plus unchanged `serve`, `convert`, `check-convert`. Define all argument groups per the CLI contract in `specs/008-model-harmonization/contracts/cli.md`. Handler functions can initially raise `NotImplementedError` — they will be wired in user story phases.
- [ ] T010 Update command dispatch in `src/price_predictor/__main__.py` to route the new nested subcommand structure (train→sklearn, train→transformer, predict→sklearn, etc.) to their handler functions. Remove dispatch for old commands (`eval`, `train-transformer`, `evaluate-transformer`, old `predict`).

**Checkpoint**: Foundation ready — all infrastructure pieces in place, CLI accepts new command structure.

---

## Phase 3: User Story 1 - Unified Model Training (Priority: P1) 🎯 MVP

**Goal**: `train sklearn` and `train transformer` both work, reading from converted card texts in `./output/` and saving to `./models/<model>/`.

**Independent Test**: Run `train sklearn` and `train transformer` and verify both produce valid model artifacts in their respective subdirectories.

### Tests for User Story 1 (MANDATORY per Constitution) ✅

- [ ] T011 [P] [US1] Update unit tests in `tests/unit/application/test_train.py` to verify sklearn training reads from converted card text files (not Forge scripts) and saves to `models/sklearn/`. Include error cases: empty `./output` folder raises clear error suggesting `convert`, missing prices file raises error.
- [ ] T012 [P] [US1] Update unit tests in `tests/unit/application/test_train_transformer.py` to verify transformer training reads converted texts directly from `./output/` without Forge script dependency and saves to `models/transformer/` with versioned filenames. Include error case: empty `./output` folder raises clear error.

### Implementation for User Story 1

- [ ] T013 [US1] Modify `TrainModelUseCase.execute()` in `src/price_predictor/application/train.py` to replace `parse_forge_cards(forge_cards_path)` with `parse_converted_cards(output_dir)` from the new converted card parser. Update price matching to work with cards parsed from converted text. Change default output path to `models/sklearn/`. Remove `--forge-cards-path` argument dependency.
- [ ] T014 [US1] Modify `train_transformer()` in `src/price_predictor/application/train_transformer.py` to remove the Forge script parsing step. Currently it parses Forge scripts to get card names, then matches to converted text files. Instead, read converted text files directly from `./output/`, extract card names from the `name:` line, and match to prices. Change default model output to `models/transformer/` (already correct, but verify versioned save via updated transformer_store).
- [ ] T015 [US1] Wire `run_train_sklearn()` and `run_train_transformer()` handler functions in `src/price_predictor/infrastructure/cli.py` to call the updated `TrainModelUseCase` and `train_transformer()` respectively. Replace the `NotImplementedError` stubs from T009.

**Checkpoint**: `train sklearn` and `train transformer` both work end-to-end from converted card texts.

---

## Phase 4: User Story 2 - Unified Prediction Command (Priority: P1)

**Goal**: `predict sklearn --file/--card` and `predict transformer --file/--card` both work locally without the REST service.

**Independent Test**: Run `predict sklearn --file output/l/lightning_bolt.txt` and `predict transformer --card "<text>"` and verify both return price estimates.

### Tests for User Story 2 (MANDATORY per Constitution) ✅

- [ ] T016 [P] [US2] Update unit tests in `tests/unit/application/test_predict.py` to verify sklearn prediction accepts a `Card` parsed from converted text (instead of manual attribute input) and loads model from `models/sklearn/latest.joblib`. Include error case: model not found raises clear error.
- [ ] T017 [P] [US2] Write CLI-level tests in `tests/unit/infrastructure/test_cli_predict.py` to verify `predict sklearn --file` reads a file path, `predict sklearn --card` reads inline text, mutual exclusivity of `--file`/`--card`, error on missing arguments, and error when `--file` points to nonexistent file.

### Implementation for User Story 2

- [ ] T018 [US2] Modify `PredictPriceUseCase` in `src/price_predictor/application/predict.py` to accept a `Card` entity directly (parsed from converted text by the caller) instead of building a Card from individual CLI arguments. Update default model path to `models/sklearn/latest.joblib`.
- [ ] T019 [US2] Wire `run_predict_sklearn()` and `run_predict_transformer()` handler functions in `src/price_predictor/infrastructure/cli.py`. For sklearn: read file or inline text → `parse_converted_text()` → `PredictPriceUseCase`. For transformer: read file or inline text → `PredictTransformerUseCase`. Output JSON with `predicted_price_eur` and `model_version`.

**Checkpoint**: `predict sklearn` and `predict transformer` both work locally, no REST service needed.

---

## Phase 5: User Story 3 - Unified Model Evaluation (Priority: P1)

**Goal**: `evaluate sklearn` and `evaluate transformer` both work, reading converted texts and loading models from `./models/<model>/`.

**Independent Test**: Run `evaluate sklearn` and `evaluate transformer` and verify both produce quality metrics.

### Tests for User Story 3 (MANDATORY per Constitution) ✅

- [ ] T020 [P] [US3] Update unit tests in `tests/unit/application/test_evaluate.py` to verify sklearn evaluation reads from converted card texts and loads model from `models/sklearn/`. Include error case: model not found raises clear error.
- [ ] T021 [P] [US3] Update unit tests in `tests/unit/application/test_evaluate_transformer.py` to verify transformer evaluation reads converted texts directly and loads model from `models/transformer/latest.pt`. Include error case: model not found raises clear error.

### Implementation for User Story 3

- [ ] T022 [US3] Modify `EvaluateModelUseCase.execute()` in `src/price_predictor/application/evaluate.py` to replace `parse_forge_cards()` with `parse_converted_cards()`. Update default model path to `models/sklearn/latest.joblib`.
- [ ] T023 [US3] Modify `evaluate_transformer()` in `src/price_predictor/application/evaluate_transformer.py` to remove Forge script dependency, read converted texts directly, and load model from `models/transformer/latest.pt`.
- [ ] T024 [US3] Wire `run_evaluate_sklearn()` and `run_evaluate_transformer()` handler functions in `src/price_predictor/infrastructure/cli.py`. Replace the `NotImplementedError` stubs from T009.

**Checkpoint**: `evaluate sklearn` and `evaluate transformer` both work. All three unified CLI commands (train, predict, evaluate) are functional.

---

## Phase 6: User Story 4 - REST API Accepts Converted Card Format (Priority: P2)

**Goal**: The `/api/v1/predict` endpoint accepts converted card text format and returns predictions from all available models.

**Independent Test**: POST a converted card text body to the endpoint and verify the response contains sklearn (and optionally transformer) price predictions.

### Tests for User Story 4 (MANDATORY per Constitution) ✅

- [ ] T025 [P] [US4] Update server integration tests in `tests/integration/test_server_integration.py` to send converted card text format (not Forge script) and verify response. Test renamed endpoint `/api/v1/predict`.
- [ ] T026 [P] [US4] Update unit tests in `tests/unit/infrastructure/test_server.py` to verify server parses converted card text via `parse_converted_text()` for sklearn and passes raw text for transformer tokenization.

### Implementation for User Story 4

- [ ] T027 [US4] Modify `create_app()` in `src/price_predictor/infrastructure/server.py`: rename endpoint from `/api/v1/evaluate` to `/api/v1/predict`, replace `parse_forge_text()` call with `parse_converted_text()` for sklearn feature extraction, update transformer tokenization to use the raw converted text body (instead of extracting Forge body). Update model loading paths to `models/sklearn/` and `models/transformer/`.
- [ ] T028 [US4] Update `run_serve()` in `src/price_predictor/infrastructure/cli.py` to pass updated default model paths for both sklearn and transformer models.

**Checkpoint**: REST API works with converted card text format. All four user stories are complete.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Cleanup, integration testing, and documentation.

- [ ] T029 Remove all dead code: old handler functions (`run_eval()`, old `run_predict()`, old `run_train()`, old `run_evaluate()`, `run_train_transformer()`, `run_evaluate_transformer()`) from `src/price_predictor/infrastructure/cli.py`.
- [ ] T030 Update integration tests in `tests/integration/test_end_to_end.py` to use new CLI commands (`train sklearn`, `predict sklearn --file`, `evaluate sklearn`).
- [ ] T031 [P] Update `README.md` with new CLI commands, workflows, and model artifact paths per constitution principle VI.
- [ ] T032 Run quickstart.md validation: execute all commands from `specs/008-model-harmonization/quickstart.md` and verify they work.
- [ ] T033 Run full test suite (`cd src; pytest; ruff check .`) and fix any failures.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: N/A — project already exists.
- **Foundational (Phase 2)**: No external dependencies — can start immediately. BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion.
- **User Story 2 (Phase 4)**: Depends on Phase 2 completion. Independent of US1.
- **User Story 3 (Phase 5)**: Depends on Phase 2 completion. Independent of US1/US2.
- **User Story 4 (Phase 6)**: Depends on Phase 2 completion. Independent of US1/US2/US3.
- **Polish (Phase 7)**: Depends on all user stories being complete.

### Within Each User Story

- Tests MUST be written and FAIL before implementation.
- Application layer changes before CLI wiring.
- Story complete before moving to next priority.

### Parallel Opportunities

**Phase 2** (all [P] tasks can run in parallel):
- T001 + T002 (parser) || T003 + T004 (transformer store) || T005 + T006 (model store) || T007 + T008 (predict transformer)
- T009 + T010 depend on the above completing first.

**Phase 3-6** (user stories can run in parallel after Phase 2):
- US1, US2, US3, US4 are all independent of each other.
- Within each story: test tasks marked [P] can run in parallel.

---

## Parallel Example: Phase 2 Foundational

```text
# Launch all infrastructure pieces in parallel:
Task T001: "Create converted_card_parser.py"
Task T003: "Update transformer_store.py versioning"
Task T005: "Update model_store.py default path"
Task T007: "Create predict_transformer.py"

# Then their tests in parallel:
Task T002: "Test converted_card_parser"
Task T004: "Test transformer_store versioning"
Task T006: "Test model_store paths"
Task T008: "Test predict_transformer"

# Then sequential:
Task T009: "Rewrite cli.py build_parser()"
Task T010: "Update __main__.py dispatch"
```

## Parallel Example: User Stories (after Phase 2)

```text
# All four user stories can run in parallel:
US1 (Phase 3): T011-T015  # Training
US2 (Phase 4): T016-T019  # Prediction
US3 (Phase 5): T020-T024  # Evaluation
US4 (Phase 6): T025-T028  # REST API
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
2. Complete Phase 3: User Story 1 — Unified Training
3. **STOP and VALIDATE**: Verify `train sklearn` and `train transformer` work
4. Proceed to remaining stories

### Incremental Delivery

1. Phase 2: Foundation → Infrastructure ready
2. Phase 3: US1 Training → Train commands work (MVP!)
3. Phase 4: US2 Prediction → Predict commands work
4. Phase 5: US3 Evaluation → Evaluate commands work
5. Phase 6: US4 REST API → API updated
6. Phase 7: Polish → Cleanup, docs, full test pass

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
