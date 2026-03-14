# Tasks: Transformer Model Architecture

**Input**: Design documents from `/specs/007-transformer-model-arch/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add PyTorch dependencies, create directory structure, and prepare test fixtures

- [ ] T001 Add `torch` and `transformers` dependencies to pyproject.toml
- [ ] T002 [P] Create test fixture converted card text files in tests/fixtures/converted_cards/ (lightning_bolt.txt, grizzly_bears.txt, jace_the_mind_sculptor.txt)
- [ ] T003 [P] Create models/transformer/ directory with .gitkeep

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core domain entity, PyTorch model, dataset, model store, and evaluation use case that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Tests (write first, ensure they FAIL before implementation)

- [ ] T004 [P] Write TransformerConfig validation tests (all fields > 0, d_model divisible by n_heads, dropout in [0, 1)) in tests/unit/domain/test_entities.py
- [ ] T005 [P] Write CardPriceTransformerModel tests (forward pass shape, CLS extraction, determinism, attention mask handling) in tests/unit/infrastructure/test_transformer_model.py
- [ ] T006 [P] Write TransformerTrainingDataset tests (construction from card tuples, tokenization, padding/truncation, shifted-log target transform) in tests/unit/infrastructure/test_transformer_dataset.py
- [ ] T007 [P] Write TransformerStore tests (save/load roundtrip, config preserved, missing file error) in tests/unit/infrastructure/test_transformer_store.py
- [ ] T008 [P] Write evaluate_transformer use case tests (MAE and median percentage error computation, shifted-log inverse transform, per-card breakdown) in tests/unit/application/test_evaluate_transformer.py

### Implementation

- [ ] T009 Add TransformerConfig frozen dataclass with validation to src/price_predictor/domain/entities.py
- [ ] T010 [P] Create CardPriceTransformerModel (nn.Module) with token embedding, learned positional embedding, 4-layer TransformerEncoder, CLS extraction, and linear output head in src/price_predictor/infrastructure/transformer_model.py
- [ ] T011 [P] Create TransformerTrainingDataset (torch Dataset) with BERT WordPiece tokenization, padding/truncation, and shifted-log price targets in src/price_predictor/infrastructure/transformer_dataset.py
- [ ] T012 Create TransformerStore with save (state_dict + config dict) and load (.pt artifact reconstruction) in src/price_predictor/infrastructure/transformer_store.py
- [ ] T013 Create evaluate_transformer use case (load model, build validation dataset, compute MAE and median percentage error, return TransformerEvalResult) in src/price_predictor/application/evaluate_transformer.py

**Checkpoint**: Foundation ready — all core components tested and working. User story implementation can now begin.

---

## Phase 3: User Story 1 — Train Model on Consumer Hardware (Priority: P1) MVP

**Goal**: A developer can run `price_predictor train-transformer` on a GeForce RTX 3060 Ti (8 GB VRAM) and the model trains to completion, saving a `.pt` artifact and printing per-epoch progress plus auto-evaluation metrics.

**Independent Test**: Run the training CLI on a machine with 8 GB VRAM and verify: (1) no OOM errors, (2) training loss decreases, (3) model artifact saved to models/transformer/model.pt, (4) auto-evaluation metrics printed.

### Tests for User Story 1 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T014 [P] [US1] Write sequence length analysis tests (95th percentile calculation, round-up to multiple of 8, clamp to min 64, CLI logging of distribution stats) in tests/unit/application/test_train_transformer.py
- [ ] T015 [P] [US1] Write train_transformer use case tests (training loop with mocked model on CPU, early stopping logic, best-checkpoint saving, auto-evaluation call, TransformerTrainResult fields) in tests/unit/application/test_train_transformer.py
- [ ] T016 [P] [US1] Write train-transformer CLI argument parsing and invocation tests in tests/unit/infrastructure/test_cli_train_transformer.py

### Implementation for User Story 1

- [ ] T017 [US1] Implement sequence length analysis function: tokenize all card texts, compute 95th percentile, round up to nearest multiple of 8, clamp min 64, log distribution stats (95th/99th/max, truncation %) per research.md §1 in src/price_predictor/application/train_transformer.py
- [ ] T018 [US1] Create train_transformer use case (load cards via forge_parser + mtgjson_loader, match to output/ texts, call sequence length analysis from T017, build dataset, 80/20 split seed 42, AdamW lr=1e-4, linear warmup 2 epochs, early stopping patience 5, save best checkpoint, auto-evaluate, return TransformerTrainResult) in src/price_predictor/application/train_transformer.py
- [ ] T019 [US1] Add `train-transformer` CLI subcommand with all arguments per contracts/cli-subcommands.md, per-epoch summary output, and final summary in src/price_predictor/infrastructure/cli.py

**Checkpoint**: At this point, User Story 1 should be fully functional — `price_predictor train-transformer` produces a trained model artifact with progress output.

---

## Phase 4: User Story 2 — Predict Card Prices from Tokenized Input (Priority: P2)

**Goal**: The API evaluation endpoint loads both the sklearn and transformer models, runs both on every request, and returns dual predictions. The transformer prediction gracefully degrades to `null` if no model artifact exists.

**Independent Test**: Start the server, POST a card script to `/api/v1/evaluate`, and verify the response contains both `sklearn` and `transformer` prediction objects per contracts/evaluate-endpoint.md. Repeat with no transformer model artifact and verify `transformer` is `null`.

### Tests for User Story 2 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T020 [P] [US2] Update server unit tests for dual-model response format (both models present, transformer null when unavailable, response schema per contracts/evaluate-endpoint.md) in tests/unit/infrastructure/test_server.py
- [ ] T021 [P] [US2] Update eval CLI tests for new nested response format (both models displayed, transformer "not available" when null) in tests/unit/infrastructure/test_cli_eval.py

### Implementation for User Story 2

- [ ] T022 [US2] Modify `create_app()` to accept optional transformer model, run both models on evaluate endpoint, return nested response with graceful degradation in src/price_predictor/infrastructure/server.py
- [ ] T023 [US2] Update `eval` CLI subcommand to parse nested response and display both sklearn and transformer predictions in src/price_predictor/infrastructure/cli.py

**Checkpoint**: At this point, the prediction service returns dual-model predictions and the eval CLI displays both.

---

## Phase 5: User Story 3 — Evaluate Model Quality (Priority: P3)

**Goal**: A developer can run `price_predictor evaluate-transformer` to re-evaluate any saved model against the validation dataset and review MAE and median percentage error metrics.

**Independent Test**: Train a model (US1), then run `price_predictor evaluate-transformer` and verify JSON output with MAE, median percentage error, and sample count. Test `--model-path` override and `--output-csv` per-card breakdown.

### Tests for User Story 3 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T024 [P] [US3] Write evaluate-transformer CLI argument parsing and invocation tests (default model path, --model-path override, --output-csv, JSON output format) in tests/unit/infrastructure/test_cli_evaluate_transformer.py

### Implementation for User Story 3

- [ ] T025 [US3] Add `evaluate-transformer` CLI subcommand with all arguments per contracts/cli-subcommands.md, JSON output, and optional CSV export in src/price_predictor/infrastructure/cli.py

**Checkpoint**: All user stories are independently functional — training, API prediction, and standalone evaluation all work.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Integration tests and end-to-end validation across all stories

- [ ] T026 [P] Create integration test: train tiny model on fixture data (3 cards, 2 epochs, CPU) and verify artifact saved and eval metrics returned in tests/integration/test_transformer_training.py
- [ ] T027 [P] Update server integration tests for dual-model responses (load both models, verify nested response) in tests/integration/test_server_integration.py
- [ ] T028 Run quickstart.md end-to-end validation (install deps, train, evaluate, serve, eval CLI)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **User Story 2 (Phase 4)**: Depends on Foundational phase completion (independent of US1)
- **User Story 3 (Phase 5)**: Depends on Foundational phase completion (independent of US1, US2)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2. No dependencies on other stories. Uses evaluate_transformer (Phase 2) for auto-evaluation after training.
- **User Story 2 (P2)**: Can start after Phase 2. No dependencies on other stories. Requires TransformerStore (Phase 2) to load model at server startup.
- **User Story 3 (P3)**: Can start after Phase 2. No dependencies on other stories. Uses evaluate_transformer (Phase 2) — this phase only adds the CLI subcommand.

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Use cases before CLI subcommands
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1**: T002 and T003 can run in parallel
- **Phase 2**: All 5 test tasks (T004–T008) can run in parallel; T010 and T011 can run in parallel after T009
- **Phase 3**: T014, T015, and T016 (tests) can run in parallel; T017 (seq len analysis) before T018 (training use case)
- **Phase 4**: T020 and T021 (tests) can run in parallel
- **Phase 5**: T024 (single test task) is independent
- **Phase 6**: T026 and T027 can run in parallel
- **Cross-phase**: US1, US2, US3 can run in parallel once Phase 2 is complete

---

## Parallel Example: Phase 2 (Foundational)

```bash
# Launch all foundational tests together:
Task T004: "TransformerConfig tests in tests/unit/domain/test_entities.py"
Task T005: "CardPriceTransformerModel tests in tests/unit/infrastructure/test_transformer_model.py"
Task T006: "TransformerTrainingDataset tests in tests/unit/infrastructure/test_transformer_dataset.py"
Task T007: "TransformerStore tests in tests/unit/infrastructure/test_transformer_store.py"
Task T008: "evaluate_transformer tests in tests/unit/application/test_evaluate_transformer.py"

# Then launch parallel implementations:
Task T010: "CardPriceTransformerModel in src/price_predictor/infrastructure/transformer_model.py"
Task T011: "TransformerTrainingDataset in src/price_predictor/infrastructure/transformer_dataset.py"
```

## Parallel Example: User Stories After Phase 2

```bash
# All three user stories can proceed in parallel after Phase 2:
Stream A (US1): T014+T015+T016 → T017 → T018 → T019
Stream B (US2): T020+T021 → T022 → T023
Stream C (US3): T024 → T025
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1 — Train Model
4. **STOP and VALIDATE**: Run `price_predictor train-transformer` on RTX 3060 Ti, verify training completes within 8 GB VRAM and < 10 minutes
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Train model -> Validate on GPU (MVP!)
3. Add User Story 2 -> API serves dual predictions -> Test via eval CLI
4. Add User Story 3 -> Standalone evaluation -> Test with --model-path override
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (training pipeline)
   - Developer B: User Story 2 (API + eval CLI changes)
   - Developer C: User Story 3 (evaluate CLI subcommand)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The evaluate_transformer use case is in Phase 2 (Foundational) because it is used by US1 (auto-evaluation after training) and US3 (standalone CLI)
- max_seq_len is determined empirically in T017 (sequence length analysis) — see research.md §1 for the full algorithm (95th percentile, round to multiple of 8, clamp min 64, stored in artifact)
