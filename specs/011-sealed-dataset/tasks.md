# Tasks: Sealed Dataset Preparation

**Input**: Design documents from `/specs/011-sealed-dataset/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/cli.md ✅, quickstart.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `sealed` package skeleton so all subsequent tasks have a consistent import root.

- [ ] T001 Create sealed package directory structure with all `__init__.py` files: `src/sealed/__init__.py`, `src/sealed/domain/__init__.py`, `src/sealed/application/__init__.py`, `src/sealed/infrastructure/__init__.py`

**Checkpoint**: `python -c "import sealed"` succeeds (empty package).

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: Expose the pooled text representation from the pretrained model — required by `CardEncoder` (US1) before any encoding can happen.

**⚠️ CRITICAL**: US1 cannot be implemented until this phase is complete.

- [ ] T002 Add `encode(input_ids, attention_mask) -> Tensor` method to `CardPriceTransformerModel` in `src/price_predictor/infrastructure/transformer_model.py` per the exact implementation in `research.md §2` (returns `cat([max_pooled, mean_pooled])`, decorated with `@torch.no_grad()`)

**Checkpoint**: `model.encode(input_ids, mask)` returns a `(batch, 2*d_model)` tensor without error.

---

## Phase 3: User Story 1 — Encode Card Embeddings (Priority: P1) 🎯 MVP

**Goal**: `python -m sealed encode-cards` reads card scripts, strips `name:` lines, generates embeddings via the pretrained encoder, and writes one `.npz` per card atomically. Already-encoded cards are skipped; a re-run is safe and near-instant.

**Independent Test**: Point the command at a small fixture folder (3–5 card scripts); verify one `.npz` per card is produced; re-run and confirm no files change and zero cards are processed.

### Tests for User Story 1

- [ ] T003 [P] [US1] Write unit tests for `CardEncoder` (name-line stripping, tokenization, output shape) in `tests/unit/sealed/domain/test_card_encoder.py`
- [ ] T004 [P] [US1] Write unit tests for `EmbeddingStore` (atomic save with temp rename, load round-trip, no partial file on simulated interrupt) in `tests/unit/sealed/infrastructure/test_embedding_store.py`
- [ ] T005 [P] [US1] Write unit tests for `EncodeCardsUseCase` (skip-if-exists logic, processed/skipped counts, error accumulation) in `tests/unit/sealed/application/test_encode_cards.py`
- [ ] T006 [P] [US1] Write integration test for `encode-cards` CLI on a small fixture folder: all cards encoded on first run, zero processed on second run, partial folder correctly processes only missing cards in `tests/integration/sealed/test_encode_cards_integration.py`

### Implementation for User Story 1

- [ ] T007 [P] [US1] Implement `CardEncoder` (strip `name:` line, tokenize via `MtgTokenizer`, call `model.encode()`, return `float32` numpy array) in `src/sealed/domain/card_encoder.py`
- [ ] T008 [P] [US1] Implement `EmbeddingStore` (`save` with atomic temp-rename pattern from `research.md §3`, `load` returning `np.load(path)["embedding"]`) in `src/sealed/infrastructure/embedding_store.py`
- [ ] T009 [US1] Implement `EncodeCardsUseCase` (`execute(cards_path, encoder, store) -> EncodeCardsResult` with skip logic, progress every 100 cards via `\r`, error collection) in `src/sealed/application/encode_cards.py`
- [ ] T010 [US1] Implement `encode-cards` subcommand in `src/sealed/infrastructure/cli.py` (argparse with `--encoder-path`, `--vocab-path`, `--cards-path` and defaults per `contracts/cli.md`; validates paths exist; exits with code 2 on fatal error, code 1 if any card failed)
- [ ] T011 [US1] Implement `src/sealed/__main__.py` entry point that dispatches to `encode-cards` (and later `generate-pools`) subcommands

**Checkpoint**: `python -m sealed encode-cards --cards-path <fixture>` encodes cards, prints progress, and exits 0. Re-run reports zero processed.

---

## Phase 4: User Story 2 — Generate Sealed Pools (Priority: P2)

**Goal**: `python -m sealed generate-pools` invokes the forge-connector JAR to produce N sealed pools for a given set code, filters basic lands, and writes `pools.txt` (one pool per line, semicolon-separated). Overwrites any existing file.

**Independent Test**: Generate 10 pools for a known set; verify the output file has exactly 10 lines, each with 84–90 card names and no basic land names; verify an invalid set code exits with a clear error and no output file.

### Tests for User Story 2

- [ ] T012 [P] [US2] Write JUnit 5 unit test for `PoolGenerator` (correct pool count, no basic lands in output, `IllegalArgumentException` for null/unknown set code) in `forge-connector/src/test/java/com/pricepredictor/connector/PoolGeneratorTest.java`
- [ ] T013 [P] [US2] Write unit tests for `GeneratePoolsUseCase` (output directory created if absent, `PoolConnector.generate` called with correct args) in `tests/unit/sealed/application/test_generate_pools.py`

### Implementation for User Story 2

- [ ] T014 [P] [US2] Implement `PoolGenerator` using Forge booster API (`FModel.getMagicDb().getBoosters().get(setCode)` + `UnOpenedProduct.get()`, 6 boosters per pool, `isBasicLand()` filter, null-check for unknown set code) per `research.md §1` in `forge-connector/src/main/java/com/pricepredictor/connector/PoolGenerator.java`
- [ ] T015 [US2] Implement `PoolMain` CLI entry point (args: `--set`, `--size`, `--pools-path`; initializes Forge environment; writes `pools.txt` with semicolon-separated pool lines; streams progress every 1000 pools to stdout per `contracts/cli.md`; exits 1 on error) in `forge-connector/src/main/java/com/pricepredictor/connector/PoolMain.java`
- [ ] T016 [US2] Build updated forge-connector JAR via `mvn package -DskipTests` in `forge-connector/` so `PoolMain` is available to the Python subprocess
- [ ] T017 [P] [US2] Implement `PoolConnector` (resolve JAR path using the same classpath logic as the existing `ConvertMain` invocation, call `java -cp ... com.pricepredictor.connector.PoolMain`, stream stdout line by line) in `src/sealed/infrastructure/pool_connector.py`
- [ ] T018 [US2] Implement `GeneratePoolsUseCase` (`execute(set_code, pool_count, pools_path, connector)` — `mkdir(parents=True)`, delegate to `connector.generate()`) in `src/sealed/application/generate_pools.py`
- [ ] T019 [US2] Add `generate-pools` subcommand to `src/sealed/infrastructure/cli.py` (argparse with `--set`, `--size`, `--pools-path` and defaults per `contracts/cli.md`; resolves `{set}` placeholder in default path; exits 2 on JAR-not-found or invalid set code)

**Checkpoint**: `python -m sealed generate-pools --set RVR --size 10` produces `output/sealed/pools/RVR/pools.txt` with 10 lines, 84–90 names each, no basic lands.

---

## Phase 5: User Story 3 — Incremental Encoding After Encoder Retrain (Priority: P3)

**Goal**: Confirm that deleting a subset of `.npz` files and re-running `encode-cards` re-encodes only those missing files, leaving untouched embeddings unchanged.

**Independent Test**: After a full encode run, delete 2 of the 5 fixture `.npz` files; re-run and verify exactly 2 cards are processed and the remaining 3 are skipped.

*No new implementation is required — this behavior is already delivered by the skip logic in `EncodeCardsUseCase` (US1). This phase adds a targeted test to make the guarantee explicit.*

- [ ] T020 [US3] Add incremental re-encoding test scenario (delete a subset of `.npz` files, re-run encode-cards, assert only deleted cards are re-processed) to `tests/integration/sealed/test_encode_cards_integration.py`

**Checkpoint**: All integration tests pass; incremental re-run reports the expected processed/skipped counts.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and end-to-end validation.

- [ ] T021 Update `README.md` with `python -m sealed` module section: `encode-cards` and `generate-pools` commands, default paths, workflow description (encode first, then generate pools), `.npz` embedding format, `pools.txt` format
- [ ] T022 [P] Run `quickstart.md` validation steps end-to-end on a small fixture set to confirm the documented workflow matches actual behaviour

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — **blocks US1**
- **US1 (Phase 3)**: Depends on Phase 2 — US2 and US3 can start once US1 is complete
- **US2 (Phase 4)**: Depends on Phase 1 only (Java side is independent); Python side depends on Phase 1
- **US3 (Phase 5)**: Depends on Phase 3 (adds a test to the integration test file created there)
- **Polish (Phase 6)**: Depends on all user story phases

### User Story Dependencies

- **US1 (P1)**: Requires Phase 2 (`encode()` method on model) — no dependency on US2 or US3
- **US2 (P2)**: Requires Phase 1 only — fully independent of US1 at the implementation level
- **US3 (P3)**: Requires Phase 3 completion (integration test file exists) — no new code

### Within Each User Story

- Tests (T003–T006, T012–T013, T020) written before implementation
- Domain/infrastructure components (CardEncoder, EmbeddingStore, PoolGenerator, PoolConnector) before use cases
- Use cases before CLI wiring
- CLI before `__main__.py` dispatch

### Parallel Opportunities

- T003, T004, T005, T006 — all US1 test files, no shared state
- T007, T008 — CardEncoder and EmbeddingStore touch different files
- T012, T013 — Java and Python test files are independent
- T014, T017 — PoolGenerator (Java) and PoolConnector (Python) touch different files
- T021, T022 — README update and quickstart validation are independent

---

## Parallel Example: User Story 1

```bash
# Write all US1 tests together (all different files):
Task T003: tests/unit/sealed/domain/test_card_encoder.py
Task T004: tests/unit/sealed/infrastructure/test_embedding_store.py
Task T005: tests/unit/sealed/application/test_encode_cards.py
Task T006: tests/integration/sealed/test_encode_cards_integration.py

# Implement domain + infrastructure together (different files):
Task T007: src/sealed/domain/card_encoder.py
Task T008: src/sealed/infrastructure/embedding_store.py
# Then sequentially:
Task T009: src/sealed/application/encode_cards.py
Task T010: src/sealed/infrastructure/cli.py
Task T011: src/sealed/__main__.py
```

## Parallel Example: User Story 2

```bash
# Write tests and start Java implementation together:
Task T012: forge-connector/.../PoolGeneratorTest.java
Task T013: tests/unit/sealed/application/test_generate_pools.py
Task T014: forge-connector/.../PoolGenerator.java  (Java, independent)
Task T017: src/sealed/infrastructure/pool_connector.py  (Python, independent)
# Then sequentially (Java side):
Task T015: forge-connector/.../PoolMain.java
Task T016: mvn package (build JAR)
# Then sequentially (Python side):
Task T018: src/sealed/application/generate_pools.py
Task T019: src/sealed/infrastructure/cli.py (add generate-pools subcommand)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (add `encode()` to model)
3. Complete Phase 3: User Story 1 (encode-cards command)
4. **STOP and VALIDATE**: Run integration tests; point at a real fixture folder
5. US2 and US3 can follow independently

### Incremental Delivery

1. Phase 1 + 2 → package skeleton + encode() method
2. Phase 3 (US1) → `encode-cards` command working end-to-end
3. Phase 4 (US2) → `generate-pools` command working end-to-end
4. Phase 5 (US3) → incremental re-encode test confirmed
5. Phase 6 → documentation and quickstart validation

### Parallel Team Strategy

With two developers after Phase 1:
- **Developer A**: Phase 2 → Phase 3 (encode-cards, Python-only)
- **Developer B**: Phase 4 Java side (PoolGenerator + PoolMain + JAR build) in parallel with Developer A

---

## Notes

- [P] tasks = different files, no dependencies between them
- [Story] label maps each task to a specific user story for traceability
- US3 requires no implementation — it is a test-only phase confirming existing US1 behaviour
- Atomic write pattern for `.npz` files: write `{stem}.tmp.npz`, then `os.replace()` to `{stem}.npz`
- `np.savez_compressed` appends `.npz` automatically — use `{stem}.tmp.npz` as the temp name to avoid double-extension
- Java `PoolGenerator` uses `ForgeEnvironmentInitializer.initialize()` (already present in forge-connector) before calling `FModel`
- The `{set}` placeholder in `--pools-path` default must be resolved at CLI parse time before passing to the use case
