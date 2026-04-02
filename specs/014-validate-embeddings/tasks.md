# Tasks: Validate Card Embeddings

**Input**: Design documents from `/specs/014-validate-embeddings/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Single user story (P1) — tasks organized as foundational prerequisites followed by the story implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No new project setup needed — this feature adds to the existing `sealed` module structure.

No tasks in this phase.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add `compute_mana_value()` to the shared mana scorer module. This function is needed by the embedding probe ground truth extraction and must exist before US1 implementation begins.

**⚠️ CRITICAL**: US1 implementation depends on this function being available.

- [X] T001 Add `compute_mana_value(cost_str: str) -> float` function to `src/sealed/domain/mana_scorer.py` that parses brace-format mana cost strings (e.g. `{2}{W}{W}` → 4.0). Rules: colored/colorless pips → +1, generic `{N}` → +N, hybrid/phyrexian → +1, `{X}` → +0. Follow the same brace-parsing pattern as `_accumulate_pips()`. See research.md R1 for full rules.
- [X] T002 Add unit tests for `compute_mana_value()` in `tests/unit/sealed/domain/test_mana_scorer.py`. Test cases: simple colored (`{W}` → 1), generic+colored (`{2}{W}{W}` → 4), hybrid (`{G/R}` → 1), phyrexian (`{W/P}` → 1), variable (`{X}{R}` → 1), colorless (`{C}` → 1), multi-generic (`{4}{B}{B}{B}` → 7), empty string → 0.

**Checkpoint**: `compute_mana_value()` works and is tested. US1 can begin.

---

## Phase 3: User Story 1 — Run Embedding Validation (Priority: P1) 🎯 MVP

**Goal**: A researcher can run `python -m sealed validate-embeddings --cards-path output/cardsfolder/` to verify that card embeddings encode the features Stage 2 training depends on, receiving per-probe pass/fail results and an overall exit code.

**Independent Test**: Run `pytest tests/unit/sealed/domain/test_embedding_probe.py tests/unit/sealed/infrastructure/test_cli_sealed_validate.py tests/integration/sealed/test_validate_embeddings_integration.py` — all tests pass.

### Tests for User Story 1 (MANDATORY per Constitution) ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T003 [P] [US1] Write unit tests for ground truth extraction functions in `tests/unit/sealed/domain/test_embedding_probe.py`. Cover: `extract_is_land()` (land vs non-land text), `extract_card_color()` per color (single-color, multi-color, colorless cards), `extract_pip_counts()` per color (zero pips, fractional hybrid pips), `extract_mana_value()` (delegates to `compute_mana_value` via mana cost line), `extract_mana_produced()` per color (lands, mana rocks like Sol Ring, non-mana-producing cards → 0). Use synthetic `CardData` objects with hand-written card text strings — no real embeddings needed.
- [X] T004 [US1] Write unit tests for probe runner in `tests/unit/sealed/domain/test_embedding_probe.py` (same file as T003 — implement sequentially after T003). Cover: `build_default_probes()` returns 20 ProbeSpecs with correct names/types/thresholds, `build_default_probes()` with custom threshold overrides, `run_probes()` returns ProbeResult list with correct pass/fail status. Mock `cross_val_score` to avoid real sklearn fitting in unit tests.
- [X] T005 [P] [US1] Write unit tests for CLI argument parsing in `tests/unit/sealed/infrastructure/test_cli_sealed_validate.py`. Cover: `validate-embeddings` subcommand is recognized by the parser, `--cards-path` / `--threshold-accuracy` / `--threshold-r2` arguments are parsed correctly with defaults, handler returns exit code 2 when cards-path does not exist.

### Implementation for User Story 1

- [X] T006 [P] [US1] Implement value objects (`CardData`, `ProbeSpec`, `ProbeResult`, `ValidationResult`) and ground truth extraction functions (`extract_is_land`, `extract_card_color`, `extract_pip_counts`, `extract_mana_value`, `extract_mana_produced`) in `src/sealed/domain/embedding_probe.py`. Extraction functions take `list[CardData]` and return `np.ndarray`. Reuse `count_pips()` and `count_actual_sources()` from `sealed.domain.mana_scorer` for parsing. Use type-line check (same logic as `EmbeddingAdapter.is_land()`) for land detection. See data-model.md for entity fields and ground truth extraction table.
- [X] T007 [US1] Implement probe runner (`build_default_probes`, `run_probes`) in `src/sealed/domain/embedding_probe.py`. `build_default_probes(threshold_accuracy, threshold_r2)` returns a list of 20 `ProbeSpec` objects covering all 5 categories (is-land, card color ×6, pip counts ×6, mana value, mana produced ×6) with thresholds per FR-007 (is-land uses `max(threshold_accuracy, 0.99)`, mana value uses `max(threshold_r2, 0.90)`). `run_probes(cards, probes)` runs each probe using `cross_val_score` with `StratifiedKFold(5)` for classification and `KFold(5, shuffle=True)` for regression. See research.md R2.
- [X] T008 [US1] Implement `ValidateEmbeddingsUseCase` in `src/sealed/application/validate_embeddings.py`. The `execute(cards_path, threshold_accuracy, threshold_r2)` method: discovers `.npz` files via `Path.rglob("*.npz")`, pairs each with `.txt` file at same path (exclude cards missing either file), loads embeddings via `np.load(path)["embedding"]` and text via `Path.read_text()`, raises `ValueError` if fewer than 50 paired cards, counts lands among loaded cards (check type line for "land") and includes as `n_lands` in result, builds probe specs via `build_default_probes()`, runs probes, returns `ValidationResult`. See data-model.md relationships diagram.
- [X] T009 [US1] Add `validate-embeddings` subcommand and `run_validate_embeddings()` handler to `src/sealed/infrastructure/cli.py`. Add subparser with `--cards-path` (default `output/cardsfolder/`), `--threshold-accuracy` (float, default 0.95), `--threshold-r2` (float, default 0.85). Handler: validates cards-path exists (exit 2 if not), calls use case, prints result table (Feature / Score / Threshold / Status per probe), prints summary line with total cards and overall PASS/FAIL, returns exit code 0 or 1. Wire into `main()` dispatcher. See research.md R5 for output format.
- [X] T010 [US1] Write integration test in `tests/integration/sealed/test_validate_embeddings_integration.py`. Create a `tmp_path` fixture directory with ~100 synthetic cards (random 512-dim embeddings as `.npz`, hand-written `.txt` files covering lands, colored spells, artifacts with mana abilities). Run `ValidateEmbeddingsUseCase.execute()` end-to-end with real sklearn fitting. Verify it returns a `ValidationResult` with 20 `ProbeResult` entries. Verify exit code logic (all_passed reflects actual probe outcomes). This test exercises the full pipeline: file discovery → loading → ground truth extraction → cross-validation → result aggregation. Also include a test case that constructs cards with random noise embeddings (e.g., `np.random.default_rng(42).standard_normal((512,))`) paired with realistic card texts, asserts that `validation_result.all_passed` is `False` and that at least 3 of the 5 probe categories contain a failing probe, confirming the validation rejects meaningless embeddings (SC-002).

**Checkpoint**: User Story 1 is fully functional and testable. `python -m sealed validate-embeddings --cards-path output/cardsfolder/` works end-to-end.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Final quality checks.

- [X] T011 Run `ruff check .` from `src/` and fix any lint issues in new files
- [ ] T012 Run quickstart.md validation — execute the basic usage command against real card data and verify output matches expected format

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Empty — no work needed
- **Foundational (Phase 2)**: No dependencies — can start immediately. BLOCKS Phase 3.
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion (T001, T002)
- **Polish (Phase 4)**: Depends on Phase 3 completion

### Within Phase 2 (Foundational)

- T001 → T002 (implement before testing)

### Within Phase 3 (User Story 1)

- T003, T005 can run in parallel (different files, no dependencies)
- T004 follows T003 (same file)
- T006 can run in parallel with tests (different file)
- T007 depends on T006 (value objects and extraction functions must exist)
- T008 depends on T007 (use case calls probe runner)
- T009 depends on T008 (CLI handler calls use case)
- T010 depends on T008 (integration test exercises use case)

### Parallel Opportunities

```
Phase 2:    T001 → T002

Phase 3:    ┌─ T003 (unit: extraction)  ─┐
            │    └─ T004 (unit: runner)  ─┤
            ├─ T005 (unit: CLI)          ─┤
            └─ T006 (impl: extraction)   ─┘
                        │
                      T007 (impl: probe runner)
                        │
                      T008 (impl: use case)
                       / \
                    T009   T010
                  (CLI)    (integration test)
```

---

## Parallel Example: User Story 1

```text
# After Phase 2 completes, launch tests and first implementation in parallel:
Task T003: "Unit tests for ground truth extraction in tests/unit/sealed/domain/test_embedding_probe.py"
Task T004: "Unit tests for probe runner in tests/unit/sealed/domain/test_embedding_probe.py"
Task T005: "Unit tests for CLI in tests/unit/sealed/infrastructure/test_cli_sealed_validate.py"
Task T006: "Implement value objects + extraction in src/sealed/domain/embedding_probe.py"

# Then sequential:
Task T007: "Implement probe runner in src/sealed/domain/embedding_probe.py"
Task T008: "Implement use case in src/sealed/application/validate_embeddings.py"

# Then parallel again:
Task T009: "Add CLI subcommand to src/sealed/infrastructure/cli.py"
Task T010: "Integration test in tests/integration/sealed/test_validate_embeddings_integration.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (`compute_mana_value()`)
2. Complete Phase 3: User Story 1 (all probes + CLI + tests)
3. **STOP and VALIDATE**: Run `python -m sealed validate-embeddings --cards-path output/cardsfolder/` against real data
4. Verify exit code 0 (all probes pass) with production encoder

### Incremental Delivery

This feature has a single user story — delivery is atomic. The validation either works or it doesn't. No partial delivery makes sense.

---

## Notes

- [P] tasks = different files, no dependencies
- [US1] = maps to User Story 1: "Run Embedding Validation"
- All ground truth extraction reuses existing mana_scorer parsers — do NOT reimplement parsing logic
- Unit tests use synthetic CardData with hand-written text — no real embeddings or sklearn fitting in fast suite
- Integration test uses real sklearn fitting on synthetic data — separated per constitution Principle I
- The `compute_mana_value()` function is the ONLY addition to existing code outside the new feature files
