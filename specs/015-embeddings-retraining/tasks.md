# Tasks: Embeddings Retraining with Auxiliary Supervision

**Input**: Design documents from `/specs/015-embeddings-retraining/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Set up test fixtures with real card files (FR-010) and verify baseline.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T001 Copy representative real card text files from output/cardsfolder/ into tests/fixtures/cards/ for use as test fixtures (FR-010). Include: a land (e.g., forest), a red spell, a multicolor card, a hybrid mana card, a devoid card (e.g., world_breaker), a card with {C} in cost, a generic-only cost card (e.g., {3}), a mana dork (e.g., llanowar_elves), and an artifact mana producer (e.g., sol_ring). Add a helper in tests/conftest.py to load fixture card text by filename.

**Checkpoint**: Test infrastructure ready — user story implementation can begin.

---

## Phase 2: User Story 2 — Correct Card Color Label Definition (Priority: P1)

**Goal**: Fix card color labels to follow MTG rules: devoid cards are colorless, no-mana-cost cards are colorless, "C" label means "card is colorless" (not "has {C} pips").

**Independent Test**: Verify label generation against known cards per spec acceptance scenarios 1–6.

### Tests for User Story 2

- [ ] T002 [US2] Write tests for corrected card color labels in tests/unit/sealed/domain/test_embedding_probe.py. Use real card fixture files. Test cases from spec: (1) {2}{R} → R=1, others 0; (2) {C}{C} → C=1, colors 0; (3) {3} generic only → C=1; (4) land with no mana cost → C=1; (5) devoid card with colored pips → C=1, W/U/B/R/G=0; (6) hybrid {G/R} → G=1, R=1, C=0. Also test that pip counts are unaffected by devoid (devoid only changes color labels, not pip counts).

### Implementation for User Story 2

- [ ] T003 [US2] Add `_has_devoid(text: str) -> bool` helper and fix `extract_card_color()` in src/sealed/domain/embedding_probe.py. The `_has_devoid` function scans card text lines for `static: devoid` (case-insensitive). For colors W/U/B/R/G: return 0.0 if devoid, else check pip count > 0 as before. For color C: return 1.0 if all W/U/B/R/G are 0 (after devoid override), i.e., card is colorless. Update existing test expectations if any relied on the old incorrect C behavior.

**Checkpoint**: Card color labels are correct. Existing probe tests still pass.

---

## Phase 3: User Story 1 — Train Encoder with Mana-Aware Embeddings (Priority: P1) MVP

**Goal**: Retrain the transformer with 20 auxiliary heads so embeddings encode card color, pip counts, mana value, and mana production. All 20 probes must pass after training.

**Independent Test**: Run training with `--aux-lambda 0.2`, then run `validate-embeddings`. All 20 probes must pass.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T004 [P] [US1] Write tests for `compute_aux_labels()` in tests/unit/sealed/domain/test_embedding_probe.py. Use real card fixture files. Verify: returns 20-element ndarray; is_land label correct for land/non-land; card color labels use corrected definition (consistent with US2); pip counts match `count_pips()` output; mana value matches `compute_mana_value()`; mana produced labels correct for land, mana dork, and non-mana card.
- [ ] T005 [P] [US1] Write tests for `_embed()` method and `AuxiliaryTrainingModel` in tests/unit/infrastructure/test_transformer_model.py. Test: `_embed()` output shape is (batch, 2*d_model); `_embed()` produces same values as old `encode()` when no_grad applied; `AuxiliaryTrainingModel.forward()` returns tuple of (price_pred, list[aux_preds]); price_pred shape is (batch,); each aux_pred shape is (batch,); aux_heads count equals n_aux parameter; gradients flow through aux heads back to encoder.
- [ ] T006 [P] [US1] Write tests for aux_labels in `TransformerTrainingDataset` in tests/unit/infrastructure/test_transformer_dataset.py. Test: when aux_labels=None, __getitem__ returns dict without "aux_labels" key (backward compat); when aux_labels provided, __getitem__ includes "aux_labels" tensor of correct shape; aux_labels values round-trip correctly through dataset.
- [ ] T007 [P] [US1] Write tests for auxiliary label statistics computation and combined loss in tests/unit/application/test_train_transformer.py. Test: pos_weight computation (num_neg / num_pos) for known label distributions; regression standardization with known mean/std; std floor of 1.0 when variance is near zero; combined loss equals L_price + lambda * sum(L_aux_i) for known inputs.
- [ ] T008 [P] [US1] Write test for `--aux-lambda` CLI argument in tests/unit/infrastructure/test_cli_train_transformer.py. Test: flag is parsed as float; default is 0.0; value is passed to train_transformer() call.

### Implementation for User Story 1

- [ ] T009 [P] [US1] Add `compute_aux_labels(card_text: str) -> np.ndarray` to src/sealed/domain/embedding_probe.py. Returns a 20-element float array in the order defined in data-model.md (is_land, card_color×6, pip_count×6, mana_value, mana_produced×6). Uses existing `count_pips()`, `compute_mana_value()`, `count_actual_sources()` from mana_scorer and the corrected card color logic from T003. Reuse `_is_land_text()` and `_has_devoid()` internally.
- [ ] T010 [P] [US1] Extract `_embed()` method from `CardPriceTransformerModel` in src/price_predictor/infrastructure/transformer_model.py. Move the shared embedding computation (token embed → position embed → dropout → transformer encoder → max pool → mean pool → concat) into `_embed(input_ids, attention_mask) -> Tensor`. Make `encode()` a `@torch.no_grad()` wrapper that calls `_embed()`. Make `forward()` call `_embed()` then apply output dropout + output head. Verify no logic duplication between encode() and forward().
- [ ] T011 [US1] Add `AuxiliaryTrainingModel(nn.Module)` to src/price_predictor/infrastructure/transformer_model.py. Constructor takes `base: CardPriceTransformerModel` and `n_aux: int = 20`. Creates `nn.ModuleList` of `n_aux` `nn.Linear(2 * d_model, 1)` heads. `forward(input_ids, attention_mask, meta)` calls `base._embed()` for pooled representation, runs `base.output_dropout` + `base.output_head` for price prediction, runs each aux head on the raw pooled embedding, returns `(price_pred, [aux_pred_0, ..., aux_pred_19])`.
- [ ] T012 [P] [US1] Extend `TransformerTrainingDataset` with optional aux_labels in src/price_predictor/infrastructure/transformer_dataset.py. Add `aux_labels: torch.Tensor | None = None` parameter to `__init__()`. Store as instance attribute. In `__getitem__()`, include `"aux_labels": self.aux_labels[idx]` in returned dict when aux_labels is not None. No change to existing behavior when aux_labels is None.
- [ ] T013 [US1] Add auxiliary label pre-computation and statistics to `train_transformer()` in src/price_predictor/application/train_transformer.py. When `aux_lambda > 0`: (1) print progress per FR-011; (2) call `compute_aux_labels()` for each card text; (3) split labels with the same train/val split; (4) compute `pos_weight = n_neg / n_pos` for each of 13 classification head indices (0,1-6,14-19); (5) compute mean and std (floor=1.0) for each of 7 regression head indices (7-12,13); (6) standardize regression columns in both train and val label arrays; (7) pass label tensors to dataset constructors. Add `aux_lambda: float = 0.0` parameter to function signature.
- [ ] T014 [US1] Modify `_train_loop()` for combined loss with auxiliary heads in src/price_predictor/application/train_transformer.py. When aux parameters are provided: (1) wrap base model in `AuxiliaryTrainingModel`; (2) create 13 `BCEWithLogitsLoss` with per-head `pos_weight` and 1 `MSELoss` for regression heads; (3) in training step, compute `L_total = L_price + aux_lambda * sum(L_aux_i)`; (4) in validation step, compute and log aux_loss alongside price loss; (5) early stopping remains based on price validation loss only; (6) after training, extract `model.base` from wrapper before returning; (7) print aux_loss per epoch alongside train/val loss. Pass base model (not wrapper) to `save_model()`.
- [ ] T015 [US1] Add `--aux-lambda` CLI flag to `train transformer` subparser in src/price_predictor/infrastructure/cli.py. Type: float, default: 0.0. Help: "Weight for auxiliary mana-feature losses (0=disabled, 0.2=recommended starting point)". Pass value through to `train_transformer()` in the `run_train_transformer_new()` function.

**Checkpoint**: Training with `--aux-lambda > 0` produces a checkpoint with mana-aware embeddings. All 20 probes pass. Checkpoint loadable by existing code.

---

## Phase 4: User Story 3 — Tune Auxiliary Loss Weight (Priority: P2)

**Goal**: Find the lambda value that makes all 20 probes pass while preserving price prediction accuracy.

**Independent Test**: Train with different lambda values and compare probe scores and price validation loss across runs.

**Note**: This story requires no new code — the `--aux-lambda` flag built in US1 provides the tuning mechanism. Tasks below are validation and documentation.

- [ ] T016 [US3] Verify quickstart.md lambda tuning workflow is accurate and complete in specs/015-embeddings-retraining/quickstart.md. Ensure it documents: how to run training with different lambda values, how to validate with probe commands, expected console output format, and guidance on when to increase vs. decrease lambda.

**Checkpoint**: Lambda tuning workflow is documented and reproducible.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation.

- [ ] T017 Run full test suite (`cd src && pytest`) and fix any regressions across both price_predictor and sealed packages
- [ ] T018 Run `ruff check .` from src/ and fix any lint issues in modified files
- [ ] T019 [P] Update README.md with new `--aux-lambda` training workflow, referencing quickstart.md for detailed instructions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — start immediately
- **US2 (Phase 2)**: Depends on Phase 1 (needs card fixtures for tests)
- **US1 (Phase 3)**: Depends on Phase 2 (needs corrected card color labels)
- **US3 (Phase 4)**: Depends on Phase 3 (needs working aux training)
- **Polish (Phase 5)**: Depends on Phases 2–4

### User Story Dependencies

- **US2 (P1)**: Start after Phase 1 — no dependency on other stories. BLOCKS US1.
- **US1 (P1)**: Start after US2 — uses corrected `extract_card_color()` and needs `_has_devoid()` for `compute_aux_labels()`
- **US3 (P2)**: Start after US1 — manual tuning using the training pipeline built in US1

### Within User Story 1

```
T004, T005, T006, T007, T008  ← all tests in parallel (different files)
         │
         ▼
T009, T010, T012              ← parallel implementation (different files)
    │       │
    ▼       ▼
   T013 ← T011                ← T011 depends on T010 (same file, _embed→AuxModel)
    │                            T013 depends on T009+T012 (needs labels + dataset)
    ▼
   T014                        ← depends on T011+T013 (needs AuxModel + label setup)
    │
    ▼
   T015                        ← depends on T014 (CLI wires to complete train fn)
```

### Parallel Opportunities

**Phase 2 (US2)**: T002 (test) first, then T003 (implementation) — sequential
**Phase 3 (US1) tests**: T004, T005, T006, T007, T008 — all parallel (5 different files)
**Phase 3 (US1) implementation**: T009, T010, T012 — parallel (3 different files); then T011, T013 — sequential; then T014, T015 — sequential

---

## Parallel Example: User Story 1

```text
# Launch all US1 tests in parallel (5 tasks, 5 different files):
T004: tests/unit/sealed/domain/test_embedding_probe.py (compute_aux_labels tests)
T005: tests/unit/infrastructure/test_transformer_model.py (_embed + AuxModel tests)
T006: tests/unit/infrastructure/test_transformer_dataset.py (aux_labels tests)
T007: tests/unit/application/test_train_transformer.py (stats + loss tests)
T008: tests/unit/infrastructure/test_cli_train_transformer.py (CLI flag test)

# Then launch parallel implementation (3 tasks, 3 different files):
T009: src/sealed/domain/embedding_probe.py (compute_aux_labels)
T010: src/price_predictor/infrastructure/transformer_model.py (_embed + AuxModel)
T012: src/price_predictor/infrastructure/transformer_dataset.py (aux_labels)
```

---

## Implementation Strategy

### MVP First (User Story 2 + User Story 1)

1. Complete Phase 1: Foundational (card fixtures)
2. Complete Phase 2: User Story 2 (correct card color labels)
3. Complete Phase 3: User Story 1 (auxiliary training)
4. **STOP and VALIDATE**: Run `validate-embeddings` against retrained encoder
5. All 20 probes must pass

### Incremental Delivery

1. Phase 1 → Fixtures ready
2. Phase 2 (US2) → Color labels correct, probes use corrected definitions
3. Phase 3 (US1) → Training with aux heads works, probes pass → **MVP DELIVERED**
4. Phase 4 (US3) → Lambda tuned for optimal probe/price tradeoff
5. Phase 5 → Polished, documented, all tests green

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- FR-010: All new tests must use real card files from fixtures, not inline text strings
- FR-011: Print progress messages for label computation and statistics phases
- Existing `save_model()` and `load_model()` require NO changes — save `wrapper.base` instead of wrapper
- Early stopping in training loop remains based on price validation loss, not auxiliary losses
