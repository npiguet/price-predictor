# Tasks: Sealed Deck Scorer

**Input**: Design documents from `/specs/013-sealed-deck-scorer/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure dependencies are available and project structure is ready for new modules.

- [ ] T001 Verify PyTorch and numpy are in project dependencies; add if missing (check requirements.txt or equivalent)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No blocking foundational tasks needed. US1 is self-contained using existing infrastructure. Shared modules (match data loader, scorer store) are introduced within the user stories that first require them (US2).

**Checkpoint**: Setup complete — US1 implementation can begin.

---

## Phase 3: User Story 1 — Extend Card Encoding with Deterministic Features (Priority: P1) MVP

**Goal**: The `encode-cards` command produces 544-dimensional card vectors: 512-dim text embedding + 32 deterministic game features parsed from converted card text. Existing 512-dim files require `--clean` to re-encode.

**Independent Test**: Run `encode-cards` on a small cards folder and verify each `.npz` file contains a 544-element vector with deterministic features matching manual inspection of the card text.

### Tests for User Story 1 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T002 [P] [US1] Write unit tests for deterministic feature parsing in tests/unit/sealed/domain/test_deterministic_features.py — cover: mana cost breakdown (e.g. `{2}{R}{R}` → red=2, generic=2, mv=4), land detection from types line, devoid overriding color flags to colorless, mana production from activated abilities with `add` patterns, power/toughness/loyalty parsing (`*`/`X` = 0), cards with no mana cost, zero-padding at indices 538-543
- [ ] T003 [P] [US1] Extend card encoder tests for 544-dim output in tests/unit/sealed/domain/test_card_encoder.py — verify: output vector is exactly 544 dimensions, first 512 dims match text embedding, last 32 dims match deterministic features from the card text
- [ ] T004 [P] [US1] Extend encode-cards tests for 544-dim skip logic in tests/unit/sealed/application/test_encode_cards.py — verify: fresh encode produces 544-dim files, existing 544-dim files are skipped, existing 512-dim files are skipped (not auto-upgraded), `--clean` deletes and re-encodes all files to 544-dim

### Implementation for User Story 1

- [ ] T005 [US1] Implement deterministic feature parsing in src/sealed/domain/deterministic_features.py — parse converted card text to produce a 32-element float array following the index specification in data-model.md (is_land, mana cost breakdown W/U/B/R/G/C/generic/X/mv, color flags with devoid handling, mana production from activated abilities, power/toughness/loyalty, zero padding)
- [ ] T006 [US1] Extend card encoder to concatenate 512-dim text embedding with 32 deterministic features into a 544-dim vector in src/sealed/domain/card_encoder.py — call deterministic_features to parse the card text, concatenate with the existing text embedding output
- [ ] T007 [US1] Update encode-cards skip logic to check embedding dimension in src/sealed/application/encode_cards.py — skip cards with any existing `.npz` file regardless of dimension (512 or 544); `--clean` deletes all `.npz` files before encoding; no auto-upgrade from 512 to 544

**Checkpoint**: `encode-cards` produces 544-dim vectors. US1 is independently testable.

---

## Phase 4: User Story 2 — Train Deck Scorer on Match Outcomes (Priority: P2)

**Goal**: Train a Set Transformer scorer on pairwise match outcomes using Bradley-Terry loss. Produces saved model checkpoints with normalization statistics. Includes 80/20 validation split by match for best-checkpoint selection.

**Independent Test**: Run training on a small match outcomes subset, verify training loss decreases, checkpoint saves successfully, and a loaded model can score an arbitrary deck.

### Tests for User Story 2 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T008 [P] [US2] Write unit tests for Set Transformer scorer model in tests/unit/sealed/domain/test_scorer_model.py — cover: forward pass with synthetic data produces scalar output, permutation invariance (same cards in different order produce identical score), padding mask prevents padded cards from influencing score, normalization buffers (feat_mean/feat_std) transform indices 512-543 correctly, shared weights (same model instance scores both decks)
- [ ] T009 [P] [US2] Write unit tests for match data loader in tests/unit/sealed/infrastructure/test_match_data_loader.py — cover: parse match-outcomes.txt format (`deckA_card1|...|card40;deckB_card1|...|card40;winsA;winsB`), identify winner (deck with 2 wins), filter basic lands from deck card lists, look up 544-dim embeddings per card, build PyTorch dataset, variable-length collate function with padding and boolean masks, error on missing card embedding
- [ ] T010 [P] [US2] Write unit tests for scorer checkpoint save/load round-trip in tests/unit/sealed/infrastructure/test_scorer_store.py — cover: save checkpoint dict (model_state_dict, optimizer_state_dict, epoch, best_val_loss, config), load checkpoint and verify all fields match, feat_mean/feat_std buffers survive round-trip, latest.pt vs best.pt file naming
- [ ] T011 [P] [US2] Write unit tests for training use case in tests/unit/sealed/application/test_train_scorer.py — cover: normalization stats computed from training corpus (mean and std of deterministic features), Bradley-Terry loss computation (BCE on score difference with target=1), training step reduces loss on synthetic data, validation split by match (80/20, all games from a match in same split), best checkpoint saved only when validation loss improves, latest checkpoint saved every validation interval

### Implementation for User Story 2

- [ ] T012 [P] [US2] Implement Set Transformer scorer model in src/sealed/domain/scorer_model.py — SAB blocks (nn.MultiheadAttention + feedforward + LayerNorm), PMA pooling with learned seed vectors, scoring MLP (Linear→ReLU→Linear→ReLU→Linear→scalar), register_buffer for feat_mean and feat_std, normalize indices 512-543 in forward pass; use config from CLI contract (n_layers, n_heads, n_seeds, d_ff, mlp_hidden, d_model=544)
- [ ] T013 [P] [US2] Implement match data loader in src/sealed/infrastructure/match_data_loader.py — parse match-outcomes.txt line by line, identify winner/loser from win counts, load card embeddings from .npz files in cards-path, filter out basic land cards, build TrainingExample (winner_cards tensor, loser_cards tensor), implement collate_fn for variable-length batching with padding and boolean masks, raise clear error for missing card embeddings
- [ ] T014 [P] [US2] Implement scorer checkpoint store in src/sealed/infrastructure/scorer_store.py — save_checkpoint(model, optimizer, epoch, best_val_loss, config, path) using torch.save, load_checkpoint(path) returning dict with all fields, handle latest.pt and best.pt file naming
- [ ] T015 [US2] Implement training use case in src/sealed/application/train_scorer.py — load match outcomes and card embeddings via match_data_loader, split 80/20 by match into train/validation sets, compute per-feature mean/std for indices 512-543 across training set, set normalization buffers on model, training loop: score both decks with shared model, compute Bradley-Terry loss (BCE with logits on score_winner - score_loser), backprop and step, validate every val_interval epochs computing validation loss, save latest.pt after each validation and best.pt when validation loss improves via scorer_store
- [ ] T016 [US2] Add train-scorer CLI subcommand in src/sealed/infrastructure/cli.py — wire all options from CLI contract (--outcomes-path, --cards-path, --checkpoint-dir, --resume, --epochs, --batch-size, --lr, --n-layers, --n-heads, --n-seeds, --d-ff, --mlp-hidden, --val-interval), delegate to train_scorer use case

**Checkpoint**: Scorer trains on match data, saves checkpoints, validation loss drives best-checkpoint selection. US2 is independently testable.

---

## Phase 5: User Story 3 — Track Validation Metrics During Training (Priority: P3)

**Goal**: Add prediction accuracy metric (fraction of held-out matchups where higher-scored deck won) to validation evaluation and report alongside validation loss at each interval.

**Independent Test**: Run training with a validation split and verify prediction accuracy is reported periodically; confirm overfitting is visible as divergence between training and validation loss.

### Tests for User Story 3 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T017 [US3] Extend training tests to verify prediction accuracy metric in tests/unit/sealed/application/test_train_scorer.py — cover: accuracy computed as fraction of validation matchups where score_winner > score_loser, accuracy and validation loss both reported each val_interval, accuracy = 1.0 when model perfectly separates winners from losers on synthetic data

### Implementation for User Story 3

- [ ] T018 [US3] Add prediction accuracy metric to validation evaluation in src/sealed/application/train_scorer.py — during validation, for each matchup compute scores for both decks, count matchups where the actual winner's score > actual loser's score, report accuracy = correct_predictions / total_predictions
- [ ] T019 [US3] Add structured periodic console reporting of training loss, validation loss, and prediction accuracy in src/sealed/application/train_scorer.py — print at each val_interval: epoch, training loss, validation loss, prediction accuracy; format for easy reading and comparison across epochs

**Checkpoint**: Validation metrics (loss + accuracy) visible during training; overfitting detectable by divergence. US3 is independently testable with the training pipeline from US2.

---

## Phase 6: User Story 4 — Evaluate Scorer Against Forge Baseline (Priority: P4)

**Goal**: End-to-end evaluation pipeline: Python generates pools, builds deck A via scorer-guided greedy search, Java workers build deck B via Forge's SealedDeckBuilder and play best-of-3 matches, Python collects and reports aggregate win rate.

**Independent Test**: Run evaluation with a trained checkpoint and verify it produces a win rate percentage and completes within ~4 minutes for 20 pools.

### Tests for User Story 4 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T020 [P] [US4] Write unit tests for greedy deck search and result aggregation in tests/unit/sealed/application/test_evaluate_scorer.py — cover: greedy search starts from 23 random non-land cards, evaluates all single-card swaps, applies best-improving swap, stops when no swap improves score, basic land computation from non-land color requirements, result aggregation across multiple outcome files (sum wins/games, compute win rate)
- [ ] T021 [P] [US4] Write unit tests for evaluation connector in tests/unit/sealed/infrastructure/test_evaluation_connector.py — cover: worker command construction (java -cp, correct main class, correct args), match file splitting across N workers, outcome file path derivation ({input}-outcomes.txt)
- [ ] T022 [P] [US4] Write JUnit tests for ValidationMatchPlayer in forge-connector/src/test/java/com/pricepredictor/connector/ValidationMatchPlayerTest.java — cover: parse validation-matches line format (deck_A|cards;pool_B|cards), write outcome as wins_A;wins_B, crash recovery (skip matches already in outcomes file by comparing line counts)

### Implementation for User Story 4

- [ ] T023 [P] [US4] Implement ValidationMatchPlayer in forge-connector/src/main/java/com/pricepredictor/connector/ValidationMatchPlayer.java — read validation matches file line by line, for each match: parse deck A card names and pool B card names from pipe-separated format, build deck B from pool B using Forge's SealedDeckBuilder, play best-of-3 via GamePlayer, append `winsA;winsB` to outcomes file; on startup check outcomes file line count and skip already-played matches
- [ ] T024 [US4] Implement ValidationWorkerMain in forge-connector/src/main/java/com/pricepredictor/connector/ValidationWorkerMain.java — Java entry point that initializes Forge environment and delegates to ValidationMatchPlayer (depends on T023)
- [ ] T025 [P] [US4] Implement evaluation connector in src/sealed/infrastructure/evaluation_connector.py — split validation matches file across N workers, construct java command per worker, launch worker subprocesses, wait for completion, retry failed workers (re-launch; worker skips completed matches via outcomes file line count), collect outcome file paths
- [ ] T026 [US4] Implement greedy deck search in src/sealed/application/evaluate_scorer.py — start from random 23 non-land cards from pool, iteratively try all single-card swaps (one non-land out, one pool card in), score each candidate deck with the model, apply swap with highest score improvement, recompute basic lands after each swap (fill remaining slots to 40 total — i.e. `40 - len(non_land_cards)` basics — distributed proportional to color pips of the non-land cards), stop when no swap improves the score
- [ ] T027 [US4] Implement evaluation orchestration in src/sealed/application/evaluate_scorer.py — generate fresh pools (reuse existing pool generation from generate-pools), build deck A per pool using greedy search from T026, write validation matches files (deck_A as pipe-separated 40 card names ; pool_B as pipe-separated full pool), launch workers via evaluation_connector, collect all outcome files, aggregate results (pools evaluated, total games, win rate), print summary to console
- [ ] T028 [US4] Add evaluate-scorer CLI subcommand in src/sealed/infrastructure/cli.py — wire all options from CLI contract (--checkpoint, --cards-path, --pools, --workers, --work-dir), delegate to evaluate_scorer use case

**Checkpoint**: Full evaluation pipeline runs: generate pools, build scorer-guided decks, play matches against Forge, report win rate. US4 is independently testable with a trained checkpoint from US2.

---

## Phase 7: User Story 5 — Resume Training with Embedding Fine-Tuning (Priority: P5)

**Goal**: After initial training with frozen card representations (Phase A), resume with embeddings unfrozen at a reduced learning rate. Report embedding drift to monitor fine-tuning stability.

**Independent Test**: Resume from a Phase A checkpoint with embeddings unfrozen, verify training continues, and confirm embedding drift metrics are reported.

### Tests for User Story 5 (MANDATORY per Constitution)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T029 [P] [US5] Extend training tests to verify embedding unfreezing, differential learning rates, and drift metrics in tests/unit/sealed/application/test_train_scorer.py — cover: with --unfreeze-embeddings, embedding parameters have requires_grad=True; without it, requires_grad=False; optimizer uses separate parameter group for embeddings with --embedding-lr; drift metric computed as mean L2 distance from initial embedding values; resumed training preserves existing model weights

### Implementation for User Story 5

- [ ] T030 [US5] Add --unfreeze-embeddings and --embedding-lr support in src/sealed/application/train_scorer.py — when unfrozen: set requires_grad=True on embedding lookup, create optimizer with separate parameter groups (scorer params at --lr, embedding params at --embedding-lr); when frozen (default): exclude embeddings from optimizer or set requires_grad=False; wire --unfreeze-embeddings and --embedding-lr CLI flags already defined in T016
- [ ] T031 [US5] Add embedding drift metric reporting in src/sealed/application/train_scorer.py — at training start, snapshot initial embedding values; at each validation interval (when unfrozen), compute mean L2 distance of current embeddings from snapshot; report drift alongside other metrics in console output

**Checkpoint**: Embedding fine-tuning works with differential learning rates and drift monitoring. US5 is independently testable by resuming from a Phase A checkpoint.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and cleanup.

- [ ] T032 Run quickstart.md validation — execute the full workflow (encode-cards --clean, train-scorer Phase A, optionally Phase B, evaluate-scorer) and verify expected outcomes match success criteria (SC-001 through SC-006)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: N/A (no blocking prerequisites)
- **US1 (Phase 3)**: Depends on Setup only — can start immediately after Phase 1
- **US2 (Phase 4)**: Depends on US1 (needs 544-dim embeddings to exist)
- **US3 (Phase 5)**: Depends on US2 (extends the training pipeline)
- **US4 (Phase 6)**: Depends on US2 (needs a trained model checkpoint)
- **US5 (Phase 7)**: Depends on US2 (resumes from a Phase A checkpoint)
- **Polish (Phase 8)**: Depends on US1-US5 completion

### User Story Dependencies

```text
Phase 1: Setup
    │
    v
Phase 3: US1 (encode-cards 544-dim) ──────── MVP stop point
    │
    v
Phase 4: US2 (train scorer) ─────────┬───── Core training works
    │                                 │
    v                                 v
Phase 5: US3 (validation metrics)  Phase 7: US5 (embedding fine-tuning)
    │                                 │
    v                                 │
Phase 6: US4 (Forge evaluation) ◄────┘
    │
    v
Phase 8: Polish
```

Note: US4 and US5 can run in parallel after US2 completes. US3 is a prerequisite for US4 in practice (meaningful evaluation requires validation metrics), but not a strict code dependency.

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Domain modules before application modules
- Infrastructure modules can parallel domain modules (different files)
- Application use case depends on domain + infrastructure
- CLI wiring depends on application use case

### Parallel Opportunities

**Phase 3 (US1)**: T002, T003, T004 can run in parallel (different test files)
**Phase 4 (US2)**: T008-T011 can all run in parallel (different test files); T012, T013, T014 can all run in parallel (different source files); T015 depends on T012-T014; T016 depends on T015
**Phase 5 (US3)**: Sequential (T017 → T018 → T019), all in same files as US2
**Phase 6 (US4)**: T020, T021, T022 can all run in parallel (different test files, different languages); T023 and T025 can run in parallel (Java vs Python, different files); T024 depends on T023; T026 depends on T012 (scorer model); T027 depends on T025, T026; T028 depends on T027
**Phase 7 (US5)**: Sequential (T029 → T030 → T031), extending US2 files

---

## Parallel Example: User Story 2

```bash
# Launch all US2 tests in parallel (4 different test files):
Task T008: "scorer model tests in tests/unit/sealed/domain/test_scorer_model.py"
Task T009: "match data loader tests in tests/unit/sealed/infrastructure/test_match_data_loader.py"
Task T010: "scorer store tests in tests/unit/sealed/infrastructure/test_scorer_store.py"
Task T011: "training use case tests in tests/unit/sealed/application/test_train_scorer.py"

# Then launch all US2 domain/infrastructure modules in parallel (3 different source files):
Task T012: "scorer_model.py in src/sealed/domain/"
Task T013: "match_data_loader.py in src/sealed/infrastructure/"
Task T014: "scorer_store.py in src/sealed/infrastructure/"

# Then sequentially:
Task T015: "train_scorer.py" (depends on T012, T013, T014)
Task T016: "CLI subcommand" (depends on T015)
```

## Parallel Example: User Story 4

```bash
# Launch all US4 tests in parallel (3 different test files, 2 languages):
Task T020: "evaluate_scorer tests in tests/unit/sealed/application/"
Task T021: "evaluation_connector tests in tests/unit/sealed/infrastructure/"
Task T022: "ValidationMatchPlayer JUnit tests in forge-connector/"

# Then launch Java and Python infrastructure in parallel:
Task T023: "ValidationMatchPlayer.java in forge-connector/"
Task T025: "evaluation_connector.py in src/sealed/infrastructure/"

# Then sequentially:
Task T024: "ValidationWorkerMain.java" (depends on T023)
Task T026: "greedy deck search in evaluate_scorer.py" (depends on scorer model T012)
Task T027: "evaluation orchestration in evaluate_scorer.py" (depends on T025, T026)
Task T028: "CLI subcommand" (depends on T027)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 3: User Story 1 (encode-cards 544-dim)
3. **STOP and VALIDATE**: Run `encode-cards --clean` on a small cards folder, verify 544-dim output
4. This is the minimum useful increment — 544-dim embeddings ready for training

### Incremental Delivery

1. US1 → 544-dim card vectors ready
2. US2 → Scorer trains, checkpoints saved, validation loss tracked
3. US3 → Prediction accuracy visible, overfitting detectable
4. US4 → Forge baseline evaluation running, win rate measured
5. US5 → Embedding fine-tuning with stability monitoring
6. Each story adds value without breaking previous stories

### Single Developer Strategy

Work stories in priority order (P1 → P2 → P3 → P4 → P5). Within each story, write tests first, then implement domain, infrastructure, application, and CLI in that order. Commit after each task or logical group.

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The scorer model (T012) is the most architecturally complex task — reference research.md for SAB/PMA/Bradley-Terry decisions
- The match data loader (T013) must handle variable-length decks and basic land filtering — reference data-model.md for the MatchOutcome and TrainingExample schemas
- The greedy search (T026) is the most algorithmically complex task — reference FR-019 and parent spec for the full search procedure
