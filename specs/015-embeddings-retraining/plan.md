# Implementation Plan: Embeddings Retraining with Auxiliary Supervision

**Branch**: `015-embeddings-retraining` | **Date**: 2026-04-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/015-embeddings-retraining/spec.md`

## Summary

Retrain the price-predictor transformer from scratch with 20 auxiliary linear heads that
supervise mana-relevant features (card color, pip counts, mana value, mana production).
The auxiliary heads attach to the existing pooled embedding and are discarded after
training, producing a checkpoint in the same format as the current `latest.pt`. This
forces the encoder to represent features needed by Stage 2 sealed training, which
feature 014 probes confirmed are currently missing.

The implementation wraps the existing `CardPriceTransformerModel` in an
`AuxiliaryTrainingModel` during training, adds a `--aux-lambda` CLI flag, and corrects
the card color label definition (retroactive fix to features 013/014).

## Technical Context

**Language/Version**: Python 3.14+  
**Primary Dependencies**: PyTorch (nn.Module, BCEWithLogitsLoss, ordinal EMD loss), numpy, scikit-learn  
**Storage**: Files — `.pt` checkpoints (unchanged format), `.txt` card files (read-only)  
**Testing**: pytest (unit tests, CPU-only — no GPU required for unit tests)  
**Target Platform**: Windows 11, CUDA GPU (training), CPU (testing)  
**Project Type**: CLI / ML training pipeline  
**Performance Goals**: Training on ~30k card corpus; label pre-computation < 30s  
**Constraints**: Saved checkpoint must be loadable by existing `load_model()`; embedding dimensionality (2×d_model) unchanged  
**Scale/Scope**: ~30k cards, 20 auxiliary heads (13 binary classification, 7 ordinal classification)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Fast Automated Tests | PASS | Unit tests for label extraction, aux model forward pass, loss computation, checkpoint format. All CPU-only, fast. Real card text fixtures per FR-010. |
| II. Simplicity First | PASS | Each aux head is a single `nn.Linear` — simplest possible architecture. Wrapper model composes existing model without modifying it. Lambda is one scalar CLI parameter. |
| III. Data Integrity | PASS | Labels deterministically computed from card text via existing mana_scorer parsers. Standardization stats from training set only (no data leakage). Std floor prevents division-by-near-zero. |
| IV. DDD & Separation | PASS | Label extraction lives in domain layer (`sealed.domain`). Aux head model in infrastructure (`price_predictor.infrastructure`). Training orchestration in application layer. No domain → infrastructure dependency introduced. |
| V. Forge Interoperability | N/A | Checkpoint format unchanged. No API changes. Java stub unaffected. |
| VI. Documentation | PASS | quickstart.md documents the training workflow. README update captured as a task. |

**Re-check after Phase 1**: All gates still PASS. No new abstractions beyond
`AuxiliaryTrainingModel` (justified by clean separation of training-only code). No
infrastructure dependencies introduced in domain layer.

## Project Structure

### Documentation (this feature)

```text
specs/015-embeddings-retraining/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0 output — 6 design decisions
├── data-model.md        # Phase 1 output — entities and label schema
├── quickstart.md        # Phase 1 output — training and validation commands
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (files to modify or create)

```text
src/
├── price_predictor/
│   ├── application/
│   │   └── train_transformer.py      # MODIFY: aux label computation, combined loss, lambda param
│   ├── domain/
│   │   └── entities.py               # (unchanged — TransformerConfig unchanged)
│   └── infrastructure/
│       ├── cli.py                     # MODIFY: add --aux-lambda flag
│       ├── transformer_model.py       # MODIFY: extract _embed(), add AuxiliaryTrainingModel
│       ├── transformer_dataset.py     # MODIFY: optional aux_labels in dataset
│       └── transformer_store.py       # (unchanged — save wrapper.base, not wrapper)
├── sealed/
│   └── domain/
│       ├── embedding_probe.py         # MODIFY: fix card color labels, add compute_aux_labels()
│       └── mana_scorer.py             # (unchanged — parsers reused as-is)

tests/
├── unit/
│   ├── sealed/domain/
│   │   └── test_embedding_probe.py    # ADD: devoid, colorless, compute_aux_labels tests
│   ├── application/
│   │   └── test_train_transformer.py  # ADD: aux label stats, combined loss tests
│   └── infrastructure/
│       ├── test_transformer_model.py  # ADD: _embed, AuxiliaryTrainingModel tests
│       └── test_transformer_dataset.py # ADD: aux_labels in dataset tests
```

**Structure Decision**: All changes fit within the existing DDD-layered structure.
No new packages or directories needed. The `AuxiliaryTrainingModel` class lives in
`transformer_model.py` alongside the base model it wraps.

## Design: Key Implementation Details

### 1. Model Architecture Change

**File**: `src/price_predictor/infrastructure/transformer_model.py`

Refactor `CardPriceTransformerModel`:
- Extract shared embedding logic from `forward()` and `encode()` into `_embed(input_ids, attention_mask) -> Tensor`
- `encode()` becomes `@torch.no_grad()` wrapper around `_embed()`
- `forward()` calls `_embed()` then applies output head

Add `AuxiliaryTrainingModel(nn.Module)`:
- `__init__(base, ordinal_class_maps)`: Stores base model + `nn.ModuleList` of 20 heads. Classification heads are `nn.Linear(2*d_model, 1)`; ordinal heads are `nn.Linear(2*d_model, K)` where K comes from `ordinal_class_maps`.
- `forward(input_ids, attention_mask, meta)`: Calls `base._embed()` for pooled representation, runs price head and all aux heads, returns `(price_pred, [aux_preds])`. Classification predictions have shape `(batch,)`; ordinal predictions have shape `(batch, K)`.

### 2. Card Color Label Correction

**File**: `src/sealed/domain/embedding_probe.py`

Fix `extract_card_color()`:
- Add `_has_devoid(text: str) -> bool`: Scan lines for `static: devoid` pattern
- For colors W/U/B/R/G: if devoid, return 0.0 regardless of pips
- For color C: return 1.0 if card is colorless (all colored pips are zero after devoid override, or no mana cost)

Add `compute_aux_labels(card_text: str) -> np.ndarray`:
- Returns 20-element array in the order defined in data-model.md
- Uses existing `count_pips()`, `compute_mana_value()`, `count_actual_sources()` from `mana_scorer`
- Uses corrected card color logic

### 3. Dataset Extension

**File**: `src/price_predictor/infrastructure/transformer_dataset.py`

Extend `TransformerTrainingDataset.__init__()` with optional `aux_labels` parameter:
- When provided: store as tensor, include in `__getitem__()` output dict under key `"aux_labels"`
- When None: no change to existing behavior (backward compatible)

### 4. Training Loop Modification

**File**: `src/price_predictor/application/train_transformer.py`

Add pre-training setup (when `aux_lambda > 0`):
1. Print "Computing auxiliary labels..." (FR-011)
2. Call `compute_aux_labels()` for each card text → `(n_cards, 20)` array
3. Split labels into train/val sets (same split as price data)
4. Print "Computing class weights and target statistics..." (FR-011)
5. For classification heads (indices 0,1–6,14–19): compute `pos_weight = n_neg / n_pos`
6. For ordinal heads (indices 7–13): use fixed class maps from FR-012/FR-013/FR-014; print class distribution per head
7. Pass labels to dataset, class maps and pos_weights to training loop

Modify `_train_loop()`:
- Accept optional aux parameters: `cls_loss_fns`, `ordinal_class_maps`, `aux_lambda`
- When aux parameters present: compute combined loss `L_price + lambda * sum(L_aux_i)`
  - Classification heads: `BCEWithLogitsLoss` as before
  - Ordinal heads: EMD loss — convert raw float label to class index via `ordinal_class_maps`,
    compute softmax CDF of logits, L1 distance between predicted and true CDFs
- Print aux_loss alongside train_loss and val_loss per epoch
- Early stopping still based on price validation loss only (aux is a means, not the objective)

Save: Extract `model.base` from wrapper before passing to `save_model()`.

### 5. CLI Extension

**File**: `src/price_predictor/infrastructure/cli.py`

Add `--aux-lambda` argument to `train transformer` subparser:
- Type: float, default: 0.04
- Help text: "Weight for auxiliary mana-feature losses (0=disabled, 0.2=recommended starting point)"
- Pass through to `train_transformer()` call

## Complexity Tracking

No constitution violations. No complexity justifications needed.
