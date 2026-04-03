# Data Model: Embeddings Retraining with Auxiliary Supervision

**Feature**: 015-embeddings-retraining  
**Date**: 2026-04-03

## Entities

### AuxiliaryTrainingModel (training-only, not persisted)

A PyTorch `nn.Module` wrapper that composes the existing `CardPriceTransformerModel`
with 20 auxiliary linear heads. Exists only during training.

| Field | Type | Description |
|-------|------|-------------|
| `base` | `CardPriceTransformerModel` | The encoder + price head (persisted after training) |
| `aux_heads` | `nn.ModuleList[nn.Linear]` | 20 linear projections from pooled embedding to scalar |

**Lifecycle**: Created at training start → used during training → `base` extracted and
saved as checkpoint → wrapper discarded.

**Forward pass returns**: `(price_prediction, list[aux_prediction])` where each aux
prediction has shape `(batch_size,)`.

### Auxiliary Label Tensor (pre-computed, not persisted)

A 20-element float vector per card, computed once from card text before training begins.

| Index | Head Name | Type | Source | Value |
|-------|-----------|------|--------|-------|
| 0 | is_land | classification | Type line contains "land" | 0.0 or 1.0 |
| 1 | card_color_W | classification | ≥1 W pip and not devoid | 0.0 or 1.0 |
| 2 | card_color_U | classification | ≥1 U pip and not devoid | 0.0 or 1.0 |
| 3 | card_color_B | classification | ≥1 B pip and not devoid | 0.0 or 1.0 |
| 4 | card_color_R | classification | ≥1 R pip and not devoid | 0.0 or 1.0 |
| 5 | card_color_G | classification | ≥1 G pip and not devoid | 0.0 or 1.0 |
| 6 | card_color_C | classification | Colorless: no colored pips, devoid, or no mana cost | 0.0 or 1.0 |
| 7 | pip_count_W | regression | `count_pips()` for W | float ≥ 0.0 |
| 8 | pip_count_U | regression | `count_pips()` for U | float ≥ 0.0 |
| 9 | pip_count_B | regression | `count_pips()` for B | float ≥ 0.0 |
| 10 | pip_count_R | regression | `count_pips()` for R | float ≥ 0.0 |
| 11 | pip_count_G | regression | `count_pips()` for G | float ≥ 0.0 |
| 12 | pip_count_C | regression | `count_pips()` for C | float ≥ 0.0 |
| 13 | mana_value | regression | `compute_mana_value()` | float ≥ 0.0 |
| 14 | mana_produced_W | classification | Has `{T}: add {W}` ability | 0.0 or 1.0 |
| 15 | mana_produced_U | classification | Has `{T}: add {U}` ability | 0.0 or 1.0 |
| 16 | mana_produced_B | classification | Has `{T}: add {B}` ability | 0.0 or 1.0 |
| 17 | mana_produced_R | classification | Has `{T}: add {R}` ability | 0.0 or 1.0 |
| 18 | mana_produced_G | classification | Has `{T}: add {G}` ability | 0.0 or 1.0 |
| 19 | mana_produced_C | classification | Has `{T}: add {C}` ability | 0.0 or 1.0 |

**Classification indices**: 0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19 (13 heads)  
**Regression indices**: 7, 8, 9, 10, 11, 12, 13 (7 heads)

### AuxiliaryLabelStats (pre-computed, not persisted)

Statistics computed once from the training set before training begins.

| Field | Type | Description |
|-------|------|-------------|
| `pos_weights` | `dict[int, float]` | Per-classification-head pos_weight (neg/pos ratio) |
| `reg_means` | `dict[int, float]` | Per-regression-head mean from training set |
| `reg_stds` | `dict[int, float]` | Per-regression-head std from training set (floor=1.0) |

**Usage**:
- `pos_weights` → passed to `BCEWithLogitsLoss(pos_weight=...)` per classification head
- `reg_means` / `reg_stds` → used to standardize regression targets before MSE loss

## Value Objects

### CardColorLabels (corrected definition)

A 6-element binary vector (W, U, B, R, G, C) derived from a card's mana cost and text.

**Rules** (FR-003):
1. For each color in {W, U, B, R, G}: label = 1 if `count_pips()` returns > 0 for that
   color
2. **Devoid override**: If card text contains `static: devoid` (case-insensitive),
   then W = U = B = R = G = 0 regardless of pips
3. C (colorless) = 1 if ALL of W, U, B, R, G are 0 (after devoid override)
4. Cards with no `mana cost:` line → all pips are 0 → C = 1

**Relationship to pip counts**: Card color is derived FROM pip counts but is not
identical. Pip counts are fractional floats; color labels are binary. A card with hybrid
{G/R} has pip_G = 0.5 and pip_R = 0.5, but color_G = 1 and color_R = 1 (≥1 pip
threshold uses > 0, not ≥ 1).

## Relationships

```
CardPriceTransformerModel
  └──→ AuxiliaryTrainingModel.base (composition, training only)
         └──→ AuxiliaryTrainingModel.aux_heads (20 × nn.Linear)
                └──→ reads from: _embed() pooled output (batch, 2×d_model)

TransformerTrainingDataset
  └──→ aux_labels tensor (optional, shape: n_cards × 20)
         └──→ computed by: compute_aux_labels() from sealed.domain.embedding_probe
                └──→ uses: count_pips(), compute_mana_value(), count_actual_sources()
                      from sealed.domain.mana_scorer

AuxiliaryLabelStats
  └──→ computed from: training-set slice of aux_labels tensor
  └──→ consumed by: loss functions in _train_loop()
```

## Checkpoint Format (unchanged)

The saved `.pt` file contains exactly the same structure as before:

```python
{
    "state_dict": CardPriceTransformerModel.state_dict(),  # encoder + price head only
    "config": asdict(TransformerConfig),                    # unchanged
}
```

The 20 auxiliary head weights are **not saved**. They are discarded by saving only
`wrapper.base` (the inner `CardPriceTransformerModel`) rather than the full
`AuxiliaryTrainingModel`.
