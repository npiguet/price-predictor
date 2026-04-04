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
| `aux_heads` | `nn.ModuleList[nn.Linear]` | 20 linear projections from pooled embedding to logits (variable output size per head — see table below) |

**Lifecycle**: Created at training start → used during training → `base` extracted and
saved as checkpoint → wrapper discarded.

**Forward pass returns**: `(price_prediction, list[aux_prediction])` where classification
head predictions have shape `(batch_size,)` and ordinal head predictions have shape
`(batch_size, K)` for K classes.

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
| 7 | pip_count_W | ordinal (11 classes) | `count_pips()` for W | class index in {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| 8 | pip_count_U | ordinal (11 classes) | `count_pips()` for U | class index in {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| 9 | pip_count_B | ordinal (11 classes) | `count_pips()` for B | class index in {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| 10 | pip_count_R | ordinal (11 classes) | `count_pips()` for R | class index in {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| 11 | pip_count_G | ordinal (11 classes) | `count_pips()` for G | class index in {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| 12 | pip_count_C | ordinal (5 classes) | `count_pips()` for C | class index in {0, 1, 2, 2.5, 3} |
| 13 | mana_value | ordinal (17 classes) | `compute_mana_value()` | class index in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} |
| 14 | mana_produced_W | classification | Has `{T}: add {W}` ability | 0.0 or 1.0 |
| 15 | mana_produced_U | classification | Has `{T}: add {U}` ability | 0.0 or 1.0 |
| 16 | mana_produced_B | classification | Has `{T}: add {B}` ability | 0.0 or 1.0 |
| 17 | mana_produced_R | classification | Has `{T}: add {R}` ability | 0.0 or 1.0 |
| 18 | mana_produced_G | classification | Has `{T}: add {G}` ability | 0.0 or 1.0 |
| 19 | mana_produced_C | classification | Has `{T}: add {C}` ability | 0.0 or 1.0 |

**Classification indices**: 0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19 (13 heads, 1 logit each)  
**Ordinal indices**: 7, 8, 9, 10, 11, 12, 13 (7 heads, K logits each)

### Ordinal Class Definitions

| Head index | Head name | Classes | K |
|------------|-----------|---------|---|
| 7–11 | pip_count_W/U/B/R/G | {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} | 11 |
| 12 | pip_count_C | {0, 1, 2, 2.5, 3} | 5 |
| 13 | mana_value | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} | 17 |

**Label storage**: Raw float values (e.g. 1.5 for a hybrid pip) are stored in the auxiliary
label tensor. At loss computation time, raw floats are mapped to class indices using the
class definitions above. Class lists are derived from a full scan of the card corpus — every
value that appears in any card's mana cost line is an explicit class. No overflow classes are
used.

### AuxiliaryLabelStats (pre-computed, not persisted)

Statistics computed once from the training set before training begins.

| Field | Type | Description |
|-------|------|-------------|
| `pos_weights` | `dict[int, float]` | Per-classification-head pos_weight (neg/pos ratio) |
| `ordinal_class_maps` | `dict[int, list[float]]` | Per-ordinal-head sorted class boundary values |

**Usage**:
- `pos_weights` → passed to `BCEWithLogitsLoss(pos_weight=...)` per classification head
- `ordinal_class_maps` → used to convert raw float labels to class indices at loss computation
  time, and to construct `nn.Linear(2*d_model, K)` heads with the correct output size

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

## Meta Vector Layout (30 elements)

The meta vector passed to the regression head during price prediction training is 30-dimensional.
It is constructed by `metadata_encoder.py` as:

```
meta_vector = cat([encode_metadata(printing_data), encode_mana_features(card_text)])
```

| Indices | Group | Source | Size |
|---------|-------|--------|------|
| 0–14 | Printing features | `encode_metadata(PrintingData)` | 15 |
| 15–29 | Mana features | `encode_mana_features(card_text)` via `normalize_mana_features(extract_mana_features(...))` | 15 |

**Mana feature layout within indices 15–29**:

| Offset | Feature | Raw source | Normalization |
|--------|---------|------------|---------------|
| 0 | pip_W | `count_pips()["W"]` | ÷ 8 |
| 1 | pip_U | `count_pips()["U"]` | ÷ 8 |
| 2 | pip_B | `count_pips()["B"]` | ÷ 8 |
| 3 | pip_R | `count_pips()["R"]` | ÷ 8 |
| 4 | pip_G | `count_pips()["G"]` | ÷ 8 |
| 5 | pip_C | `count_pips()["C"]` | ÷ 3 |
| 6 | generic | `count_generic()` | ÷ 15 |
| 7 | x_count | `count_x()` | ÷ 3 |
| 8 | mana_value | `compute_mana_value()` | ÷ 16 |
| 9 | produced_W | `count_mana_produced()["W"]` | ÷ 3, clamp ≤ 1 |
| 10 | produced_U | `count_mana_produced()["U"]` | ÷ 3, clamp ≤ 1 |
| 11 | produced_B | `count_mana_produced()["B"]` | ÷ 3, clamp ≤ 1 |
| 12 | produced_R | `count_mana_produced()["R"]` | ÷ 3, clamp ≤ 1 |
| 13 | produced_G | `count_mana_produced()["G"]` | ÷ 3, clamp ≤ 1 |
| 14 | produced_C | `count_mana_produced()["C"]` | ÷ 3, clamp ≤ 1 |

## Card Embedding Format

Each `.npz` file stores a single key `"embedding"` of shape `(2*d_model + 15,)`, dtype `float32`.

```
embedding = cat([transformer_embedding, mana_features])
```

| Region | Indices | Content | Size |
|--------|---------|---------|------|
| Transformer embedding | 0 .. 2*d_model-1 | max+mean pooled transformer output | 2*d_model |
| Mana features | 2*d_model .. 2*d_model+14 | normalized explicit mana features (same layout as meta indices 15–29 above) | 15 |

The mana features in the embedding use **the same normalization** as the meta vector, ensuring
consistency between training and inference.

When probing the transformer embedding quality, pass `--embed-dim <2*d_model>` to
`validate-embeddings` to slice off the appended mana features.

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

`TransformerConfig.meta_dim` is now **30** (was 15). The checkpoint config reflects this.
