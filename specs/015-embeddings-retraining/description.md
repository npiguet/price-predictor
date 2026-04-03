# Goal

Retrain the price-predictor transformer from scratch so that its card embeddings encode mana-relevant 
features — card color, pip counts, mana value, and mana production — in addition to the price-predictive features 
they already capture. This is a prerequisite for Stage 2 sealed training, which depends on the model reasoning 
about casting costs and mana base alignment.

The change is purely in the training procedure. The encoder architecture and the stored `.npz` format are
unchanged; the auxiliary heads exist only during training and are discarded afterward.

# Motivation

Feature 014 introduced linear probing to validate that the frozen embeddings used by Stage 2 actually encode the
features it needs. Running the probes against the current encoder revealed that while lands and mana production are
well-encoded, card color and cost structure are not:

| Feature category       | Typical score | Threshold | Status |
|------------------------|---------------|-----------|--------|
| Is land                | 0.992         | ≥ 0.990   | PASS   |
| Card color (W/U/B/R/G) | ~0.91         | ≥ 0.950   | FAIL   |
| Pip counts             | ~0.51         | ≥ 0.850   | FAIL   |
| Mana value             | 0.411         | ≥ 0.900   | FAIL   |
| Mana produced          | ~0.99         | ≥ 0.950   | PASS   |

This is expected: the price predictor was trained to predict card prices, and price is largely determined by a card's
effects, not its mana cost. A powerful effect at `{1}{W}` and a powerful effect at `{3}{W}{W}` are both expensive
because of what the card does. The transformer learned accordingly.

The fix is to give the encoder explicit supervision on mana-relevant features during training, forcing the pooled
representation to carry that information.

> *Note on price prediction quality:* The price-predictor's primary output — card price — is also an indirect encoding 
of card strength and desirability. This signal is expected to become important in Stage 3 of the sealed training 
curriculum, where the model learns to pick good cards rather than just legal or castable ones. Preserving price 
prediction accuracy is therefore not just a sanity check on the retraining; it ensures the embedding retains the
quality signal that Stage 3 depends on.

# Architecture

The transformer encoder and pooling layer are unchanged. During training, 20 lightweight linear heads are attached
to the pooling output — the same 2×d_model-dimensional vector stored in the `.npz` files:

[card text tokens]
↓
[transformer encoder]           (unchanged)
↓
[mean + max pool → 2×d_model-dim]  (unchanged; this is the embedding)
├──→ [price MLP] → predicted price
└──→ [20 auxiliary heads] → 20 feature values

Each auxiliary head is a single linear layer with no activation — a direct linear projection from the 
2×d_model-dim embedding to a scalar output. Using a linear head is intentional: it guarantees that the features 
are linearly decodable from the embedding, which is exactly the property the Stage 2 linear probes test for.

The 20 heads mirror the 20 probes from feature 014:

| Head                             | Type                         | Target                                                     |
|----------------------------------|------------------------------|------------------------------------------------------------|
| Is land                          | Binary classification        | 1 if type line contains "land"                             |
| Card color (× 6: W/U/B/R/G/C)    | Binary classification        | 1 if card has that color (see note below)                  |
| Pip counts W/U/B/R/G             | Ordinal classification (K=11)| Pip count class in {0,0.5,1,1.5,2,2.5,3,4,5,6,8}          |
| Pip count C                      | Ordinal classification (K=5) | Pip count class in {0,1,2,2.5,3}                           |
| Mana value                       | Ordinal classification (K=17)| Mana value class in {0,1,…,16}                             |
| Mana produced (× 6: W/U/B/R/G/C) | Binary classification        | 1 if card has a `{T}: add {color}` ability                 |

# Label Generation

All auxiliary labels are extracted from the card's `.txt` file at training time using the same parsers already 
present in `sealed.domain.mana_scorer`:

- **Is land**: type-line check (same logic as `EmbeddingAdapter.is_land()`)
- **Pip counts**: `count_pips()` on the `mana cost:` line
- **Card color**: derived from pip counts, but **not identical**. A card has color W/U/B/R/G if it has ≥ 1 pip
  of that color. A card is colorless (C = 1) if its mana cost contains no colored pips — i.e., the cost is 
  made exclusively of `{C}`, generic (`{1}`, `{2}`, …), and/or `{X}` pips. Cards with no mana cost at all
  (e.g., lands) are also colorless (C = 1). Cards with the "devoid" keyword 
  ability are colorless regardless of their mana cost (W/U/B/R/G = 0, C = 1, even if colored pips are 
  present). This definition also applies retroactively to features 013 and 014 — if the shared implementation 
  needs to change to support this, that is a correction toward the intended behavior.
- **Mana value**: `compute_mana_value()` on the `mana cost:` line
- **Mana produced**: `count_actual_sources()` scanning `activated[N]: {T}: add` lines

Non-land cards that have tap-for-mana abilities (Sol Ring, Orzhov Signet, Llanowar Elves) receive positive labels
for the colors they produce. A card with no mana ability gets 0 for all six mana-produced heads.

# Training

The model is trained from scratch with the combined loss from epoch 1 (not fine-tuned from an existing checkpoint).

The combined loss is:

L_total = L_price + λ × Σ L_aux_i

Where:
- `L_price` is the existing price prediction loss (unchanged)
- Each classification head uses `BCEWithLogitsLoss`
- Each ordinal head (pip counts and mana value) uses **Earth Mover's Distance (EMD) loss**: the head outputs
  K logits (one per class); the loss is the L1 distance between the CDF of the softmax output and the CDF of
  the one-hot true label: `EMD = sum_{k=0}^{K-2} |CDF_pred(k) - CDF_true(k)|`. This penalises predictions
  proportionally to how many class boundaries they are from the truth — predicting class 3 when truth is 2 is
  penalised less than predicting class 7. Classes are derived from a full scan of the card corpus so no
  overflow class is needed (FR-012/013/014).
- `λ` is a weighting hyperparameter controlling the strength of auxiliary supervision. Starting value: ~0.2.
  The goal is "as low as possible while all 20 probes pass" — if probes fail, increase λ; if price accuracy
  degrades too much, decrease λ.

## Class imbalance handling

Binary classification heads use `pos_weight` in `BCEWithLogitsLoss` to handle imbalanced targets. The weight
is computed once from the training set before training begins:

    pos_weight = num_negatives / num_positives

This ensures that rare-positive heads (e.g., "is land" at ~5%, individual "mana produced" colors) contribute 
meaningful gradient rather than being drowned out by a trivial all-zero prediction.

## Gradient flow and saving

The gradients from the auxiliary heads flow back through the pooling layer and the full transformer, updating 
all encoder weights to encode the supervised features into the pooled representation.

After training, the 20 auxiliary heads are discarded. The saved checkpoint contains the encoder
(transformer + pooling) and the price head, identical in format to the current `latest.pt`. Only the 
auxiliary heads are excluded.

# Validation

After retraining, run feature 014's embedding validation against the new encoder:

```bash
  python -m sealed encode-cards --clean
  python -m sealed validate-embeddings --cards-path output/cardsfolder/
```

The retrained encoder must pass all 20 probes. This is the acceptance criterion — if any probe fails, the auxiliary
supervision was insufficient (likely a loss weighting issue) and training should be repeated with a higher λ.

Price prediction accuracy on the held-out validation set must not degrade unacceptably relative to the feature 
007 baseline. There is no fixed threshold — this is a manual judgment call made after exploring various λ values
and inspecting the trade-off between probe scores and price accuracy. This guards against λ being set so high 
that the auxiliary losses dominate and the encoder forgets how to predict prices.
