# Goal

Retrain the price-predictor transformer so that its card embeddings encode mana-relevant features — card color, 
pip counts, mana value, and mana production — in addition to the price-predictive features they already capture.
This is a prerequisite for Stage 2 sealed training, which depends on the model reasoning about casting costs and 
mana base alignment.

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
to the pooling output — the same 512-dimensional vector stored in the `.npz` files:

[card text tokens]
↓
[transformer encoder]           (unchanged)
↓
[mean + max pool → 512-dim]     (unchanged; this is the embedding)
├──→ [price MLP] → predicted price
└──→ [20 auxiliary heads] → 20 feature values

Each auxiliary head is a single linear layer with no activation — a direct linear projection from the 512-dim 
embedding to a scalar output. Using a linear head is intentional: it guarantees that the features are linearly 
decodable from the embedding, which is exactly the property the Stage 2 linear probes test for.

The 20 heads mirror the 20 probes from feature 014:

| Head                             | Type                  | Target                                                     |
|----------------------------------|-----------------------|------------------------------------------------------------|
| Is land                          | Binary classification | 1 if type line contains "land"                             |
| Card color (× 6: W/U/B/R/G/C)    | Binary classification | 1 if card has ≥ 1 pip of that color                        |
| Pip counts (× 6: W/U/B/R/G/C)    | Linear regression     | Pip count for that color (fractional for hybrid/phyrexian) |
| Mana value                       | Linear regression     | Sum of all pips including generic; X = 0                   |
| Mana produced (× 6: W/U/B/R/G/C) | Binary classification | 1 if card has a `{T}: add {color}` ability                 |

# Label Generation

All auxiliary labels are extracted from the card's `.txt` file at training time using the same parsers already 
present in `sealed.domain.mana_scorer`:

- **Is land**: type-line check (same logic as `EmbeddingAdapter.is_land()`)
- **Card color / pip counts**: `count_pips()` on the `mana cost:` line
- **Mana value**: `compute_mana_value()` on the `mana cost:` line
- **Mana produced**: `count_actual_sources()` scanning `activated[N]: {T}: add` lines

Non-land cards that have tap-for-mana abilities (Sol Ring, Orzhov Signet, Llanowar Elves) receive positive labels
for the colors they produce. A card with no mana ability gets 0 for all six mana-produced heads.

# Training

The combined loss is:

L_total = L_price + λ × Σ L_aux_i

Where:
- `L_price` is the existing price prediction loss (unchanged)
- Each classification head uses binary cross-entropy
- Each regression head uses MSE
- `λ` is a weighting hyperparameter controlling the strength of auxiliary supervision

The gradients from the auxiliary heads flow back through the pooling layer and the full transformer, updating 
all encoder weights to encode the supervised features into the pooled representation.

After training, the 20 auxiliary heads are discarded. The saved checkpoint contains only the encoder
(transformer + pooling), identical in format to the current `latest.pt`.

# Validation

After retraining, run feature 014's embedding validation against the new encoder:

```bash
  python -m sealed encode-cards --clean
  python -m sealed validate-embeddings --cards-path output/cardsfolder/
```

The retrained encoder must pass all 20 probes. This is the acceptance criterion — if any probe fails, the auxiliary
supervision was insufficient (likely a loss weighting issue) and training should be repeated with a higher λ.

Price prediction accuracy on the held-out validation set must not degrade by more than an acceptable tolerance 
relative to the feature 007 baseline. This guards against λ being set so high that the auxiliary losses dominate 
and the encoder forgets how to predict prices.
