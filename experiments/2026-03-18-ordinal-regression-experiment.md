# Ordinal Regression Experiment — March 2026

## Context

Starting point: commit `bd9a7b13` ("Make log_offset configurable and persist it in the model artifact").

At that point the transformer model used **Huber loss on log(price + offset)** — a single scalar output per card trained to predict shifted log-price. The model had a three-layer output head: `Linear(d_model → 64) → ReLU → Linear(64 → 1)`.

The validation results at that baseline (with ABU flag and no-mana-cost improvements applied) were:

```
median_abs_error_log: 0.864   top_20_overlap: 0.62

Bucket   med|log|   med_signed_log
<€2        0.816        +0.746
€2–10      1.172        -1.154
€10–50     1.952        -1.952
>€50       3.275        -3.275
```

The per-bucket breakdown revealed a systematic bias: the model overpredicts cheap cards and severely underpredicts expensive ones. A >€50 card was being predicted at ~4% of its actual value on average.

---

## Root cause analysis

~90% of cards cost less than €2 (5461/6057 in validation). The model's gradient is dominated by cheap cards. Any signal from the ~85 expensive training cards is overwhelmed.

---

## What we tried

### 1. Soft ordinal regression (replacing Huber loss)

**The hope:** Replace the scalar regression target with K threshold neurons, each predicting P(price > t_i). Soft labels computed via the normal CDF: `target_i = Φ((log(price) − log(t_i)) / σ)`. Thresholds at `0.5 × 3^n`: €0.5, €1.5, €4.5, €13.5, €40.5, €121.5, €364.5, €1093.5. σ=0.75.

The idea was that each threshold neuron would only receive meaningful gradient from cards near its price range (the gradient of BCE is naturally small when the target is near 0 or 1 and the model is calibrated). This would structurally reduce the gradient imbalance without manual weighting.

Price reconstruction uses probit-inverse: `log(price) = log(t_i) + σ × Φ⁻¹(p_i)` for each neuron, combined as a weighted average with weight `φ(z) = exp(−z²/2)`. This allows extrapolation above the last threshold (e.g. Black Lotus at p₇=0.998 reconstructs to ~€9500, not capped at the bucket center).

**What actually happened:** The model still regressed toward the cheap majority. Overall metrics got slightly worse (median_abs_error_log 0.864 → various worse values through iterations). The >€50 bucket did not improve significantly.

**Note:** There was a sign bug in an intermediate commit (proportional odds variant) that produced median_pct_error of 3329% — this was caught and fixed.

---

### 2. Per-bucket evaluation (kept)

Added `BucketMetrics` and a per-bucket table to evaluation output. This was the right diagnostic tool — without it, the severity of the bias for expensive cards was invisible. **This change is retained in the final codebase.**

---

### 3. Proportional odds model (architectural change)

**The hope:** Replace the K independent output neurons with a shared weight vector `w` and K monotonically increasing threshold biases. All thresholds share the same feature projection: `logit_i = w·h − α_i`. Since cheap card data is abundant, it trains the direction `w` in feature space. Expensive cards only need to calibrate K scalar offsets rather than learning independent weight vectors from ~85 examples.

Monotonicity enforced by parameterizing: `α_i = α_0 + cumsum(softplus(deltas))`.

**What actually happened:**

```
Proportional odds results:
median_abs_error_log: 1.022   top_20_overlap: 0.64

Bucket   med_signed_log (old → new)
<€2        +0.746 → +0.993   (WORSE)
€2–10      -1.154 → -1.028   (slightly better)
€10–50     -1.952 → -1.897   (slightly better)
>€50       -3.275 → -2.683   (better)
```

The shared weight vector forced a compromise: cheap cards got worse, expensive cards got slightly better. The "parallel slopes" assumption (same feature importance at all price levels) was too strong. The overall median_abs_error_log got worse despite the >€50 improvement.

---

### 4. Distance-weighted BCE loss

**The hope:** Weight each (card, threshold) loss element by how close the card's price is to that threshold: `w_i = floor + (1−floor) × exp(−z_i²/2)`. A €0.10 card contributes near-floor gradient to the €1093.5 threshold neuron, and full gradient to the €0.5 threshold. The floor (default 0.1) preserves some "output low for cheap cards" signal to high thresholds.

Tried with separate `loss_weight_sigma` (2.0) for gentler falloff. Also tried reverting to simple linear output head (no ReLU, no hidden layer).

**What actually happened:**

```
Distance-weighted loss (sigma=0.75, floor=0.1):
median_abs_error_log: 1.280   top_20_overlap: 0.63

Bucket   med_signed_log
<€2        +1.295   (much worse — 3.65× too high)
€2–10      -0.747   (better)
€10–50     -1.748   (slightly better)
>€50       -2.750   (similar)
```

The €2–10 bucket improved meaningfully. But cheap cards got much worse (+0.746 → +1.295 signed log error). With the floor at 0.1, upper threshold neurons were receiving too little "cheap=low" signal and drifting upward, inflating cheap card predictions.

Tried larger sigma (2.0) — same pattern, different magnitudes. No combination solved both problems simultaneously.

---

### 5. Price-proportional weighted sampling

**The hope:** Use PyTorch `WeightedRandomSampler` with `weight = price^alpha`. Each epoch, expensive cards are oversampled and cheap cards are undersampled. The model simply sees more expensive examples per epoch without any architectural changes.

Tried alpha=0.5 (sqrt weighting). A €100 card is drawn ~70× more often than a €0.02 card per epoch.

**What actually happened:**

```
alpha=0 (uniform):     median_abs_error_log: 0.959   top_20_overlap: 0.63
alpha=0.5 (sqrt):      median_abs_error_log: 1.214   top_20_overlap: 0.59
```

Worse on almost every metric. The top_20_overlap dropped from 0.63 to 0.59. The cheap bucket degraded significantly. Even the >€50 bucket got slightly worse despite seeing more expensive cards, possibly due to overfitting to the small set of expensive training examples.

---

## Summary

Every rebalancing intervention consistently produced the same tradeoff: slightly better for mid/expensive cards, noticeably worse for cheap cards, worse overall metrics.

The best overall model remained the simple architecture with uniform sampling and no special loss weighting.

**Final architecture (current codebase):**
- Transformer encoder → masked mean pooling → `Linear(d_model → 1)` → scalar shifted-log prediction
- Huber loss on `log(price + log_offset)`
- Uniform sampling
- ABU flag (filters non-constructed-legal prices)
- `mana cost: none` token for lands (distinguishes zero-cost from no-cost)
- Per-bucket evaluation output

**Best observed metrics (simple arch, uniform sampling):**

```
median_abs_error_log: 0.959   top_20_overlap: 0.63

Bucket   med|log|   med_signed_log
<€2        0.928        +0.881
€2–10      1.181        -1.169
€10–50     2.044        -2.044
>€50       2.308        -2.308
```

The >€50 improvement over baseline (−3.275 → −2.308) came from architectural iteration, not sampling.

---

## Why rebalancing didn't work

With ~85 expensive training cards vs ~22k cheap ones, there is simply not enough data for the model to learn what distinguishes a €100 card from a €10 card. Rebalancing changes *how often* the model sees expensive cards, but the model still cannot generalize reliably from 85 examples of a complex multi-factor phenomenon.

The most promising remaining direction is **domain-specific features** that are strong, explicit signals for expensive cards: reserve list membership, set code (Alpha/Beta/Unlimited/Limited), Power Nine membership. These are categorical facts that would require only a handful of examples to learn from once exposed explicitly as features.

---

## Pooling method experiments — March 2026

After rolling back to the simple scalar architecture, three pooling strategies were compared. All results use the same transformer encoder (d_model=128, 4 layers, 4 heads) and output head (`Linear → ReLU → Linear`).

**Important metric note:** `median_abs_error_log` uses `log(price + 2.0)`, which heavily compresses errors for cheap cards. A 3.3× prediction error on a €0.10 card appears as 0.10 in shifted log space. The `med_signed_log` column (raw `log(price)`, no offset, negative = underprediction) is the honest per-bucket measure used for comparison throughout.

### 1. Mean pooling

Sum non-padding token representations, divide by real token count.

```
MAE: €2.78   median_abs_error_log: 0.044   top_20_overlap: 0.62

Bucket   med_signed_log
<€2        +0.379
€2–10      -1.229
€10–50     -1.471
>€50       -3.018
```

Best for cheap cards. Worst for expensive cards.

---

### 2. Max pooling

Per dimension, take the maximum value across all non-padding positions (padding filled with −∞).

```
MAE: €2.35   median_abs_error_log: 0.128   top_20_overlap: 0.63

Bucket   med_signed_log
<€2        +1.196
€2–10      -1.158
€10–50     -1.817
>€50       -2.644
```

Better for expensive cards, significantly worse for cheap cards (+1.196 vs +0.379). The improved `median_abs_error_log` is a metric artifact from the log+offset compression, not a genuine overall improvement.

---

### 3. Concatenated max + mean pooling (current)

Both poolings computed independently, concatenated to a `2×d_model` vector fed into the output head (`Linear(2×d_model → 64) → ReLU → Linear(64 → 1)`).

```
MAE: €2.45   median_abs_error_log: 0.053   top_20_overlap: 0.62

Bucket   med_signed_log
<€2        +0.493
€2–10      -1.253
€10–50     -1.818
>€50       -2.621
```

The >€50 bucket is the best observed (−2.621), slightly beating max pooling alone (−2.644). Cheap card error (+0.493) is between mean (+0.379) and max (+1.196) — the head partially learned to temper the max pooling bias. This approach was kept.

**Attention pooling** (learned scalar scores per position, softmax weights) was also tried as a third concatenated pooling. It did not improve results over max + mean and was removed.

---

### 4. Dropout ablation

Tested `--dropout 0.05` and `--dropout 0.0` against the default `0.1`, all with concatenated max + mean pooling.

```
Bucket        dropout=0.1   dropout=0.05   dropout=0.0
<€2             +0.493        +0.687         +0.452
€2–10           -1.253        -1.310         -1.207
€10–50          -1.818        -1.988         -1.814
>€50            -2.621        -2.071         -2.607
```

`dropout=0.05` produced the best-ever >€50 result (−2.071) but at the cost of significantly worse cheap cards (+0.687). `dropout=0.0` is nearly identical to 0.1. The default 0.1 was retained as the best overall.

The suggestion that lower dropout helps small datasets is partially correct: it did help the expensive card signal, but made the cheap majority worse — the same tradeoff seen in every other rebalancing attempt.
