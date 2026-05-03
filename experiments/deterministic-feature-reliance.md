# Deterministic-Feature Reliance Diagnostic

## Background

The sealed scorer's per-card embedding is a 544-dim concatenation:

- **Indices 0–511 (512 dims)** — `cat([max_pool, mean_pool])` of the
  price-predictor transformer's token outputs over the card text. Trained
  end-to-end during the price-predictor pretrain (and during Phase B
  fine-tuning when enabled).
- **Indices 512–543 (32 dims)** — hand-extracted deterministic features
  parsed by `parse_deterministic_features` from the converted card text:
  `is_land`, per-color pip counts (W/U/B/R/G/C), generic mana, X count,
  total mana value, color flags (WUBRG + colorless), mana production
  (WUBRG + C + count), power, toughness, starting loyalty, plus 6
  zero-padding slots. **These features are not learnable.**

The full layout is documented in `specs/sealed-deck-picker.md` § "Deterministic
Feature Encoding (indices 512–543)".

The concern: the scorer may be using the 32 deterministic features almost
exclusively, with the 512 transformer dims contributing little. This was
not the original hypothesis — it surfaced from win-rate diagnostics
showing the scorer has learned macro deck shape but not per-card quality.

## How the win-rate analysis led here

A new family of `analyze_winrates.py` tables (added 2026-05-02) sliced the
gen-2 corpus several ways:

1. **By deck color count** — confirmed monotonic decline from 2-color
   (63.6% Overall) to 5-color (28.9% Overall). gen2a built 4-5 colors
   44% of the time despite the 35pp gap.
2. **By color presence** — modest spread (3.5pp). gen1's most-built
   color (G) was its second-best winning color; gen2a's color order
   roughly matched its win-rate order — partial calibration.
3. **By creature count** — clear ≥16-creatures sweet spot (54%); decks
   with ≤13 creatures lose at 36%.
4. **By avg nonland MV** — winning band 3.4-4.0 (53-54%); gen2a deliberately
   played MV 3.19, *below* the empirical optimum.

Each of those tables is consistent with the scorer reasoning about deck
shape — color count, color identity, creature count, mana curve are all
direct aggregations over the 32 deterministic features (sum of color
flags, mean of `mana_value`, count of `is_land == 0`, etc.). The Set
Transformer can compute every one of them without ever reading the 512
transformer dims.

The smoking-gun observation came from **within-bucket comparisons**:

| Comparison                              | forge-best | gen2a    | Gap   |
|-----------------------------------------|------------|----------|-------|
| 2-color decks                           | 64.1%      | 56.0%    | 8pp   |
| MV 3.1-3.4 decks                        | 69.7%      | 52.6%    | 17pp  |
| MV 2.8-3.1 decks                        | 59.5%      | 51.8%    | 8pp   |

When you condition on deck shape matching, forge-best still wins by
8-17pp. Deck shape is what the deterministic features can express;
card quality is what the transformer dims would have to express.
Forge-best — a hand-coded heuristic that does know which 2-mana red
instant is Lightning Bolt vs Shock — wins those within-bucket
comparisons. The scorer does not.

The hypothesis falls out of that gap: if the scorer were leveraging the
512 transformer dims to read card abilities, it should be closing some
of the within-bucket distance. The fact that it isn't, combined with
the architecture exposing the 32 hand-features as a parallel input, is
consistent with the scorer leaning almost entirely on those.

## What this hypothesis would explain

- **gen2a's macro-feature coherence.** Color count, mana value, creature
  count, type balance — every working stat is computable from the 32
  features alone via Set Transformer aggregation.
- **Within-bucket card-quality blindness.** Distinguishing Lightning Bolt
  from Shock requires reading the oracle text. That information is in
  the 512 transformer dims and apparently not in use.
- **Phase B's null result** (`gen2-unfrozen-embeddings.md`). Phase B
  fine-tuning updates only the 512 transformer dims. If the scorer
  barely reads them, gradient updates to those dims don't change the
  deck score and don't move val_acc.
- **gen2a ≈ gen2b1 ≈ gen2ba within-bucket.** All three families share
  the same 32 deterministic features (these are not trained). If those
  carry the bulk of the signal, the three variants should land in the
  same band — which the win-rate tables show they do.
- **Why the scorer over-extends on color count.** It can read color
  count just fine but apparently doesn't price the cost of stretched
  manabases against per-card quality, because pricing per-card quality
  is what it can't do.

## Diagnostics planned

Three tests, cheapest first.

### 1. Zero-ablation of transformer dims (two-stage)

Two ablation runs answer different questions about the transformer dims.
The cheaper one runs first and may obviate the second.

**1a. Inference-only zero-ablation.** Mask indices 0-511 of every
per-card embedding to zero at the scorer's input (in `match_data_loader`
when loading `.npz` files, so eval and any later retrain go through the
same code path). Reload an unchanged gen2a Phase A checkpoint and run
`evaluate-scorer` / play ~100 matches against forge-best. Compare win
rate to baseline.

- Win rate unchanged → current model already ignores those dims.
  Hypothesis confirmed without retraining; stop here.
- Win rate drops → current model uses those dims. Run 1b.

Mirror-image ablation in the same run (zero indices 512-543 instead of
0-511) tells us how much the deterministic features contribute to the
current model.

Cost: ~30-line mask in the data loader + one `evaluate-scorer` run.
~1 hour.

**1b. Retrained zero-ablation** (only if 1a shows the current model
relies on transformer dims). Train a fresh Phase A scorer on
`match-outcomes-all.txt` with the same 512-dim mask in place from the
first epoch onward. The architecture is unchanged; the model just never
sees nonzero values in those dims. Compare peak val_acc and forge-best
win rate to the gen2a baseline.

- Retrained model matches baseline → transformer dims are *redundant*
  with the deterministic features. The current model leans on them but
  a fresh model substitutes deterministic-feature signal and recovers.
  Same actionable conclusion as the 1a-confirmed case: in their current
  form the embeddings don't pay rent.
- Retrained model drops materially → transformer dims contribute unique
  signal that deterministic features can't replicate. Hypothesis
  falsified. The within-bucket gap to forge-best is then explained by
  something other than scorer feature reliance — most likely the Bo7
  label-noise floor binding before the scorer can extract per-card
  quality cleanly.

The retrained variant directly tests "even if the embeddings are used,
are they useful in their current form?" — which the inference-only
ablation can't answer because the current model might be using them as
a crutch that a fresh model could replace.

A side benefit: 1b's val_acc is the upper bound an "all-deterministic-
features" scorer can reach on this corpus, useful regardless of which
way the hypothesis lands.

Cost: one full Phase A run on `match-outcomes-all.txt` (~25s/epoch,
~30-50 epochs to converge with patience). A few hours of GPU time.

### 2. Within-deck transformer-dim permutation

For each deck, randomly permute indices 0-511 across the 23 cards (card
A keeps its own 32 deterministic features but gets card B's transformer
dims). Re-score the permuted deck with the unmodified scorer. Compare
to original deck score.

- If scores are essentially unchanged: the model is invariant to *which*
  card has *which* transformer features — it only reads the aggregate
  / set-level signal in those dims, or ignores them altogether.
- If scores change a lot: per-card transformer features matter; the
  scorer is associating specific transformer patterns with specific
  cards.

Cost: ~50-line script. ~1-2 hours.

### 3. Block-wise gradient norms during training

In `train_scorer`, log `||∂L/∂x_text||` and `||∂L/∂x_det||` separately
per training step (where `x_text` is dims 0-511 and `x_det` is dims
512-543 of the per-card embedding input).

- If the deterministic-block gradient norm is 5-10× the transformer-block
  norm: the loss is mostly explained by the deterministic features and
  the scorer is barely tuning around the rest.
- Roughly equal norms: both blocks contribute, but doesn't rule out the
  hypothesis on its own (gradients flow even into ignored features
  through residual paths).

Cost: ~10 lines of training-time instrumentation. Run during a normal
Phase A training pass.

## Actionable conclusions, conditional on results

If the diagnostics confirm the hypothesis, the next steps fork:

- **Lean into hand-engineering.** If the scorer only uses hand-features,
  adding more carefully-extracted features (removal-spell flag, evasion
  keyword count, ETB-trigger count, mana-cost-tier-relative-to-stats
  ratio, etc.) buys direct capacity. The encoder can stay frozen —
  Phase B can't compete with hand-engineering on a noisy-label budget.
- **Force the model off hand-features.** Train without the 32-dim
  deterministic block and let the transformer dims be the sole source.
  Slower, data-hungry, but the resulting embeddings would actually
  encode card quality. High risk if labels remain noisy at the Bo7
  floor.
- **Multi-task encoder pretraining.** If the bottleneck is that
  match-outcome gradients can't teach card-quality through a noisy
  512-dim path, pre-train the encoder on dense per-card auxiliary
  signals (predict types, keywords, ability count, oracle-text
  classification, etc.) before scorer training. Already on
  `future-experiments.md` under "Pre-train the encoder on a closer
  auxiliary task".

If the diagnostics rule out the hypothesis (transformer dims are doing
substantial work), then the within-bucket gap to forge-best is harder
to explain via architecture and points more strongly at the label-noise
floor — back to the data-side levers in
`gen2-unfrozen-embeddings.md`'s "Decisions taken" § 3.

## Status

Hypothesis surfaced from gen-2 win-rate analysis. No diagnostic run yet.
Test (1) is the immediate next step.
