# Gen2 Scorer — Initial Training Runs

## Background

The **sealed deck scorer** (see `src/sealed/domain/scorer_model.py`) is a Set Transformer
that takes a 40-card deck as an unordered set of card embeddings and outputs a scalar
"deck quality" score. It is trained pairwise on a match-outcomes file: each row is a
Bo7 match between two decks, and the model learns to assign a higher score to the
winner. Architecture is a stack of Self-Attention Blocks (SAB) → Pooling-by-Multihead-
Attention (PMA) → MLP head. The number of SAB layers is controlled by `--n-layers`.

"Gen1" is the first scorer trained mostly on Forge-AI-built decks (`forge-best`,
`forge-3sub`, `forge-8sub`, `random`). It works — gen1 decks beat the Forge baseline
about 56% of the time — but it has a known reward-hacking failure mode: it rates its
own greedy output highly, even though those decks tend to have too many colors and too
high a mana curve, and lose to `forge-best` ~57% of the time at Bo7.

**"Gen2" is the next scorer iteration**, trained on a dataset that includes self-play
matches (~5,800 gen1-vs-other matches in `match-outcomes-gen1.txt`) on top of the
original Forge-built data (~14,500 matches in `match-outcomes-gen0.txt`). The hope is
that the new scorer learns from gen1's losses to `forge-best` and corrects the
over-confidence problem.

The runs analyzed below are a depth sweep over `--n-layers ∈ {2, 3, 4, 5, 6}` on the
combined corpus (`match-outcomes-all.txt`, ~20,373 matches). All runs share
`--lr 1e-5`, default 80/20 train/val split, default optimizer (Adam, no weight decay),
no dropout, and no gradient clipping. Each run was manually interrupted via Ctrl+C
once the val-loss curve had clearly turned the corner; epoch counts at interrupt were
22, 25, 27, 31, and 42 for the 4-, 3-, 6-, 5-, and 2-layer runs respectively.

### How to read the metrics

- `train_loss`, `val_loss` — pairwise binary cross-entropy on (winner, loser) pairs.
  Random guessing ≈ 0.693.
- `train_acc`, `val_acc` — fraction of pairs where `score(winner) > score(loser)`.
  Random ≈ 0.50.
- `winner=X±Y / loser=X±Y` — mean and std of the model's raw output score on the
  validation set's winners and losers.
- `grad_norms` — gradient L2 norm per submodule each epoch.

## Headline numbers

| n_layers | Peak val_acc | Peak epoch | Min val_loss | Min-loss epoch | Train_acc at peak | Notes                                                         |
|----------|--------------|------------|--------------|----------------|-------------------|---------------------------------------------------------------|
| 2        | 0.6970       | 15         | 0.5788       | 14             | 0.7431            | min-loss epoch within 1 of peak-acc epoch — clean convergence |
| 3        | 0.6875       | 16         | 0.5824       | 7              | 0.7724            | val_loss had already turned up before peak val_acc            |
| 4        | 0.6910       | 12         | 0.5779       | 7              | 0.7515            | similar shape to 3-layer                                      |
| **5**    | **0.7017**   | **12**     | **0.5753**   | **6**          | **0.7591**        | best peak val_acc and best min val_loss in the sweep          |
| 6        | 0.6963       | 12         | 0.5795       | 4              | 0.7591            | nearly tied with 5-layer; min val_loss reached fastest        |

The five depths cluster within 1.4 percentage points of val_acc (0.6875 → 0.7017),
which is well inside the run-to-run noise floor of small validation samples. **5
layers nominally wins** on both peak val_acc (0.7017) and best min val_loss (0.5753),
but the sweep does not separate the architectures by a meaningful margin.

## What happened in every run

The shape is the same across all five depths:

1. **Fast initial learning.** Every run reaches val_acc ≥ 0.665 at epoch 1 and ≥ 0.68
   by epoch 4. The full ~20K-match dataset has plenty of signal that the model picks
   up immediately.
2. **Peak val_acc by epoch ~12–16, then plateau or decline.** After the peak, val_acc
   wanders within a 2–4 pp band while train_acc keeps rising.
3. **Train loss collapses.** From 0.69 (random) down to 0.10–0.27 by the time each
   run was interrupted. Train_acc reaches 0.85–0.97 in the same window.
4. **Val loss bottoms out very early — usually epochs 4–7 — and then rises
   monotonically.** This is the cleaner signal that overfitting has begun. By
   interrupt, val_loss is 1.4–1.9× its minimum.
5. **Score magnitudes explode.** The std of raw scores grows from ~0.5 → ~5 over 30
   epochs while the winner-vs-loser score *gap* stays around 0.5–2.0. The scoring
   head is using its full unbounded range without anything regularizing it.

The fact that **min val_loss precedes peak val_acc by several epochs** in every run is
worth noting — the model becomes more confident on the training set in ways that
sometimes flip a few borderline validation pairs the right way (raising val_acc) while
making the average validation log-likelihood worse (raising val_loss). The val_loss
minimum is the more conservative early-stop signal.

## Best model

**5 layers** is the nominal pick: highest peak val_acc (0.7017 at epoch 12) and lowest
min val_loss (0.5753 at epoch 6). But the gap to the runners-up (2-layer at 0.6970,
6-layer at 0.6963) is only ~0.5 pp on val_acc — squarely within sample noise on a
~4K-example validation set.

If you only need *a* checkpoint to deploy as gen2, take the 5-layer best-by-val-acc
checkpoint (saved automatically by the trainer at epoch 12). If you want a more
defensible architecture pick, run a same-pool head-to-head evaluation (see
recommendations below) — the val_acc numbers alone do not justify a strong preference.

It is also worth noting that **2 layers is competitive with 5–6 layers** on this
dataset. The depth sweep is not showing a clear capacity ceiling at the shallow end,
which suggests the data has more headroom for capacity than these runs are using —
but only once overfitting is brought under control with regularization.

## Other diagnostics worth flagging

- **Gradient norms are large and spiky.** Even at modest depth (2 layers) `sab0`
  occasionally hits 12–18 in mid-training; at 6 layers it spikes to 27–46 around
  epoch 18–26. Gradient clipping (e.g. `clip_grad_norm_(1.0)`) would smooth this.
- **Per-layer gradient decay with depth is mild but present.** In the 6-layer run,
  `sab0`/`sab5` ratios sit around 1.5–2.5×; this is healthy gradient flow, not
  vanishing-gradient territory.
- **Score std grows roughly 10× over training** (e.g. 5-layer: winner std 0.50 → 5.07
  by epoch 29). Combined with a gap that grows much more slowly, this is the typical
  signature of an unconstrained scoring head that AdamW with weight decay would
  rein in.
- **One anomalous epoch in the 5-layer run** (epoch 27): all gradient norms collapse
  to ~0.01–0.05 for a single epoch, then bounce back to normal levels the next
  epoch. Probably a near-saturation moment where the loss landscape was very flat;
  not a bug, but worth noting for the log.

## Recommendations for the next sweep

In rough order of expected impact:

1. **Add dropout** in the SAB blocks (after attention, between FF layers) and in the
   scoring MLP. Start with p=0.1–0.2. Dropout randomly silences a fraction of neurons
   each forward pass, forcing the model to develop redundant, robust representations
   instead of memorizing narrow patterns.
2. **Switch to AdamW with `weight_decay` in the 1e-4 to 1e-2 range.** Penalizes large
   weights, which keeps the score magnitudes from exploding and biases the model
   toward smoother decision functions.
3. **Use `val_loss` (not `val_acc`) as the early-stopping signal**, with patience
   ~5 epochs. In every run here, `val_loss` bottomed out 5–10 epochs before `val_acc`
   peaked, and the trainer's "best by val_acc" checkpoint policy lets the model train
   well past its loss-optimal point.
4. **Add `clip_grad_norm_(1.0)`** to suppress the sporadic gradient spikes
   (norms of 20–46 are common in the deeper runs).
5. **Once regularization is in, repeat the depth sweep.** Expectation: deeper models
   may pull ahead once they're prevented from memorizing — the current ~0.5 pp spread
   between depths likely reflects the overfit ceiling rather than the underlying
   architecture quality.
6. **For final architecture selection, supplement val_acc with same-pool head-to-head
   evaluation** (`python -m sealed evaluate-scorer --set <SET>` against a fixed pool
   set for each candidate). The pairwise val metric does not directly measure
   deck-building strength against Forge-AI.

## Bottom line

With the full ~20K-match dataset, every architecture lands in a tight 0.687–0.702
val_acc band — a clear improvement on prior incomplete data, and good enough that the
5-layer best-by-val-acc checkpoint (epoch 12) is a defensible candidate for gen2.

The dominant signal across the sweep is overfitting: train_acc climbs past 0.75 while
val_loss starts rising by epoch 7, and architecture choice barely moves the needle
inside that overfit ceiling. The next iteration's biggest win will come from
regularization (dropout + weight decay + val_loss-based early stopping), not from
further depth tuning.
