# Gen2 Scorer — Phase B (Unfrozen Embeddings)

## Background

This is the follow-up to `gen2-initial-training.md`, which left the val_acc
ceiling at ~0.70 across four orthogonal interventions (depth sweep, dropout
sweep, multi-view pooling, hand-computed deck stats). That document concluded
the ceiling is largely data-limited (Bo7 noise floor ~0.72–0.78) but flagged
**unfreezing the card embeddings** as the next plausible model-side lever:
the price-predictor encoder was trained on "what makes this card cost €X",
only loosely correlated with "what makes this card good in a sealed deck",
so fine-tuning it toward deckbuilding-relevant features could plausibly
close the model-imperfection gap before the oracle ceiling binds.

Spec `specs/encoder-fine-tuning.md` (and the implementation in
`specs/015-encoder-fine-tuning/`) defines this as **Phase B**: a non-zero
`--embedding-lr` puts the encoder in the training graph alongside the
scorer, jointly trained on match outcomes. Phase A keeps the encoder
frozen (the existing `.npz` cache feeds the scorer); Phase B unfreezes it.

Starting point for the runs below: the gen2 best Phase A checkpoint —
6 SAB layers, dropout 0.2, AdamW, `--lr 1e-5`, trained on
`match-outcomes-all.txt` (~27K matches as of run time), peak val_acc
~0.7002. The encoder is the price-predictor's `latest.pt`
(d_model=256, max_seq_len=256, 6 transformer layers).

### How to read the metrics

Same as `gen2-initial-training.md`, plus:

- `embedding_drift` — mean L2 distance between the current encoder's text
  vectors on a fixed reference batch (the unique cards in step 0's first
  Phase B batch) and their step-0 values. Captured and recomputed in
  `encoder.eval()` mode so dropout doesn't muddle the signal.
- `grad_norms: <name>=mean(...)/max(...)` — per-parameter-group L2 norm
  pre-clip, aggregated across the epoch's batches. Mean is the typical
  step's gradient magnitude; max tells you whether the clip threshold
  (`--max-grad-norm`, default 100) is binding on any batch.

## What was tried

Three Phase B variants on the same `match-outcomes-all.txt` corpus:

| Run | Phase A start              | `--embedding-lr` | `--patience` | Hypothesis tested                                          |
|-----|----------------------------|------------------|--------------|------------------------------------------------------------|
| 1   | converged                  | 1e-7             | 10           | Standard Phase B, baseline expectation that it improves    |
| 2   | 1-epoch warm start         | 1e-7             | 20           | Converged Phase A's overfit poisons Phase B's encoder      |
| 3   | 1-epoch warm start         | 1e-6             | 20           | Encoder LR too low to shed price-task features in time     |

## Headline numbers

| Run                    | Date       | Peak val_acc | Peak epoch | Final drift | Stop reason          |
|------------------------|------------|--------------|------------|-------------|----------------------|
| 1 (converged + 1e-7)   | 2026-04-29 | 0.6976       | 3          | 0.18        | manual interrupt (5) |
| 2 (1-epoch + 1e-7)     | 2026-04-29 | 0.6956       | 13         | 1.05        | patience (33 epochs) |
| 3 (1-epoch + 1e-6)     | 2026-04-30 | 0.6860+      | 12+        | 3.38+       | manual interrupt     |

For comparison, the Phase A baseline that fed runs 1–3 had peak val_acc
~0.7002. **No Phase B variant improved on Phase A.** Every variant peaked
below the baseline.

## Run 1 — standard Phase B (2026-04-29)

**What we expected.** Encoder fine-tuning at the spec's recommended
`--embedding-lr 1e-7` would shift the encoder slightly toward
deck-quality-relevant features over a handful of epochs, lifting val_acc
above the ~0.7002 Phase A baseline by a few pp.

**What happened (5 epochs before manual stop).** Peak val_acc 0.6976 at
epoch 3 — *below* the Phase A starting point. Drift grew smoothly from
0.037 → 0.177, well under the spec's 1.0 alarm threshold; clipping was
not binding (max scorer norm 10–15, max encoder norm 5–6, both well
under 100). Train_acc rose 0.753 → 0.773 over 5 epochs; val_acc was
flat. The train-val accuracy gap widened from 5.9pp to 8.0pp — clean
overfitting.

**Conclusion.** At default Phase B settings, the model fits the
training set faster than before but val_acc doesn't follow. The
hypothesis "Phase B improves on Phase A by default" loses.

## Run 2 — warm-start hypothesis (2026-04-29 → 2026-04-30)

**What we expected.** A converged Phase A scorer has memorized
training-specific patterns, and unfreezing the encoder gives those
memorized patterns a backchannel into the encoder via gradient flow —
the scorer's overfit could push the encoder toward "produce features
that match what I already memorized" rather than "produce features that
generalize." Starting Phase B from a *less* overfit Phase A (just
1 epoch — scorer past random init but not yet memorizing) was
hypothesized to free the encoder to learn task-relevant features
without that backchannel pull.

**What happened.** Peak val_acc 0.6956 at epoch 13, declining to 0.6789
by epoch 33 when patience fired. The trajectory is informative:

- Started at val_acc 0.6723 (epoch 1) — well below the converged Phase A
  baseline (0.7002), as expected from a 1-epoch starting point.
- Climbed monotonically over 13 epochs to peak at 0.6956 — Phase B did
  real work, encoder genuinely contributed, but only got *near* the
  baseline that converged Phase A reaches in 1 epoch.
- After epoch 13, classic overfitting: train_loss 0.61 → 0.30 (cut in
  half), val_loss 0.61 → 0.80 (31% up), train-val accuracy gap 0pp →
  18pp.
- Drift hit 1.0 at epoch 31 — exactly the spec's "lower
  `--embedding-lr` and restart" alarm threshold. val_acc dropped from
  0.6804 (epoch 30) to 0.6612 (epoch 31), the worst single-epoch decline
  in the run. Catastrophic-forgetting territory, exactly as spec § 7
  predicted.

**Conclusion.** Warm-start hypothesis loses. Different starting point,
same val_acc ceiling. Run 2 ended ~0.5pp below the converged-baseline
peak; Phase B couldn't bridge that gap even with 33 epochs of
fine-tuning.

## Run 3 — encoder-LR hypothesis (2026-04-30, manually stopped at 12 epochs)

**What we expected.** If the encoder is too constrained by its
price-task pre-training to find deck-relevant features at
`--embedding-lr 1e-7`, a 10× higher LR (1e-6) should let it shed price
features faster and reach a higher val_acc. Drift at 1e-7 grew at
~0.035/epoch with substantial headroom under the 1.0 alarm; 1e-6 should
be aggressive but not catastrophic in the first ~3 epochs.

**What happened (first 12 epochs).** The encoder is moving very
aggressively — drift hit 0.91 at epoch 1, crossed the 1.0 alarm at
epoch 2, reached 3.38 by epoch 12. That's 7× the drift run 2 had at
epoch 12. Despite this, val_acc tracks ~0.005 *below* run 2 at every
checkpoint:

| Epoch | Run 2 (1e-7) val_acc | Run 3 (1e-6) val_acc | Δ      |
|-------|----------------------|----------------------|--------|
| 1     | 0.6723               | 0.6664               | −0.006 |
| 5     | 0.6756               | 0.6728               | −0.003 |
| 9     | 0.6856               | 0.6815               | −0.004 |
| 12    | 0.6882               | 0.6860               | −0.002 |

**10× more aggressive encoder fine-tuning produced no improvement —
slightly worse, in fact.** If price-task contamination were the binding
bottleneck, letting the encoder shed those features 7× faster should
have helped. It didn't.

The run was killed at epoch 12 once this signal was clear; predicted
trajectory is the same overfit collapse run 2 saw, just sooner because
the encoder is moving 10× faster in both useful and useless directions.

## What three Phase B runs from different starts have in common

| Run                              | Final state           | Peak val_acc | Final drift |
|----------------------------------|-----------------------|--------------|-------------|
| 1 (converged + 1e-7)             | 5 epochs, manual stop | 0.6976       | 0.18        |
| 2 (1-epoch + 1e-7)               | 33 epochs, patience   | 0.6956       | 1.05        |
| 3 (1-epoch + 1e-6)               | 12 epochs, manual     | 0.6860       | 3.38        |

Three runs, three different encoder behaviors (barely moved; moderate;
heavily moved), two different scorer starting points — same val_acc
ceiling around 0.70. **The ceiling holds across encoder fine-tuning
intensity in either direction.** This is the strongest evidence yet that
Phase B can't help on this corpus.

## Why this connects to the gen2-initial-training conclusions

`gen2-initial-training.md` ended with four orthogonal interventions all
hitting the ~0.70 ceiling and concluded the binding constraint is the Bo7
label-noise floor (oracle ceiling estimated at 0.72–0.78 from the math in
that document). Phase B was flagged as the one remaining model-side lever
that wasn't covered by the four — depth, dropout, multi-pool, and
deck-stats all sit downstream of the per-card features; Phase B is the
only intervention that changes the per-card features themselves.

The Phase B runs above add a **fifth orthogonal intervention** to the
list, and it lands the same way:

| Intervention                           | Outcome                |
|----------------------------------------|------------------------|
| Architecture depth (2–6 SAB layers)    | All converge at ~0.70  |
| Dropout (0.0, 0.1, 0.2, 0.3, 0.4)      | All converge at ~0.70  |
| Multi-view pooling (PMA + max + mean)  | Same as PMA alone      |
| Hand-computed deck stats (×1 to ×200)  | Same as no deck stats  |
| **Phase B encoder fine-tuning**        | **Same as Phase A**    |

`gen2-initial-training.md` argued the four-intervention pattern was
already strong evidence the ceiling is data-limited. Phase B was the
intervention with the strongest theoretical reason to break that pattern
(it changes per-card features, not just aggregation), and it didn't. The
ceiling is now extremely well-evidenced as the Bo7 oracle ceiling, not a
property of any model component.

## Diagnostics worth recording

- **Drift threshold of 1.0 is calibrated correctly.** Run 2 hit 1.0 at
  epoch 31 and immediately produced the worst single-epoch val_acc
  decline (−1.9pp) in the run. The spec's "lower the LR if drift > 1.0
  in first 3 epochs" rule triggered correctly when run 3 pushed the LR
  10×.
- **`--max-grad-norm 100` is the right default.** Across all Phase B
  runs, max scorer norms peaked around 23 (mid-overfit) and max encoder
  norms around 12. Clipping at 100 acts as the NaN-spike guard it's
  meant to be without ever throttling the configured LR. Lower values
  (e.g. 1.0) silently scale the effective LR down, since pre-clip norms
  are routinely 5–20.
- **Phase B at default LR is much slower than Phase A.** ~10 minutes per
  epoch vs ~25 seconds for Phase A. The ~1.5× compute penalty from
  gradient checkpointing the encoder forward, plus the encoder forward
  itself being the dominant cost. Plan accordingly when scheduling
  experiments.

## Decisions taken

- **Phase B is shipped but does not move val_acc on this corpus.** All
  the implementation work in spec 015 stays — chunked encoder forward,
  per-group clipping, drift metric, distinct Phase A/B checkpoint
  filenames, etc. — because a future run with cleaner labels or a
  different encoder might benefit. For the current
  `match-outcomes-all.txt` the deployed checkpoint stays Phase A.
- **Stopped pursuing Phase B variants on this data.** Three runs across
  two starting points and two encoder LRs all converge to the same
  val_acc ceiling. Further variants (shallower scorer, even more
  aggressive LR, etc.) would be testing the same dead hypothesis.
- **The next levers to pull are data-side, not model-side.** This
  matches `gen2-initial-training.md`'s recommendation #4 ("noise
  reduction at the data side"). Concrete options, in rough order of
  upside-per-cost:
  1. **More matches per pool combination.** Replace one Bo7 label per
     (deck_A, deck_B) pair with `N` Bo7s and a majority-vote label.
     Approximately halves label noise at 3× match-generation cost.
  2. **Longer matches** (`--best-of 11` or higher). Per-match noise
     drops as `1/√N`. Same regime, lower marginal cost than running
     multiple matches.
  3. **Sharper deck-pair selection.** If too many matched decks are
     within a few percent of each other in true win-rate, the labels
     carry no signal regardless of how many games per match. A
     deckbuilder that produces more clearly differentiated decks during
     match generation would give cleaner training pairs.
- **Architectural complexity to revisit only after labels improve.** If
  the noise floor moves up (say from ~0.78 → ~0.85 with cleaner labels),
  the model-imperfection bucket above 0.70 becomes the binding
  constraint again, and architectural levers (depth, pooling,
  Phase B at higher LR) become worth revisiting. Until then they all
  hit the same ceiling.

## Bottom line

Phase B works mechanically — the encoder moves, the scorer trains, the
drift metric tracks meaningful weight movement, the saved Phase B
checkpoint can be fed back through `encode-cards` to refresh the `.npz`
cache. None of that translated into higher val_acc on
`match-outcomes-all.txt`. Three Phase B runs spanning a 7× range of
encoder movement (drift 0.18 → 1.05 → 3.38) and two scorer starting
points (converged Phase A and 1-epoch Phase A) all land in a tight band
around 0.69–0.70 val_acc — the same ceiling four other interventions hit
in `gen2-initial-training.md`.

The combined evidence across eight interventions (five from the prior
document, three Phase B variants here) makes a strong case that the
ceiling is the Bo7 label-noise floor, not anything model-side. The next
iteration's wins will come from producing cleaner labels, not from
further model work.
