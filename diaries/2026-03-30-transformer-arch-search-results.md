# Sunday, 30 March 2026 — Transformer architecture search results

**TL;DR:** Ran a systematic grid search over transformer architectures and
confirmed 256/6 (d_model=256, n_layers=6) as the best practical model.
Tried LOWESS post-hoc calibration, watched it catastrophically over-correct,
and reverted it.

The day started with implementing the architecture search script. The plan was
already written: a 9-combination grid over d_model ∈ {64, 128, 256} and
n_layers ∈ {2, 4, 6}, each scored by a weighted sum of median log errors
across price buckets, with the €2–50 range weighted highest. I extended it to
accept `--d-models` and `--n-layers` as CLI arguments so I could push the
grid further without touching the script.

The first round of results (20 epochs, patience=5) had a clear pattern: width
dominated depth. All three 256-wide configs ranked above all 128-wide configs,
which ranked above all 64-wide configs. Within each width, more layers helped
— but only at 256. At 64, adding more layers made things worse, the classic
sign of a model too narrow to use its depth.

I then noticed that 256/6 didn't early-stop — it ran all 20 epochs with
best_epoch=19, meaning it was still improving at the cutoff. That made the
search's rankings unreliable: it was comparing converged models against one
that hadn't finished training yet.

Extending the search to larger models (512/6, 512/8) with 20 epochs confirmed
that 512/8 was clearly overfit — best_epoch=5, then immediate deterioration.
512/6 outscored 256/6 on the composite metric but a per-bucket look told a
different story: 512/6 dominated the >€50 and €10–50 buckets, while 256/6
nearly perfectly nailed €2–10 (signed log +0.040, essentially unbiased).
The two models had almost perfectly complementary strengths.

I then reran with 100 epochs and patience=20. The 20-epoch search had been
cutting everything short. At full convergence, 256/6 landed at MAE €1.07,
top_20 overlap 0.70, composite score 2.371, with best_epoch=47. The 8-layer
models failed again in both widths — definitively ruled out for ~6k cards.

Along the way I noticed that nearly every model reported exactly 100% for the
`median_percentage_error` column. The explanation: when the model
underestimates a cheap card, `exp(prediction) - log_offset` goes negative and
gets clamped to zero, and `|0 - actual| / actual` is mathematically exactly
100% regardless of how close or far the prediction was. The metric was telling
me "the model predicted zero" rather than measuring error. It became a real
number (57.5%) once the model trained long enough to stop clamping.

After settling on 256/6 as the default architecture and updating the CLI
defaults accordingly, I tried adding LOWESS post-hoc calibration. The idea was
to fit a smooth correction curve over the (predicted_price, residual)
scatterplot from the validation set, then apply `np.interp` at inference.
Running it revealed a signed log bias pattern: cheap cards overpredicted
(+0.795), the €0.50–2 range underpredicted (-0.667), and the expensive buckets
roughly balanced.

The problem was that 77% of validation cards are cheap. When LOWESS tries to
estimate the correction at high predicted prices, it reaches back into the
cheap-card cluster to fill its neighbourhood window and inherits their
downward-correction bias. The results were catastrophic: MAE jumped from €1.07
to €2.21, and the €2–10 bucket went from signed +0.040 (essentially perfect)
to -1.124 (predicting cards at one-third of actual price). I reverted
everything.

The day ended with a design discussion about revisiting ordinal regression —
specifically, whether giving each regression head its own separate ReLU (rather
than sharing one, which was the previous attempt's flaw) would allow the heads
to specialise. That's where the session ended, with a plan drafted but not yet
implemented.
