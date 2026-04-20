# Saturday 21 March 2026 — Architecture search and calibration experiments

**TL;DR:** Ran a systematic grid search over transformer hyperparameters,
found d_model=256 / n_layers=6 to be the best practical architecture, then
explored and ultimately abandoned post-hoc LOWESS calibration.

The day started by implementing and running an architecture search script
covering nine combinations of d_model ∈ {64, 128, 256} × n_layers ∈ {2, 4,
6}, with n_heads fixed at 4 and ff_dim = 4 × d_model. The results were
striking: d_model turned out to be the dominant factor — all three 256-wide
models beat all three 128-wide models, which in turn beat all three 64-wide
ones. Within the 64-dim family, adding more layers actually made things worse,
which Claude explained as a too-narrow model unable to make use of the extra
depth. The 256/6 configuration came out on top with a composite score of 2.62
and MAE €1.04.

That first search only ran for 20 epochs, and 256/6 had reached best_epoch=19
without early stopping, which meant it hadn't converged. So I extended the
script to accept --epochs and --patience as CLI arguments, re-ran with 100
epochs and patience=20, and confirmed the gains were real: best_epoch moved to
47, val_loss dropped from 0.088 to 0.081, and top_20_overlap improved from
0.64 to 0.71. The median_percentage_error metric also became meaningful for the
first time — it had been stuck at exactly 100% for all models in the 20-epoch
runs because predictions for cheap cards were getting clamped to zero, which
makes the formula a mathematical identity rather than a measurement. With
longer training the model stopped doing that.

I then ran a second search comparing 256/6, 256/8, 512/6, and 512/8 at 100
epochs. 512/8 failed badly — best_epoch at 5, which meant it peaked almost
immediately and then deteriorated while train_loss kept falling, a clear
overfitting signature. 8 layers is definitively too deep for roughly 6k cards
regardless of width. 512/6 edged out 256/6 on composite score (2.309 vs
2.371), but the two models turned out to have almost perfectly complementary
strengths: 256/6 nailed €2–10 with a near-zero signed log error (+0.04),
while 512/6 dominated €10–50 and above. Since 512/6 is 2.5× slower to train
and its score advantage is within the run-to-run noise we could directly
observe (256/6 scored 2.623 in one run and 2.789 in another on identical data),
I settled on 256/6 as the practical default and locked in those hyperparameters.

The per-bucket breakdown also revealed a consistent calibration bias: the model
overshoots cheap cards and undershoots the €0.50–2 range. We explored whether
LOWESS smoothing could correct this post-hoc. The idea was to fit a smooth
curve through the (predicted_price, residual) pairs on the validation set and
apply it at inference. I implemented it, baked the curve into the .pt file
alongside the weights, and ran a full retrain. The results were catastrophic:
MAE jumped from €1.07 to €2.21, and the €2–10 bucket median signed log went
from +0.04 to -1.124. The cause was clear in retrospect: with 77% of
validation cards being cheap, the LOWESS window at expensive predicted values
was forced to include cheap-card neighbours with negative residuals,
contaminating the correction in exactly the wrong direction. LOWESS needs
roughly uniform data density to work; this dataset is anything but. Everything
was reverted.

The day ended with a discussion about Mixture of Experts output heads as an
alternative route to improving accuracy on expensive cards — specifically a
variant with one separate ReLU per expert head rather than a shared one, which
was the failure mode of an earlier ordinal regression experiment. The planning
for that carried into the next session.
