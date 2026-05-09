# April 30, 2026 — Phase B diagnosis and SA tuning

**TL;DR:** Phase B encoder fine-tuning ran its first full training loop and
the results looked bad — val accuracy peaked at epoch 1 and never recovered.
Claude identified three confounders that made the verdict untrustworthy,
and I then ran a separate parameter sweep to settle the simulated annealing
settings for the deck builder.

The first thing I saw when I pasted in the Phase B run was a clear
overfitting signature: `val_acc` hit 0.700 at epoch 1, then the training
loss steadily dropped for ten more epochs while the validation accuracy went
nowhere and patience fired. That pattern — train improving, val stalling —
would normally mean "encoder fine-tuning doesn't help." But Claude flagged
that three separate bugs were all independently pushing `val_acc` downward,
so I couldn't trust the verdict yet.

The first confounder was the drift metric. The `embedding_drift` values
showed a constant ~7.7 across all eleven epochs, which Claude recognized as
the frozen artifact of a train-vs-eval dropout delta that had already been
diagnosed and fixed locally. That meant there was no real read on whether
the encoder weights were actually moving at all.

The second — and more consequential — confounder was a dropout distribution
mismatch. The cached `.npz` files the Phase A scorer was tuned on were
produced with the encoder in `eval()` mode (no dropout). Phase B ran the
encoder in `train()` mode, meaning the scorer received a noisier version of
its training inputs on every batch. That alone could explain why val accuracy
never beat epoch 1: the scorer was perpetually re-fitting against a moving
target even before any encoder weight movement was considered.

The third issue was that gradient clipping at `max_norm=1.0` was suppressing
the effective learning rate. Pre-clip scorer norms were running between 8 and
19, so the actual step sizes were a tenth of what the learning rate implied.

I asked Claude whether we could detect norm spikes rather than just average
them, and whether it made sense to relax or remove clipping entirely. The
answer was that the current code logged only the last batch's norm, which
could hide a spike-then-normalize pattern. Claude proposed tracking per-batch
mean and max, and setting `--max-grad-norm` to a high value like 100 rather
than removing it entirely — a high-but-finite clip catches NaN-inducing
catastrophes without throttling normal training. The fixes — switching to
`encoder.eval()` during Phase B and adding `--max-grad-norm` — went in
before committing.

Later in the day I switched gears entirely to look at simulated annealing
parameters for the deck builder. I had run six versions of `build-decks` on
the same 12 pools, varying temperature (0.5 or 0.8), cooling rate (0.95 or
0.98), and restart count (1 or 4). Claude tabulated the per-pool scores and
summarized the findings. The dominant lever turned out to be restarts: going
from 1 to 4 restarts at `T=0.8, cooling=0.95` raised the mean pool score
from 2.734 to 2.770, and it eliminated the worst-case collapse that had
appeared on pool 11 with a single restart. Slower cooling (0.98) was a
regression on this scorer — it held the temperature high long enough that
exploration overshot good basins, and the best-deck tracker couldn't recover.

The gen-1 sweep had identified T=0.8 as the best single setting; the gen-2
results were consistent. Claude noted that the brittleness had migrated from
pool 12 in gen-1 to pool 11 in gen-2, and interpreted that as the failure
mode tracking the scorer's loss surface rather than being fixed to a
particular pool — which makes sense since a different checkpoint will
carve up the score landscape differently.

The per-pool oracle (best result across all six runs) was only 0.010 above
the four-restart setting alone, confirming that further temperature sweeping
at fixed restarts was unlikely to help much. The recommendation was
`T=0.8, cooling=0.95, restarts=4` for gen-2 deck generation. The open
caveat from the gen-1 write-up still stands: the scorer is miscalibrated
toward multi-color decks, so higher scores may reflect exploiting that
miscalibration rather than genuinely better decks. Win rate against Forge
still needs to be checked.
