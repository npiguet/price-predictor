# Sunday, 22 March 2026 — Ordinal regression experiment reverted

**TL;DR:** I tried giving each ordinal threshold head its own independent
Linear→ReLU→Linear path instead of a shared hidden layer. It made no
difference in practice, so I reverted.

The idea coming in was that the prior ordinal regression experiment had
failed partly because all K threshold heads shared one hidden layer —
only the final projection was head-specific. If each head instead learned
its own projection from the pooled representation, the heads might
specialise in different parts of the feature space and squeeze more signal
out of the expensive-card tail. The thresholds were placed at 0.5×3^k EUR
in shifted-log space, and soft CDF labels via the standard normal CDF
replaced the hard 0/1 targets from before.

Implementation was clean: a `self.heads` ModuleList with `n_thresholds`
independent Sequential blocks, a `self.thresholds` registered buffer,
`_encode()` factored out as shared backbone, `forward()` doing
probit-inverse weighted reconstruction, and `forward_ordinal()` returning
raw logits for `BCEWithLogitsLoss`. Two CLI flags exposed the knobs:
`--n-thresholds` (default 8) and `--ordinal-sigma` (default 0.75).

After a few training runs, the results were no better. When I floated the
idea of adding domain features to break the ceiling, it turned out
`is_reserved` and `is_abu` were already in the meta vector, already
flowing into every head. The model had the signal; the problem is data
volume. There are only around 85 expensive training cards, and that hard
ceiling holds regardless of head architecture or loss function.

We'd now tried Huber loss, ordinal BCE with shared hidden, ordinal BCE
with per-head projections, weighted sampling, max/mean/concat pooling,
and explicit domain features — all hitting the same wall. I reverted
everything back to the Huber baseline.
