# April 4, 2026 — Transformer counting limits, pivot to explicit mana features

**TL;DR:** A day of diagnosing why the embedding validation probes were
failing on pip counts and mana value, tracing it to a fundamental
architectural limitation, and ultimately deciding to bypass the transformer
entirely for those features.

The day started with feature 015 training live — auxiliary supervision with
20 heads producing card color, pip counts, mana value, and mana production
labels alongside the main price task. The first run crashed immediately with
a device mismatch (loss functions on CPU, model on CUDA), fixed trivially.
Watching the loss curves that followed was useful: color detection heads
learned fast and plateaued, while pip count and mana value kept lagging
behind.

Once training finished, validate-embeddings gave a clear split. Is-land and
card color were at >99%. Mana produced was at >99%. But pip counts were
stuck in the 85–92% exact-match range, and mana value was R²=0.669 with
exact match at only 46%. That gap — detection passing cleanly while
counting failed — prompted the right question: is this a detection vs
counting problem at the architectural level?

The answer is yes. Attention is relational and naturally good at “is there a
W pip here?” Max pooling captures the peak activation across all positions,
which is exactly what detection needs. But “how many W pip tokens appear?”
requires accumulating across positions, which is not what attention or
max/mean pooling is built for. Mean pooling is diluted by card length. Sum
pooling was tried next — it removes that dilution, accumulating activations
linearly — but it also accumulates W mana cost symbols in oracle text (like
tap abilities that add W) on equal footing with the mana cost line, so
Plains ends up with more W-signal than a W spell, which is exactly backwards
for mana value.

Going from 2 to 4 transformer layers was tried next, on the hypothesis that
deeper networks could learn the multi-step computation (identify token type,
contextualize, aggregate). That did nothing. The plateau was the same.

At that point Claude identified a second problem hiding in the numbers: the
training used K-logit ordinal heads (17 softmax outputs for mana value), but
the probe was fitting a scalar LinearRegression. A softmax output is
nonlinear — a single linear regression cannot decode it properly. The binary
detection heads all used single-logit outputs matched by a logistic probe,
which is why those worked. The probe was changed to multinomial
LogisticRegression, matching the training head structure.

That fixed the Score vs Exact Match discrepancy (they had been inconsistent,
betraying that the wrong probe branch was running), but the underlying mana
value accuracy still fell short of the 90% threshold.

The conclusion was architectural: transformers are good at detection, not
counting. Rather than fighting it, the decision was to stop asking the
transformer to count at all. Pip counts per color (W/U/B/R/G/C), generic
mana, X pips, mana value, and mana produced per color (six more) — 15
features total — would be parsed directly from card text using the existing
deterministic functions and appended to the meta vector for price predictor
training, and to the card embedding for sealed consumption. The transformer
handles semantics; the explicit features handle arithmetic. Mana efficiency
is a significant price driver anyway (the same effect at a lower mana cost
can be worth an order of magnitude more), so injecting these features into
the price regression head directly should also improve price prediction.

The sum pooling was reverted at the same time — its purpose was counting, and
that problem was now solved differently. meta_dim went from 15 to 30. Card
embeddings changed from 2*d_model (512 floats) to 2*d_model + 15 (527
floats). The sealed Pool Transformer’s d_model padded up to 536 to maintain
head-divisibility.

A subtle bug surfaced late: _build_base_tensor in episode_runner.py was
computing d_model as embed_dim + 8, which gives 535 for 527-dim embeddings,
while PoolTransformerConfig.from_embed_dim gives d_model=536 after ceiling
to the nearest multiple of n_heads=8. The 1-dim gap would have caused a
runtime failure with real embeddings. Tests had not caught it because the
test fixtures used 8-dim embeddings where 8+8=16 is already cleanly
divisible by 2.
