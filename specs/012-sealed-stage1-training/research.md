# Research: Stage 1 Training — Legal Pick Gate

**Feature**: 012-sealed-stage1-training
**Date**: 2026-03-28

---

## Decision 1: PPO Variant (with or without critic)

**Decision**: Critic-free PPO. Use the episode reward directly as the advantage signal (optionally centered by a running-mean baseline). The clipped surrogate objective applies per-step importance ratios: `r_t = exp(new_log_p_t - old_log_p_t)`, advantage = episode reward (same for all 40 steps).

**Rationale**: Stage 1's reward is a single scalar per episode (not per-step), so per-step advantages collapse to a constant anyway. A running-mean baseline reduces variance cheaply without introducing a critic head. The clipped surrogate `min(r * A, clip(r, 1-ε, 1+ε) * A)` prevents excessively large gradient steps. Standard on-policy PPO: each batch of episodes is collected fresh, trained on once, and discarded.

**Alternatives considered**:
- Full PPO with critic + GAE: Would require adding a value head to the model. Over-engineered for a single-scalar reward signal at this stage.
- Pure REINFORCE: No clipping, higher variance. Rejected in favor of clipped surrogate.
- PPO with replay buffer (off-policy): Originally specced, removed in favour of standard on-policy PPO to avoid the complexity of off-policy corrections and stale-episode detection.

---

## Decision 3: Pool Transformer Positional Encoding

**Decision**: No positional encoding. The pool is reshuffled before every pick step, making absolute position semantically meaningless. The 4 per-slot flags (particularly `is_land` and `basic_land_count`) provide sufficient slot identity. Basic land slots (always at the end, positions 90–95) are distinguished by `is_land=1` and `basic_land_count` values.

**Rationale**: Adding positional encoding would introduce positional biases that the shuffle is explicitly designed to prevent. The transformer's attention mechanism can attend freely over all 96 slots without position information; cards are distinguished by their embeddings and flags alone.

**Alternatives considered**:
- Learned slot embeddings (one per position): Could help distinguish basic land positions, but the flags already encode this and adding 96 learned vectors increases parameter count unnecessarily.
- Sinusoidal positional encoding: Wrong semantic signal for a shuffled set-like input.

---

## Decision 4: Action Masking in All Stages

**Decision**: Logits for already-picked booster slots (`available_flag = 0`) are set to −1e9 before softmax in both
the episode runner and the PPO trainer. This makes duplicate picks structurally impossible. Every episode always
completes all `best_run` steps. Basic land slots are never masked (their `available_flag` is never cleared).

**Rationale**: Without masking, duplicate avoidance and mana/spell-count optimization compete for the same model
parameters, causing destructive oscillation: the model learns mana coherence, then forgets to avoid duplicates,
then relearns that, then forgets mana — repeating in cycles. With masking, these concerns are cleanly separated:
duplicate avoidance is handled structurally, and every gradient step focuses entirely on the spell/land mix quality.

A subtler concern raised during design: "doesn't masking prevent the model from learning?" The answer is no. Without
masking, the model must learn "even though I REALLY want this card, its flag is 0 so I cannot pick it again" — a
conditional suppression that fights directly against the preference signal coming from the same card's embedding. The
useful thing masking actually teaches is a *ranking*: by forcing the model to express a preference among the remaining
available cards rather than collapsing back to already-picked favorites, the model's outputs become a ranking of
available cards by desirability. That ranking is exactly what Stage 2 needs.

Stage 1 levels 1–16 are skipped (`best_run` starts at 17) because with masking they are trivially achievable from
episode 1 — there is nothing to learn at those levels.

**Why −1e9 instead of −∞**: Using `float('-inf')` produces `exp(-inf) = 0`, then `log_softmax` gives `−inf` for
masked positions, and `−inf × 0 = NaN` in the entropy term (`−Σ p log p`). A large finite value like −1e9 avoids
this: `exp(-1e9) = 0.0` in float32 (safe underflow), but `−1e9 × 0.0 = 0.0` (no NaN).

**Alternatives considered**:
- No masking + terminate on duplicate: Causes destructive oscillation between duplicate avoidance and mana learning.
  Rejected after observing the oscillation in practice.
- Mask only during evaluation: Inconsistent with training distribution; model would behave differently at inference.

---

## Decision 6: Episode Reward Assignment to Steps

**Decision**: All 40 pick steps within an episode share the same advantage value: `A = reward - baseline`, where `baseline` is an exponential moving average of all prior episode rewards (decay 0.99). Per-step importance ratios are computed independently. The PPO loss is the mean over all steps of all episodes in the batch.

**Rationale**: The episode reward is a function of the full sequence, so all steps are equally responsible. A running EMA baseline reduces gradient variance with no extra parameters and no storage overhead.

**Alternatives considered**:
- Discount future steps with γ < 1: Inappropriate when the terminal reward is the only signal and earlier steps are not intrinsically better or worse.
- No baseline (raw reward): High variance, especially early in training when reward swings between -1 and +1.

---

## Decision 7: Testing Strategy

**Decision**: Unit tests use minimal tensor sizes (batch=2, pool_size=4, d_model=8, n_layers=1, n_heads=2) to keep them fast. The integration test runs a 2-batch training loop with 4 fake episodes per batch against a tiny fixture pools.txt and tiny .npz embeddings, verifying: checkpoint is written, best_run is updated, and the completion condition works correctly.

**Rationale**: Full-size tensors (batch=32, pool=96, d_model=516, n_layers=8) are too slow for unit tests. Miniaturized models let all domain logic be tested in milliseconds. The integration test uses real file I/O and the full application call path (CLI → use case → domain → infrastructure) to catch wiring bugs.

**Alternatives considered**:
- Mock all I/O in unit tests: Appropriate for most tests but the integration test must use real .npz and pools.txt files to validate the loading pipeline.
