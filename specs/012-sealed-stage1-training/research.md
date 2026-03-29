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

## Decision 4: No Logit Masking in Stage 1

**Decision**: No selection mask is applied to logits during Stage 1. The pool transformer produces `(batch, 96)` logits and the model samples freely from the full distribution, including already-picked slots. The `available_flag = 0` on picked slots is present as an input feature (context for the transformer) but does NOT block selection. If the model picks an already-picked non-basic-land slot, the episode terminates and the reward signal penalizes that behavior.

**Rationale**: Masking makes illegal picks mechanically impossible, which would make Stage 1 trivially solved from episode 1 — there would be nothing for the model to learn. The entire purpose of Stage 1 is to teach the model to avoid illegal picks through reinforcement. The reward signal (`(current_run / best_run) × 2 - 1`) provides the necessary gradient: illegal picks end the episode early and yield a lower reward. The `available_flag` feature gives the model the information it needs to learn this avoidance, but the learning must happen through experience, not constraint.

**Alternatives considered**:
- Apply mask to logits (`masked_fill(-inf)`): Makes Stage 1 vacuous — correct behavior is enforced externally, model learns nothing. Rejected.
- Apply mask only during evaluation/sampling: Inconsistent with training distribution; model would behave differently at inference time.

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
