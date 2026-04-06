# Research: Recoverability-Based Per-Step Stage 2 Loss

## R1: Computation Approach — Post-Episode Replay vs. Inline Rollout

**Decision**: Post-episode replay in `train_stage2.py`, not inline in `EpisodeRunner`.

**Rationale**: `EpisodeRunner` currently has zero knowledge of card text, mana costs, or pip analysis — it only knows about embeddings and action masking. Threading `card_port` text lookups into the runner would couple the generic pick-and-record loop to domain-specific mana scoring. The current architecture already establishes the post-episode pattern: `_compute_episode_mana_score()` reconstructs the final deck state by replaying picks. Extending this to compute intermediate states is straightforward — iterate through `ep.actions` one pick at a time, maintaining running pip totals and actual source counts. The cost is negligible: pure Python arithmetic on 6-element dictionaries, 40 times per episode.

**Alternatives considered**: Inline computation in `EpisodeRunner.run()` would avoid the replay step but would violate the clean separation between the generic episode runner (domain) and mana-specific scoring (application/domain boundary). The PPO trainer already replays pick sequences step-by-step to reconstruct states for training, so replay-based computation is a proven pattern in this codebase.

## R2: ~~tanh Bounding Function~~ → Discrete Shaping Signal

**Decision**: ~~Use `tanh(delta / temperature)`.~~ **Superseded (2026-04-06)**: replaced with discrete shaping.

**Original rationale**: tanh was chosen as a smooth squashing function. However, in practice the continuous signal produced magnitudes ~0.001 for early picks due to the `remaining^exponent` denominator, resulting in a constant `shaping=-0.07` and no learning.

**Revised approach**: Discrete signal: shaping = 0 when no pip demand or no mana supply; ±0.5 when imbalance < 3; ±1.0 when imbalance >= 3. This provides clear, strong directional feedback that PPO can learn from.

## R3: ~~Potential-Based Reward Shaping (PBRS) Analysis~~ → Simple Reward Assignment

**Decision**: ~~The delta formulation is PBRS-compliant.~~ **Superseded (2026-04-06)**: the discrete signal is not PBRS.

**Original rationale**: The continuous ratio-based delta matched the Ng et al. (1999) PBRS form.

**Revised note**: The discrete signal is a simple reward assignment, not a potential-based function. PBRS policy-invariance guarantees no longer apply. In practice this is acceptable: the signal is bounded, the magnitudes are controlled, and PPO's advantage normalization handles any constant offset.

## ~~R4: Remaining Picks = 0 Edge Case~~ (REMOVED)

**Superseded (2026-04-06)**: The discrete shaping signal does not use `remaining_picks`. No edge case to handle.

## R5: PPO Interaction with Per-Step Rewards

**Decision**: Current PPO hyperparameters (clip_eps=0.2, entropy_coef=0.01) are adequate. No GAE needed initially.

**Rationale**: The PPO trainer already normalizes step rewards across the entire batch by subtracting the mean and dividing by std before computing advantages (`ppo_trainer.py` lines 51-57). This prevents steps with large rewards from dominating the gradient. The clipping ratio constrains policy updates relative to the old policy, which is orthogonal to reward variance. Both hyperparameters are standard PPO values.

**No GAE needed initially**: The current implementation computes single-step advantages directly. With uniform rewards this was fine. With per-step rewards, you lose backward propagation of future reward information. However, the recoverability ratio's urgency exponent already encodes temporal information (early picks get small signals, late picks get large signals), making GAE less critical. If the model struggles to make good early-episode picks, GAE (lambda=0.95) would be the natural next step.

**Watch item**: If entropy collapse recurs, the entropy coefficient (0.01) can be bumped to 0.02–0.05, but this is independent of the reward shaping change and should be tuned separately.
