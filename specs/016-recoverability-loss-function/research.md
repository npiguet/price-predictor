# Research: Recoverability-Based Per-Step Stage 2 Loss

## R1: Computation Approach — Post-Episode Replay vs. Inline Rollout

**Decision**: Post-episode replay in `train_stage2.py`, not inline in `EpisodeRunner`.

**Rationale**: `EpisodeRunner` currently has zero knowledge of card text, mana costs, or pip analysis — it only knows about embeddings and action masking. Threading `card_port` text lookups into the runner would couple the generic pick-and-record loop to domain-specific mana scoring. The current architecture already establishes the post-episode pattern: `_compute_episode_mana_score()` reconstructs the final deck state by replaying picks. Extending this to compute intermediate states is straightforward — iterate through `ep.actions` one pick at a time, maintaining running pip totals and actual source counts. The cost is negligible: pure Python arithmetic on 6-element dictionaries, 40 times per episode.

**Alternatives considered**: Inline computation in `EpisodeRunner.run()` would avoid the replay step but would violate the clean separation between the generic episode runner (domain) and mana-specific scoring (application/domain boundary). The PPO trainer already replays pick sequences step-by-step to reconstruct states for training, so replay-based computation is a proven pattern in this codebase.

## R2: tanh Bounding Function

**Decision**: Use `tanh(delta / temperature)` as specified.

**Rationale**: tanh is a standard squashing function for bounding RL signals (used as the default output activation in SAC, commonly applied in reward normalization). Two practical advantages over hard clipping: (1) smooth non-zero gradients everywhere — the learning signal never goes fully dead even at extreme values, and (2) near-linear behavior around zero (tanh(x) ≈ x for |x| < 0.5) means small deltas pass through with minimal distortion, giving proportional feedback for routine picks. Saturation at extremes (approaching ±1 for |x| > 2) prevents catastrophic picks from producing outsized reward signals that destabilize training.

**Tuning note**: With temperature=1, deltas above ~1.5 in absolute value start saturating. Since the recoverability ratio involves division by remaining_picks^2, raw deltas will typically be small numbers (especially early). If the shaping signal is almost always near-zero during initial training, lowering the temperature amplifies the linear region's sensitivity.

**Alternatives considered**: Hard clipping (no gradient at boundaries), sigmoid (asymmetric range [0,1] — awkward for a signal that should be symmetric around zero), linear with clamping (sharp gradient discontinuity at boundaries).

## R3: Potential-Based Reward Shaping (PBRS) Analysis

**Decision**: The delta formulation is PBRS-compliant in practice.

**Rationale**: Define Φ(s) = −recoverability_ratio(s). Then the shaping reward F(s, s') = Φ(s') − Φ(s) = ratio_before − ratio_after. This matches the Ng et al. (1999) PBRS form exactly. The theorem proves this preserves optimal policy invariance — the shaped reward does not change which policy is optimal.

**Theoretical caveat**: PBRS requires Φ(terminal) = 0, but at the terminal state (remaining_picks=0) the ratio equals raw imbalance (not zero). In practice this does not matter: (1) the tanh bounding prevents terminal potential bias from creating runaway reward accumulation, (2) PPO's advantage normalization subtracts the batch mean, absorbing constant offsets, and (3) this is episodic RL with no discounting where terminal potential bias compounds.

**Alternatives considered**: Subtracting terminal potential from all steps (theoretically pure but adds complexity for no practical gain — advantage normalization already handles it).

## R4: Remaining Picks = 0 Edge Case

**Decision**: No special handling needed. The spec's degenerate case is correct.

**Rationale**: At remaining_picks=0 the episode is over — no reward needs to be computed for a nonexistent pick 41. The ratio at the terminal state is only ever used as the "after" value for pick 40's delta (transitioning from remaining_picks=1 to remaining_picks=0). That delta is well-defined: ratio_before = imbalance / 1^exp = imbalance, ratio_after = imbalance (raw). The result is simply imbalance_before − imbalance_after, which is meaningful and finite.

**Implementation note**: The computation loop runs for exactly 40 steps. No guard clause is needed for remaining_picks=0 as a "before" state.

## R5: PPO Interaction with Per-Step Rewards

**Decision**: Current PPO hyperparameters (clip_eps=0.2, entropy_coef=0.01) are adequate. No GAE needed initially.

**Rationale**: The PPO trainer already normalizes step rewards across the entire batch by subtracting the mean and dividing by std before computing advantages (`ppo_trainer.py` lines 51-57). This prevents steps with large rewards from dominating the gradient. The clipping ratio constrains policy updates relative to the old policy, which is orthogonal to reward variance. Both hyperparameters are standard PPO values.

**No GAE needed initially**: The current implementation computes single-step advantages directly. With uniform rewards this was fine. With per-step rewards, you lose backward propagation of future reward information. However, the recoverability ratio's urgency exponent already encodes temporal information (early picks get small signals, late picks get large signals), making GAE less critical. If the model struggles to make good early-episode picks, GAE (lambda=0.95) would be the natural next step.

**Watch item**: If entropy collapse recurs, the entropy coefficient (0.01) can be bumped to 0.02–0.05, but this is independent of the reward shaping change and should be tuned separately.
