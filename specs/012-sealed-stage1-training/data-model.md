# Data Model: Stage 1 Training — Legal Pick Gate

**Feature**: 012-sealed-stage1-training
**Date**: 2026-03-28

---

## Entities

### PoolTransformerConfig

Configuration for the pool-level transformer model. Immutable once training begins.

| Field | Type | Value | Notes |
|-------|------|-------|-------|
| `n_layers` | int | 8 | Number of transformer encoder layers |
| `n_heads` | int | 8 | Number of attention heads |
| `d_model` | int | 516 | Input and output dimension per slot |
| `ff_dim` | int | 2048 | Feed-forward intermediate dimension |
| `n_slots` | int | 96 | Total pool size (90 cards + 6 basic land slots) |
| `card_embed_dim` | int | 512 | Dimension of the frozen card encoder output |
| `dropout` | float | 0.1 | Dropout rate |

---

### PoolSlot

The feature vector for one card slot in the pool. 516 floats total.

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `card_embedding` | float32 | (512,) | From pretrained encoder |
| `picked_flag` | float32 | scalar | 1.0 if already picked this episode, else 0.0. Basic land slots: 0.0 when count==0, 1.0 otherwise |
| `available_flag` | float32 | scalar | 1.0 if slot can be picked, else 0.0. Basic land slots: always 1.0 |
| `is_land` | float32 | scalar | 1.0 if the card is a land (including basic lands), else 0.0 |
| `basic_land_count` | float32 | scalar | How many times this basic land slot has been picked this episode. Always 0.0 for booster card slots |

**Invariants**:
- Booster card slots: `basic_land_count` is always 0.0
- Basic land slots (indices 90–95): `available_flag` is always 1.0
- Once a non-basic-land slot is picked: `available_flag` = 0.0, `picked_flag` = 1.0
- `available_flag = 0` causes the slot's logit to be set to −1e9 before softmax — the model **cannot** select
  already-picked booster slots. This is action masking. Every episode always completes all `best_run` steps.

---

### Episode

One complete or terminated pick sequence.

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `pool_names` | str | — | Semicolon-separated card names (90 booster cards, same order as pools.txt line) |
| `shuffle_seeds` | int64 | (best_run,) | RNG seed used to shuffle the booster portion before each pick step |
| `actions` | int32 | (best_run,) | **Pool indices** of the card picked at each step. Always exactly `best_run` entries |
| `log_probs` | float32 | (best_run,) | Log-probability of each selected action under the policy at collection time |
| `step_rewards` | float32 | (best_run,) | Per-step reward (+1/−1 based on spell/land budgets) |
| `reward` | float32 | scalar | `(effective_run / best_run) × 2 - 1` |
| `effective_run` | int | scalar | `best_run - max(n_spell-23, 0) - max(n_land-17, 0)` |

**Critical distinction — pool index vs. shuffled input position**:

The model operates on a shuffled view of the pool at each pick step. It outputs logits over 96 *shuffled input positions*. However, the recorded action is the **pool index** (the card's fixed position in the original pool, 0–89 for booster cards, 90–95 for basic land slots), not the shuffled input position it happened to occupy at that step.

Example: card at pool index 3 is placed at shuffled input position 20 at step 1. Card at pool index 45 is also placed at shuffled input position 20 at step 2 (different shuffle). Both picks are legal — they refer to different pool cards. But if pool index 5 is picked twice (once from input position 35, once from input position 1 due to a different shuffle), that is an illegal pick on the second occurrence.

The shuffle seed allows reconstructing the input-position → pool-index mapping at any step, which is how the PPO trainer recovers the log-probability of stored actions from the current policy during training.

**Invariants**:
- `actions`, `log_probs`, and `step_rewards` all have length exactly `best_run`
- `shuffle_seeds` has exactly `best_run` elements (one per pick step)
- `reward` is in `[-1.0, 1.0]`
- Each value in `actions` is a pool index (0–95), not a shuffled input position
- No duplicate booster slot indices appear in `actions` (guaranteed by action masking)

---

### ReplayBuffer

A capped FIFO queue of episodes.

| Field | Type | Notes |
|-------|------|-------|
| `max_size` | int | Maximum number of episodes retained (default: 1000) |
| `episodes` | deque[Episode] | Ordered oldest-first; FIFO eviction when `len == max_size` |

**Invariants**:
- When a new episode is appended and `len(episodes) == max_size`, the oldest episode is discarded first
- All stored episodes are immutable after insertion

---

### TrainingState

Mutable training progress state. Persisted in every checkpoint.

| Field | Type | Notes |
|-------|------|-------|
| `best_run` | int | Current curriculum level. Initialized to 17, advances by 1 when batch mean reward ≥ 0.90 |
| `episode_count` | int | Total episodes run since training began (not reset on resume) |

---

### CheckpointData

The complete serialized state of a training run.

| Field | Type | Notes |
|-------|------|-------|
| `pool_transformer_state_dict` | dict | PyTorch state dict for the pool transformer |
| `optimizer_state_dict` | dict | PyTorch optimizer state |
| `training_state` | TrainingState | `best_run`, `episode_count`, `consecutive_successes`, `reward_baseline` |

---

## State Transitions

### Episode lifecycle

```
start_episode(pool)
    → for each pick step (0 .. best_run-1):
        apply shuffle_seed[step] → permutation of booster slots
        build shuffled input tensor (basic land slots unchanged at end)
        forward_pass → logits[n_slots]
        apply action masking: logits[available_flag == 0] = −1e9
        sample shuffled_input_position from masked distribution
        translate: pool_index = permutation[shuffled_input_position]  (or pool_index = shuffled_input_position for basic lands)
        log_prob = log_softmax(logits)[shuffled_input_position]
        record (pool_index, log_prob, step_reward)
        update slot flags for pool_index
    → after best_run picks:
        compute reward = (effective_run / best_run) × 2 - 1
        store Episode(pool_names, shuffle_seeds, actions, log_probs, step_rewards, reward, effective_run)
```

### best_run update

```
after each training batch:
    if batch_mean_reward >= 0.90:
        best_run += 1
        if best_run > MAX_PICKS (40):
            → Stage 1 complete
```

### Training batch

```
collect batch_size episodes sequentially from pool dataset (on-policy)
for each episode in batch:
    reconstruct episode states (pool + seeds)
    compute new log_probs for stored actions
    compute per-step importance ratios
    compute PPO loss
gradient update (pool transformer)
save latest checkpoint
print batch summary line
```
