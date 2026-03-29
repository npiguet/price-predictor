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
- `available_flag` is an **input feature only** — it informs the transformer that a slot has been picked but does NOT prevent the model from selecting it. Stage 1 teaches avoidance through the reward signal, not through masking.

---

### Episode

One complete or terminated pick sequence. Stored compactly in the replay buffer.

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `pool_names` | str | — | Semicolon-separated card names (90 booster cards, same order as pools.txt line) |
| `shuffle_seeds` | int32 | (40,) | RNG seed used to shuffle the non-basic-land portion before each pick step |
| `actions` | int32 | (n_picks,) | **Pool indices** of the card picked at each step. Length = `current_run` (≤40) |
| `log_probs` | float32 | (n_picks,) | Log-probability of each selected action under the policy at collection time |
| `reward` | float32 | scalar | `(current_run / best_run) × 2 - 1` at the time of episode completion |

**Critical distinction — pool index vs. shuffled input position**:

The model operates on a shuffled view of the pool at each pick step. It outputs logits over 96 *shuffled input positions*. However, the recorded action is the **pool index** (the card's fixed position in the original pool, 0–89 for booster cards, 90–95 for basic land slots), not the shuffled input position it happened to occupy at that step.

Example: card at pool index 3 is placed at shuffled input position 20 at step 1. Card at pool index 45 is also placed at shuffled input position 20 at step 2 (different shuffle). Both picks are legal — they refer to different pool cards. But if pool index 5 is picked twice (once from input position 35, once from input position 1 due to a different shuffle), that is an illegal pick on the second occurrence.

The shuffle seed allows reconstructing the input-position → pool-index mapping at any step, which is how the PPO trainer recovers the log-probability of stored actions from the current policy during training.

**Invariants**:
- `actions` and `log_probs` have the same length (= number of legal picks made, minimum 1)
- `reward` is in `[-1.0, 1.0]`
- `shuffle_seeds` always has exactly 40 elements (one per potential pick step); unused seeds (beyond `current_run`) are ignored
- Each value in `actions` is a pool index (0–95), not a shuffled input position

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
| `best_run` | int | High-water mark of `current_run` across all episodes. Initialized to 1, never decremented |
| `episode_count` | int | Total episodes run since training began (not reset on resume) |
| `consecutive_successes` | int | Number of consecutive episodes with `current_run == 40`. Reset to 0 when a sub-40 episode occurs |
| `pool_dataset_index` | int | NOT persisted. Always resets to 0 on resume (see clarification) |
| `reward_baseline` | float32 | Exponential moving average of episode rewards (decay 0.99). Persisted for continuity |

---

### CheckpointData

The complete serialized state of a training run.

| Field | Type | Notes |
|-------|------|-------|
| `pool_transformer_state_dict` | dict | PyTorch state dict for the pool transformer |
| `optimizer_state_dict` | dict | PyTorch optimizer state |
| `training_state` | TrainingState | `best_run`, `episode_count`, `consecutive_successes`, `reward_baseline` |
| `replay_buffer` | list[Episode] | Full serialized replay buffer |

---

## State Transitions

### Episode lifecycle

```
start_episode(pool)
    → for each pick step (up to 40):
        apply shuffle_seed[step] → permutation mapping input_position → pool_index
        build input tensor in shuffled order (slot flags reflect current pool state)
        forward_pass → logits[96]  ← no masking applied; logits are over shuffled input positions
        sample shuffled_input_position from full distribution
        translate: pool_index = permutation[shuffled_input_position]
        log_prob = log_probs_over_input_positions[shuffled_input_position]
        if pool_index already in picked_set (non-basic-land):
            terminate → compute reward → store Episode(actions=pool_indices, ...)
        else:
            add pool_index to picked_set
            record (pool_index, log_prob)
            update slot flags for pool_index
    → if 40 picks completed:
        compute reward → store Episode
```

### best_run update

```
at episode end:
    current_run = len(episode.actions)
    if current_run > best_run:
        best_run = current_run
    consecutive_successes = (consecutive_successes + 1) if current_run == 40 else 0
    if consecutive_successes >= 100:
        → Stage 1 complete
```

### Training batch

```
collect batch_size episodes sequentially from pool dataset
add each to replay_buffer (FIFO eviction as needed)
sample batch from replay_buffer
for each episode in batch:
    reconstruct episode states (pool + seeds)
    compute new log_probs for stored actions
    compute KL divergence (warn if > 1.5 nats)
    compute per-step importance ratios
    compute PPO loss
gradient update (pool transformer)
save latest checkpoint
print batch summary line
```
