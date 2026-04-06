# Data Model: Recoverability-Based Per-Step Stage 2 Loss

## New Value Objects

### RecoverabilityState

Immutable snapshot of the deck's mana recoverability at a given point in the episode. Not persisted — computed transiently during post-episode reward replay.

| Field | Type | Description |
|-------|------|-------------|
| `pip_counts` | `PipCounts` | Running pip demand from all non-land spells picked so far |
| `actual_sources` | `dict[str, float]` | Running count of mana sources per color from lands picked so far |

**Not a formal dataclass** — this is the implicit state tracked by the replay loop. Documented here for clarity.

### ShapingSignal (computed value, not a class)

| Condition | Signal |
|-----------|--------|
| `pip_counts` empty OR `actual_sources` empty | `0` |
| `imbalance_before < 3` and pick reduces imbalance | `+0.5` |
| `imbalance_before < 3` and pick increases imbalance | `-0.5` |
| `imbalance_before >= 3` and pick reduces imbalance | `+1.0` |
| `imbalance_before >= 3` and pick increases imbalance | `-1.0` |
| Pick does not change imbalance | `0` |

Positive signal = pick improved mana balance. Negative signal = pick worsened mana balance.

## Modified Entities

### Episode (existing — `replay_buffer.py`)

No schema change. The `step_rewards` field already holds per-step float32 values. The change is purely in how these values are computed:

| Before (feature 013) | After (feature 016) |
|-----------------------|---------------------|
| `step_rewards[t] = mana_score.reward` (uniform for all t) | `step_rewards[t] = budget_reward[t] + shaping_signal[t]` (per-step) |
| `reward = mana_score.reward` (scalar) | `reward = mana_score.reward` (scalar, unchanged — used only for logging) |

The `reward` field continues to hold the end-of-episode mana score for convergence checking and logging. Only `step_rewards` changes.

### ManaScore (existing — `mana_scorer.py`)

No change. Still computed once per episode for convergence checking. The shaping signal is a separate, additive computation.

## New Functions

### `compute_per_step_rewards()` — `mana_scorer.py`

Primary new function. Given a complete episode, replays the picks to compute per-step combined rewards.

**Signature**:
```python
def compute_per_step_rewards(
    actions: np.ndarray,          # int32[40] pool indices
    pool_names: str,              # semicolon-separated booster card names
    card_port: CardEmbeddingPort, # for is_land() and get_card_text()
    budget_rewards: np.ndarray,   # float32[40] from episode runner (+1/-1)
) -> PerStepRewardResult:
```

**Returns**: A result object containing:
- `step_rewards: np.ndarray` — float32[40], each in [-2, 2]
- `mean_shaping: float` — batch diagnostic: mean of shaping signals
- `final_imbalance: float` — batch diagnostic: imbalance at last step

**Algorithm** (40 iterations):
1. Initialize running pip demand (empty), actual sources (empty)
2. For each pick in `actions`:
   a. Compute `imbalance_before` from current state
   b. Update state: if spell → add pips; if land → add source
   c. Compute `imbalance_after` from updated state
   d. Determine discrete shaping based on rules table above
   e. `step_rewards[t] = budget_rewards[t] + shaping`

## State Transitions

```
Step 0 (no picks yet):
  pip_demand = {}, actual_sources = {}
  → shaping = 0 (no pip demand yet)

Step t (spell picked, no lands yet):
  pip_demand updates, actual_sources still empty
  → shaping = 0 (no mana supply yet)

Step t (land picked, both pip_demand and actual_sources non-empty):
  imbalance_before computed from ideal + actual_sources
  actual_sources += colors_from_land
  imbalance_after computed from ideal + updated actual_sources
  → shaping based on imbalance_before threshold and change direction

Step t (spell picked, both non-empty):
  imbalance_before computed from ideal + actual_sources
  pip_demand += pips_from_card → ideal shifts
  imbalance_after computed from new ideal + actual_sources
  → shaping based on imbalance_before threshold and change direction
```

## Validation Rules

- `step_rewards[t]` must be in [-2, 2] for all t (sum of ±1 budget + discrete shaping in {-1, -0.5, 0, +0.5, +1})
- `shaping_signal[t]` must be in {-1.0, -0.5, 0.0, +0.5, +1.0} for all t
