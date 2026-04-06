# Data Model: Recoverability-Based Per-Step Stage 2 Loss

## New Value Objects

### RecoverabilityState

Immutable snapshot of the deck's mana recoverability at a given point in the episode. Not persisted — computed transiently during post-episode reward replay.

| Field | Type | Description |
|-------|------|-------------|
| `pip_counts` | `PipCounts` | Running pip demand from all non-land spells picked so far |
| `actual_sources` | `dict[str, float]` | Running count of mana sources per color from lands picked so far |
| `remaining_picks` | `int` | Number of picks still to come including the current one (40 − current_step). Decremented by 1 after each pick. |

**Not a formal dataclass** — this is the implicit state tracked by the replay loop. Documented here for clarity.

### RecoverabilityRatio (computed value, not a class)

| Field | Type | Formula |
|-------|------|---------|
| `imbalance` | `float` | L1 distance: `sum(\|ideal[c] - actual[c]\|)` across all 6 colors |
| `ratio` | `float` | `imbalance / max(remaining_picks, 1) ** exponent` |

- When `remaining_picks == 0`: `ratio = imbalance` (denominator degenerates to 1^exp = 1, but using `max(remaining, 1)` handles this naturally)
- When `imbalance == 0`: `ratio = 0` regardless of remaining picks

### ShapingSignal (computed value, not a class)

| Field | Type | Formula |
|-------|------|---------|
| `delta` | `float` | `ratio_before - ratio_after` (reduction in recoverability ratio) |
| `signal` | `float` | `tanh(delta / temperature)`, bounded to (-1, 1) |

Positive signal = pick improved recoverability. Negative signal = pick worsened recoverability.

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
    urgency_exponent: float = 2.0,
    temperature: float = 1.0,
) -> PerStepRewardResult:
```

**Returns**: A result object containing:
- `step_rewards: np.ndarray` — float32[40], each in (-2, 2)
- `mean_shaping: float` — batch diagnostic: mean of shaping signals
- `final_imbalance: float` — batch diagnostic: imbalance at last step

**Algorithm** (40 iterations):
1. Initialize running pip demand (empty), actual sources (empty), remaining = 40, step = 0
2. For each pick in `actions`:
   a. Compute `ratio_before` from current state (using `remaining`)
   b. Update state: if spell → add pips; if land → add source. Decrement `remaining` by 1.
   c. Compute `ratio_after` from updated state (using decremented `remaining`)
   d. `delta = ratio_before - ratio_after`
   e. `shaping = tanh(delta / temperature)`
   f. `step_rewards[t] = budget_rewards[t] + shaping`
   g. Advance step

### `compute_recoverability_ratio()` — `mana_scorer.py`

Pure computation helper.

**Signature**:
```python
def compute_recoverability_ratio(
    pip_counts: PipCounts,
    actual_sources: dict[str, float],
    remaining_picks: int,
    exponent: float = 2.0,
) -> float:
```

**Algorithm**:
1. `ideal = compute_ideal_distribution(pip_counts)`
2. `imbalance = sum(|ideal[c] - actual[c]|)` for all 6 colors
3. `denominator = max(remaining_picks, 1) ** exponent`
4. Return `imbalance / denominator`

## State Transitions

```
Step 0 (no picks yet):
  pip_demand = {}, actual_sources = {}, remaining = 40
  → ratio_before = 0 (imbalance is 0 because ideal is {} when no spells picked)
  After pick: remaining = 39
  → ratio_after = 0 (still zero imbalance if first pick is a land with no demand)
  → shaping ≈ 0

Step t (spell picked):
  ratio_before computed with remaining = 40 - t
  pip_demand += pips_from_card, remaining decremented to 40 - t - 1
  → ideal shifts, imbalance may increase or decrease
  → ratio_after computed with remaining = 40 - t - 1

Step t (land picked):
  ratio_before computed with remaining = 40 - t
  actual_sources += colors_from_land, remaining decremented to 40 - t - 1
  → imbalance decreases (if land matches ideal) or increases (if oversupplied)
  → ratio_after computed with remaining = 40 - t - 1

Step 39 (final pick):
  remaining_before = 1 (40 - 39), remaining_after = 0 (40 - 39 - 1)
  → ratio_after uses denominator max(0, 1)^exp = 1
  → delta = ratio_before - imbalance_after
```

## Validation Rules

- `step_rewards[t]` must be in (-2, 2) for all t (sum of ±1 budget + bounded (-1,1) shaping)
- `shaping_signal[t]` must be in (-1, 1) for all t (tanh output)
- `urgency_exponent` must be > 0 (default 2.0)
- `temperature` must be > 0 (default 1.0)
- `remaining_picks` is in [0, 39] — never negative

## Hyperparameter Configuration

Two new CLI arguments on the `train --stage 2` command:

| Argument | Type | Default | Maps to |
|----------|------|---------|---------|
| `--urgency-exponent` | float | 2.0 | `urgency_exponent` in `compute_recoverability_ratio()` |
| `--temperature` | float | 1.0 | `temperature` in `compute_per_step_rewards()` |

Help text must include `(default: <value>)` suffix, matching the existing CLI convention:
- `--urgency-exponent`: `"Exponent for the recoverability ratio denominator (default: 2.0)"`
- `--temperature`: `"Temperature for the tanh shaping bounding function (default: 1.0)"`

Passed through: CLI → `run_train()` → `TrainStage2UseCase.execute()` → `compute_per_step_rewards()`.
