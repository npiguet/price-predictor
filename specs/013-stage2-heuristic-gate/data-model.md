# Data Model: Stage 2 Training — Heuristic Gate

**Feature**: 013-stage2-heuristic-gate | **Date**: 2026-03-31

## Entities

### PipCounts (value object, domain)

Per-color tally of mana requirements across all non-land cards in a deck.

| Field | Type | Description |
|-------|------|-------------|
| counts | dict[str, float] | Keys: "W", "U", "B", "R", "G", "C". Values: pip tally per color. |

**Rules**:
- Single-color pip `{W}` → +1.0 to that color
- Phyrexian pip `{W/P}` → +0.5 to that color
- Hybrid pip `{G/R}` → +0.5 to each color
- Generic `{1}`, `{2}`, `{X}` → ignored
- Colorless `{C}` → +1.0 to "C" (sixth color)
- For multi-face cards (transform, split, adventure), all faces' `mana cost:` lines are counted

### IdealDistribution (value object, domain)

Target number of mana sources per color, computed from pip counts.

| Field | Type | Description |
|-------|------|-------------|
| ideal | dict[str, float] | Keys: colors present in pips. Values: ideal source count per color. |

**Computation**:
```
colors_present = {c : pip_counts[c] > 0}
n_colors       = len(colors_present)
total_pips     = sum(pip_counts[c] for c in colors_present)
ideal[c]       = 2 + (17 - 2 * n_colors) * pip_counts[c] / total_pips   for c in colors_present
ideal[c]       = 0                                                         otherwise
```

### ActualSourceCounts (value object, domain)

Per-color tally of mana-producing lands in the deck.

| Field | Type | Description |
|-------|------|-------------|
| sources | dict[str, float] | Keys: "W", "U", "B", "R", "G", "C". Values: source count per color. |

**Rules**:
- Scan each land card's `activated[N]: {T}: add ...` lines
- Extract distinct color symbols `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}` from the add clause
- Each distinct color symbol → +1 to that color (per ability line)
- Dual lands (`add {G} or {U}`) → +1 G, +1 U
- Tri-lands (`add {R}, {G}, or {W}`) → +1 R, +1 G, +1 W
- `add one mana of any color` (no color symbols) → +0

### ManaScore (value object, domain)

Scalar quality measure of how well the deck's mana base matches its spell requirements.

| Field | Type | Description |
|-------|------|-------------|
| score | float | Value in [0.0, 1.0]. |
| reward | float | Value in [-1.0, 1.0]. Computed as `2 * score - 1`. |
| l1_error | float | Sum of |actual - ideal| per color. |
| n_lands | int | Total land count in the deck. |

**Computation**:
```
l1_error = sum(abs(actual[c] - ideal[c]) for c in all_colors)
score    = max(0.0, 1.0 - (l1_error + abs(n_lands - 17)) / 17.0)
reward   = 2 * score - 1
```

## Existing Entities (unchanged)

### Episode (domain/replay_buffer.py)

No structural changes. Stage 2 reuses the same Episode dataclass. Action masking guarantees every
episode always completes all 40 picks — there is no terminated/duplicate path. The application layer
overwrites `step_rewards` (uniform mana-score reward) and `reward` (mapped score) after each episode.

### TrainingState (application/train_stage1.py)

Reused for Stage 2 with `best_run=MAX_PICKS` (40, fixed) and `episode_count=0` at init.

### CheckpointData (infrastructure/pool_model_store.py)

No changes. Stage 2 checkpoints contain the same payload:
`{pool_transformer_state_dict, optimizer_state_dict, training_state}`.

## Extended Protocol

### CardEmbeddingPort (domain/card_embedding_port.py)

| Method | Existing | Description |
|--------|----------|-------------|
| get_embedding(card_name) -> np.ndarray | Yes | Card embedding vector |
| is_land(card_name) -> bool | Yes | Whether card is a land |
| get_card_text(card_name) -> str | **New** | Full card text content for mana analysis |

## State Transitions

### Stage 2 Training Run

```
[Start]
  │
  ├─ model-path exists? ──yes──> Resume: load full checkpoint (model + optimizer + state)
  │                                │
  └─ no                            │
     │                             │
     ├─ init-from exists? ──yes──> Init: load model weights only, fresh optimizer, episode_count=0
     │                             │
     └─ no ──> ERROR               │
                                   v
                            [Training Loop]
                                   │
                            Run batch of 32 episodes
                                   │
                            For each episode (always 40 picks, action masking):
                              └─ Compute mana score, assign uniform reward to all steps
                                   │
                            PPO update (normalise across batch)
                                   │
                            Save checkpoint
                                   │
                            All 32 score > 0.90? ──yes──> [Stage 2 Complete]
                              │
                              └─ no ──> next batch
```
