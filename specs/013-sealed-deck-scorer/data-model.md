# Data Model: Sealed Deck Scorer

**Feature**: 013-sealed-deck-scorer | **Date**: 2026-04-11

## Entities

### CardFeatureVector

A 544-dimensional numerical representation of a single card.

| Field | Type | Description |
|-------|------|-------------|
| embedding | float32[544] | Concatenation of text embedding [0:512] and deterministic features [512:544] |

**Storage**: One `.npz` file per card in the cards-path folder (e.g., `lightning_bolt.npz`). Contains a single array keyed as `"embedding"`.

**Validation rules**:
- Must be exactly 544 dimensions (512 text embedding + 32 deterministic features)
- Existing 512-dimensional files from feature 011 are NOT auto-upgraded; `--clean` flag required

### DeterministicFeatures

The 32 features at indices 512-543, parsed from converted card text.

| Index | Feature | Type | Source |
|-------|---------|------|--------|
| 512 | is_land | binary | `types:` line contains "land" |
| 513-517 | W/U/B/R/G pip counts | int | `mana cost:` line, count of each `{W}`, `{U}`, `{B}`, `{R}`, `{G}` |
| 518 | colorless pip count (C) | int | `mana cost:` line, count of `{C}` |
| 519 | generic mana | int | `mana cost:` line, numeric portion (e.g., `{2}` = 2) |
| 520 | X pip count | int | `mana cost:` line, count of `{X}` |
| 521 | mana value | int | Total mana value (sum of all pips and generic mana) |
| 522-526 | is_white..is_green | binary | 1 if corresponding pip count > 0, unless devoid |
| 527 | is_colorless | binary | 1 if no colored pips, or devoid, or no mana cost |
| 528-533 | produces W/U/B/R/G/C | binary | Parsed from `activated:` lines containing "add" |
| 534 | mana_count | int | Total mana produced per activation |
| 535 | power | float | From `power toughness:` line; `*`/`X` = 0 |
| 536 | toughness | float | From `power toughness:` line; `*`/`X` = 0 |
| 537 | loyalty | float | From `loyalty:` line; 0 for non-planeswalkers |
| 538-543 | zero padding | 0.0 | Reserved (make vector length divisible by 8) |

**Parsing rules**:
- Devoid detection: `static: devoid` line in converted card text
- No mana cost line: all mana cost features = 0
- Mana production: only `activated:` ability lines scanned for `add` patterns

### MatchOutcome

A parsed line from `match-outcomes.txt` (produced by feature 012).

| Field | Type | Description |
|-------|------|-------------|
| deck_a_names | list[str] | Pipe-separated card names for deck A |
| deck_b_names | list[str] | Pipe-separated card names for deck B |
| wins_a | int (0-2) | Games won by deck A |
| wins_b | int (0-2) | Games won by deck B |

**File format**: `deckA_card1|card2|...|card40;deckB_card1|...|card40;winsA;winsB`

**Derived fields**:
- `winner`: The deck with 2 wins (the match winner in best-of-3)
- `winner_names` / `loser_names`: For constructing training examples

### TrainingExample

A single training sample derived from one MatchOutcome.

| Field | Type | Description |
|-------|------|-------------|
| winner_cards | tensor (N, 544) | Feature vectors for non-basic-land cards of the winning deck |
| loser_cards | tensor (M, 544) | Feature vectors for non-basic-land cards of the losing deck |

**Batching**: Winner and loser decks are batched separately. Within each batch, shorter decks are padded to the longest deck in that batch. A boolean mask of shape `(batch, max_cards)` marks real cards as True.

### SetTransformerScorer (nn.Module)

The core model.

| Component | Shape / Details |
|-----------|-----------------|
| Self-attention layers | 2-4 SAB blocks, each with `nn.MultiheadAttention(d_model=544, nhead=4-8)` + feedforward + LayerNorm |
| PMA pooling | 4-8 learned seed vectors of dim 544, cross-attend over card representations |
| Scoring MLP | Linear(seeds * d_model, 256-512) → ReLU → Linear(_, 256-512) → ReLU → Linear(_, 1) |
| feat_mean | register_buffer, shape (32,) — per-feature mean for indices 512-543 |
| feat_std | register_buffer, shape (32,) — per-feature std for indices 512-543 |

**Normalization**: At forward time, indices 512-543 of input are normalized as `(x - feat_mean) / feat_std`. Indices 0-511 pass through unchanged.

**Permutation invariance**: Guaranteed by (1) no positional encodings, (2) self-attention is equivariant, (3) PMA pooling is invariant.

### ScorerCheckpoint

Saved model state.

| Field | Type | Storage |
|-------|------|---------|
| model_state_dict | dict | All model parameters + registered buffers (includes feat_mean, feat_std) |
| optimizer_state_dict | dict | Optimizer state for resuming training |
| epoch | int | Current training epoch |
| best_val_loss | float | Best validation loss seen so far |
| config | dict | Model hyperparameters (d_model, n_heads, n_layers, n_seeds, etc.) |

**Files**: Two checkpoint files maintained:
- `latest.pt` — overwritten after each validation evaluation
- `best.pt` — overwritten only when validation loss improves

### EvaluationResult

Aggregate result of the Forge baseline evaluation.

| Field | Type | Description |
|-------|------|-------------|
| pools_evaluated | int | Number of pools that completed evaluation |
| total_games | int | Total individual games played across all matches |
| wins_scorer | int | Games won by scorer-built decks |
| wins_forge | int | Games won by Forge-built decks |
| win_rate | float | wins_scorer / total_games |

## Relationships

```text
MatchOutcome ──reads──→ match-outcomes.txt (feature 012 output)
    │
    ├──lookup──→ CardFeatureVector (one per card name)
    │                │
    │                └── DeterministicFeatures (indices 512-543)
    │
    └──produces──→ TrainingExample (winner_cards, loser_cards)
                       │
                       └──trains──→ SetTransformerScorer
                                        │
                                        ├──saves──→ ScorerCheckpoint (latest.pt, best.pt)
                                        │
                                        └──scores──→ EvaluationResult (vs Forge baseline)
```

## State Transitions

### Training Pipeline

```text
[Raw card text files] 
    → encode-cards → [544-dim .npz files]

[match-outcomes.txt] + [.npz files]
    → train-scorer (Phase A: frozen embeddings) → [latest.pt, best.pt]
    → train-scorer (Phase B: unfrozen embeddings, from Phase A checkpoint) → [latest.pt, best.pt]
```

### Evaluation Pipeline

```text
[best.pt checkpoint]
    → evaluate-scorer (Python: generate pools + build deck A via greedy search)
    → [validation-matches-*.txt files] (split across workers)
    → Java workers (build deck B + play match)
    → [*-outcomes.txt files]
    → Python collects → [EvaluationResult printed to console]
```
