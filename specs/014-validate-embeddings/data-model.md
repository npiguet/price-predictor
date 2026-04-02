# Data Model: Validate Card Embeddings

**Feature**: 014-validate-embeddings | **Date**: 2026-04-02

## Entities

### CardData (Value Object)

Represents a single card's embedding and text, loaded from disk.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Card name (derived from filename stem, e.g. `"lightning_bolt"`) |
| `embedding` | `np.ndarray` | 512-dim float32 vector loaded from `.npz` file |
| `text` | `str` | Full card text loaded from `.txt` file |

**Source**: Paired `.npz` + `.txt` files under `--cards-path`, discovered via `Path.rglob("*.npz")`.

---

### ProbeSpec (Value Object)

Defines a single linear probe to run.

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | `str` | Human-readable name, e.g. `"Card color (W)"`, `"Mana value"` |
| `probe_type` | `Literal["classification", "regression"]` | Determines model and metric |
| `threshold` | `float` | Minimum score to pass (accuracy for classification, R² for regression) |
| `extract_labels` | `Callable[[list[CardData]], np.ndarray]` | Function that extracts ground truth labels from card data |

**Probe inventory** (21 probes total):

| Category | Count | Probe type | Default threshold |
|----------|-------|------------|-------------------|
| Is land | 1 | classification | 0.99 |
| Card color (W/U/B/R/G/C) | 6 | classification | 0.95 |
| Pip counts (W/U/B/R/G/C) | 6 | regression | 0.85 |
| Mana value | 1 | regression | 0.90 |
| Mana produced (W/U/B/R/G/C) | 6 | classification | 0.95 |
| **Total** | **20** | | |

---

### ProbeResult (Value Object)

Result of running a single probe.

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | `str` | Matches `ProbeSpec.feature_name` |
| `score` | `float` | Mean cross-validated score (accuracy or R²) |
| `threshold` | `float` | Required minimum to pass |
| `passed` | `bool` | `score >= threshold` |
| `n_samples` | `int` | Number of cards used for this probe |

---

### ValidationResult (Value Object)

Aggregate result of all probes.

| Field | Type | Description |
|-------|------|-------------|
| `probe_results` | `list[ProbeResult]` | One per probe spec |
| `n_cards` | `int` | Total cards loaded (with both `.npz` and `.txt`) |
| `n_lands` | `int` | Cards identified as lands |
| `all_passed` | `bool` | `True` if every probe passed |

**Exit code mapping**:
- `all_passed == True` → exit code 0
- `all_passed == False` → exit code 1
- Input error (missing directory, <50 cards) → exit code 2

## Relationships

```
cards_path/
  *.npz + *.txt  ──load──►  list[CardData]
                                  │
                                  ▼
                  ProbeSpec[] ──run──►  ProbeResult[]
                                            │
                                            ▼
                                    ValidationResult
                                            │
                                            ▼
                                   CLI output + exit code
```

## Ground Truth Extraction Functions

These functions extract labels from `list[CardData]` for each probe category. They live in `sealed.domain.embedding_probe` and delegate to existing parsers.

| Function | Reused parser | Returns |
|----------|---------------|---------|
| `extract_is_land(cards)` | Type-line check (same logic as `EmbeddingAdapter.is_land()`) | `np.ndarray` of 0/1 |
| `extract_card_color(cards, color)` | `count_pips()` → check if color count > 0 | `np.ndarray` of 0/1 |
| `extract_pip_counts(cards, color)` | `count_pips()` → extract color's count | `np.ndarray` of floats |
| `extract_mana_value(cards)` | New `compute_mana_value()` in `mana_scorer` | `np.ndarray` of floats |
| `extract_mana_produced(cards, color)` | `count_actual_sources()` on card text; any card with `{T}: add` abilities gets 1 (lands, artifacts, creatures) | `np.ndarray` of 0/1 |

## New Domain Function

### `compute_mana_value(cost_str: str) -> float`

**Module**: `sealed.domain.mana_scorer` (alongside existing `_accumulate_pips`)

Parses a brace-format mana cost string and returns total mana value:
- `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}` → +1
- `{N}` (generic digit) → +N
- `{W/P}` (Phyrexian) → +1
- `{G/R}` (hybrid) → +1
- `{X}` → +0

This is the only new function added to `mana_scorer.py`. All other ground truth extraction reuses existing functions directly.

## State Transitions

None — all entities are immutable value objects. The validation is a pure read-only operation.

## Validation Rules

| Rule | Source | Enforcement |
|------|--------|-------------|
| `cards_path` must exist | Edge case spec | CLI returns exit code 2 |
| ≥ 50 paired cards required | Edge case spec | Application layer raises `ValueError` |
