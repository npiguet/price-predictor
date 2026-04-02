# Research: Validate Card Embeddings

**Feature**: 014-validate-embeddings | **Date**: 2026-04-02

## R1: Mana Value Ground Truth Computation

**Question**: How to compute mana value (CMC) from brace-format mana cost strings in `.txt` card files?

**Decision**: Add a `compute_mana_value(cost_str: str) -> float` function in `sealed.domain.mana_scorer` alongside `_accumulate_pips()`.

**Rationale**: The existing `_accumulate_pips()` already parses the brace format (`{2}{W}{W}`) but deliberately ignores generic digits and `{X}`. A dedicated function reuses the same parsing loop but also accumulates generics. This keeps ground truth extraction in the same module as the other mana-parsing functions, satisfying FR-003's requirement to reuse Stage 2 mana scorer logic.

**Alternatives considered**:
- *Reuse `ManaCost.parse()` from `price_predictor.domain.value_objects`*: Rejected — it expects Forge format (`"2 W W"`), not brace format (`{2}{W}{W}`). Converting between formats would add unnecessary coupling between the two modules.
- *Sum `count_pips()` output + separate generic parser*: Rejected — `count_pips()` counts fractional pips for hybrids/phyrexian (0.5 each), but mana value counts each hybrid/phyrexian symbol as 1. Would need a correction step that's more confusing than a clean function.

**Mana value rules** (matching MTG rules for converted mana cost):
| Symbol | Mana value contribution |
|--------|------------------------|
| `{W}`, `{U}`, `{B}`, `{R}`, `{G}` | +1 |
| `{C}` (colorless) | +1 |
| `{N}` (generic digit) | +N |
| `{W/P}` (Phyrexian) | +1 |
| `{G/R}` (hybrid) | +1 |
| `{X}` (variable) | +0 |

## R2: Cross-Validation Strategy for Linear Probes

**Question**: What cross-validation approach for classification and regression probes?

**Decision**: Use `sklearn.model_selection.cross_val_score` with `StratifiedKFold(n_splits=5)` for classification probes and `KFold(n_splits=5, shuffle=True)` for regression probes.

**Rationale**: `cross_val_score` is the simplest sklearn API for evaluating a model with cross-validation — one function call per probe. Stratified splitting for classification preserves class balance (important for rare classes like colorless mana `{C}`). 5-fold is the sklearn default and provides a good bias-variance tradeoff for ~30k samples.

**Alternatives considered**:
- *Train/test split (e.g., 80/20)*: Rejected — higher variance in score estimates, especially for probes with class imbalance. Cross-validation gives more stable estimates.
- *Leave-one-out*: Rejected — computationally expensive for ~30k samples. Overkill for linear models.
- *10-fold*: Rejected — marginal improvement over 5-fold, doubles compute time. Not worth it for a lightweight validation step.

**Scoring metrics**:
- Classification probes: `scoring="accuracy"` (matches spec thresholds)
- Regression probes: `scoring="r2"` (matches spec thresholds)
- Report: mean score across folds

## R3: Mana-Produced Probes — No Land-Only Filtering

**Question**: The "mana produced" probes only have meaningful positive labels for land cards. Should probes filter to lands only?

**Decision**: No filtering. All cards are included in every probe. Ground truth for mana production is extracted by running `count_actual_sources()` on every card's text — any card with `activated[N]: {T}: add` abilities gets non-zero labels, whether it's a land (Forest, Breeding Pool), an artifact (Sol Ring, Orzhov Signet), or a creature (Llanowar Elves). Cards without tap-for-mana abilities naturally get 0.

**Rationale**: Mana production is not exclusive to lands — mana dorks and mana rocks are a significant part of MTG. The `count_actual_sources()` parser already works on any card text regardless of type. No filtering or special-casing needed.

**Alternatives considered**:
- *Filter to lands only*: Rejected — incorrect. Misses mana-producing artifacts and creatures, and adds unnecessary complexity.

## R4: Card File Discovery and Pairing

**Question**: How to discover and pair `.npz` embedding files with `.txt` card text files?

**Decision**: Scan `cards_path` recursively for `.npz` files using `Path.rglob("*.npz")`. For each `.npz` file, look for a matching `.txt` file at the same path with `.txt` suffix. Cards that have both files are included; cards missing either file are excluded (with a count reported). This matches the existing pattern in `cli.py:187` and `encode_cards.py:25`.

**Rationale**: The spec's assumptions section states files are colocated with the same base filename. The existing `EmbeddingAdapter` already uses `card_npz_path().with_suffix(".txt")` to find text files — the validation should work the same way but in reverse (start from `.npz`, find `.txt`).

**Alternatives considered**:
- *Scan `.txt` files and look for `.npz`*: Rejected — `.txt` files exist for all cards but `.npz` only exists after encoding. Starting from `.npz` ensures we only consider cards that have embeddings.
- *Use a card list file*: Rejected — adds an unnecessary input parameter. The cards_path directory is self-describing.

## R5: Output Format

**Question**: What format should the validation output use?

**Decision**: Plain text table printed to stdout, one row per probe, with columns: Feature, Score, Threshold, Status. Summary line at the end with total cards, pass count, fail count, and overall PASS/FAIL.

**Rationale**: Matches the informational style of other sealed CLI commands (encode-cards prints processed/skipped/errors, train prints epoch stats). A structured table is easy to scan and grep. No JSON or file output needed — this is a one-shot validation check.

**Example output**:
```
Validating embeddings (28,451 cards, 10,234 lands)...

Feature                     Score    Threshold  Status
──────────────────────────  ───────  ─────────  ──────
Is land                     0.998    ≥ 0.990    PASS
Card color (W)              0.971    ≥ 0.950    PASS
Card color (U)              0.965    ≥ 0.950    PASS
...
Pip counts (W)              0.891    ≥ 0.850    PASS
...
Mana value                  0.923    ≥ 0.900    PASS
Mana produced (W)           0.982    ≥ 0.950    PASS
...

Result: PASS (21/21 probes passed)
```
