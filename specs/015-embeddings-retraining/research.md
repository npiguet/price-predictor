# Research: Embeddings Retraining with Auxiliary Supervision

**Feature**: 015-embeddings-retraining  
**Date**: 2026-04-03

## Decision 1: Wrapper Model vs. Modifying Base Model

**Decision**: Use a wrapper `AuxiliaryTrainingModel` that composes the existing
`CardPriceTransformerModel` with 20 auxiliary linear heads.

**Rationale**: The auxiliary heads exist only during training and must be discarded
when saving the checkpoint. A wrapper cleanly separates training-only components from
the inference model. Saving becomes trivial: `save_model(wrapper.base, config, path)`.
The base model's public API (`forward()`, `encode()`) remains unchanged.

**Alternatives considered**:
- *Modify `CardPriceTransformerModel` directly*: Rejected because it pollutes the
  inference model with training-only fields and complicates save/load logic (must
  selectively exclude state_dict keys).
- *Duplicate forward pass in wrapper*: Rejected — fragile; changes to the base model
  would silently desync.

**Implementation note**: The base model's `encode()` method has `@torch.no_grad()`,
so it cannot be used during training (gradients would be blocked). The solution is to
extract the shared embedding computation into a private `_embed()` method (with
gradients), then have both `forward()` and `encode()` call it. This is a minimal
refactor — 3 methods replace 2, with zero logic duplication.

```
Before:
  encode()  → [embedding computation, @no_grad]
  forward() → [embedding computation (duplicated)] → price head

After:
  _embed()  → [embedding computation, with gradients]
  encode()  → @no_grad → _embed()
  forward() → _embed() → price head
```

The wrapper then calls `base._embed()` to get the pooled representation for both the
price head and the 20 auxiliary heads.

## Decision 2: Cross-Package Dependency for Label Computation

**Decision**: Training code in `price_predictor.application` imports label extraction
functions from `sealed.domain.embedding_probe`.

**Rationale**: The label extraction is MTG domain logic (card color rules, pip
counting, mana production detection). It belongs in the domain layer. The authoritative
parsers already exist in `sealed.domain.mana_scorer`, and the probe label extractors in
`sealed.domain.embedding_probe` use them. Creating a new function
`compute_aux_labels(card_text: str) -> np.ndarray` in `embedding_probe` that returns
all 20 labels keeps the single source of truth for label definitions.

The dependency direction — `price_predictor.application` → `sealed.domain` — is
acceptable under DDD: application layers may depend on domain layers, even across
packages.

**Alternatives considered**:
- *Duplicate extraction logic in `price_predictor`*: Rejected — violates DRY, risks
  divergence between probe labels and training labels.
- *New shared module `sealed.domain.card_labels`*: Considered, but only 2 concrete
  consumers exist (probes and training). The constitution requires 3 use cases before
  extracting a new abstraction. Adding to `embedding_probe` is simpler.

## Decision 3: Card Color Label Correction (Retroactive Fix)

**Decision**: Redefine the "card color C" label from "has {C} pips in mana cost" to
"card is colorless" (no colored pips, devoid, or no mana cost).

**Rationale**: The current `extract_card_color()` for color='C' checks whether the card
has colorless mana pips ({C}) in its mana cost. This is wrong per MTG rules: colorless
is a property of the card (having no colored identity), not a mana symbol count. A card
with cost `{3}` has no {C} pips but IS colorless. A land with no mana cost IS colorless.

The corrected definition (from spec FR-003):
- W/U/B/R/G = 1 if the card has ≥1 pip of that color in its mana cost
- C = 1 if the card has NO colored pips (W/U/B/R/G all zero), OR the card has devoid,
  OR the card has no mana cost line
- Devoid overrides: W/U/B/R/G = 0, C = 1, regardless of colored pips in cost

**Devoid detection**: In the converted card text format, devoid appears as a static
ability line: `static: devoid`. Detection scans card text lines for this pattern
(case-insensitive). The same detection is needed for both probe labels and training
labels.

**Alternatives considered**:
- *Keep current definition, add separate devoid handling*: Rejected — the fundamental
  definition of "C" was wrong, not just missing devoid support.

## Decision 4: Label Pre-computation Strategy

**Decision**: Pre-compute all 20 auxiliary labels for all cards once before training
begins. Store as a tensor in the dataset alongside existing fields.

**Rationale**: Labels are deterministic and fixed for a given card text. Computing them
per-batch would waste time re-parsing ~30k card texts every epoch. Pre-computation
matches the existing pattern where price targets are computed once in the dataset
constructor.

**Implementation**: Add an optional `aux_labels: torch.Tensor | None` parameter to
`TransformerTrainingDataset.__init__()`. When provided, `__getitem__()` includes an
`"aux_labels"` key in the returned dict. Shape: `(n_cards, 20)`.

**Alternatives considered**:
- *Compute labels on-the-fly in `__getitem__`*: Rejected — repeated string parsing per
  batch is wasteful for zero benefit.

## Decision 5: Regression Target Standardization

**Decision**: Standardize regression targets (pip counts × 6, mana value × 1) using
mean and std computed from the training set only. Apply a minimum std floor of 1.0 to
prevent near-zero division.

**Rationale**: The spec (FR-006) requires standardization to normalize regression losses
to ~1.0 for a naive predictor, matching the scale of BCE losses. Computing statistics
from only the training set prevents data leakage from the validation set.

A std floor of 1.0 (rather than a tiny epsilon) is chosen because:
- If std < 1.0, the target has very little variance (e.g., colorless pip counts are
  ~0 for 99%+ of cards). Standardizing with a tiny std would blow up the few nonzero
  values, making MSE dominated by rare outliers.
- With floor=1.0, the standardized values stay in a reasonable range and the MSE loss
  for near-constant targets stays small, which is appropriate — there's little to learn.

**Alternatives considered**:
- *No standardization*: Rejected — pip counts (~0–5) and mana value (~0–15) would have
  very different MSE scales, causing some heads to dominate gradients.
- *Epsilon floor (1e-6)*: Rejected — allows blow-up for near-constant targets.

## Decision 6: Auxiliary Loss Computation

**Decision**: Use per-head loss functions instantiated once before training, with
pre-computed `pos_weight` for classification heads and pre-standardized regression
targets.

**Rationale**: Each of the 13 classification heads gets its own `BCEWithLogitsLoss`
instance with a per-head `pos_weight = num_negatives / num_positives` (computed from
the training set). The 7 regression heads share a single `MSELoss` applied to
pre-standardized targets.

The combined loss:
```
L_total = L_price + lambda * (sum(L_cls_i) + sum(L_reg_j))
```

Pre-computing pos_weight and standardization stats as a "label setup" step before
training begins keeps the training loop clean. The setup step prints progress messages
per FR-011 since it involves scanning ~30k cards.

**Alternatives considered**:
- *Single shared BCEWithLogitsLoss for all classification heads*: Not possible —
  `pos_weight` differs per head.
- *Per-batch standardization*: Rejected — statistics would vary per batch, introducing
  noise in the loss signal.
