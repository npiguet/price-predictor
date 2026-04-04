# Feature Specification: Embeddings Retraining with Auxiliary Supervision

**Feature Branch**: `015-embeddings-retraining`  
**Created**: 2026-04-03  
**Status**: Draft  
**Input**: User description: "Retrain embeddings with auxiliary heads for mana-relevant features, based on description.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Encoder with Mana-Aware Embeddings (Priority: P1)

As a developer, I want to retrain the price-predictor transformer from scratch with auxiliary
supervision on mana-relevant features, so that the resulting embeddings encode card color, pip counts,
mana value, and mana production — enabling Stage 2 sealed training to reason about casting costs and
mana base alignment.

**Why this priority**: This is the core deliverable. Without mana-aware embeddings, Stage 2 sealed
training cannot function because the model cannot reason about casting costs or mana base alignment.
Feature 014 probes confirmed the current embeddings fail on color and cost features.

**Independent Test**: Run training with auxiliary heads, then run `validate-embeddings` against the
new encoder. All 20 probes must pass.

**Acceptance Scenarios**:

1. **Given** a card corpus with text files and price data, **When** I run training with auxiliary
   supervision enabled, **Then** training completes and produces a checkpoint in the standard format.
2. **Given** a trained encoder with auxiliary supervision, **When** I run `validate-embeddings`,
   **Then** all 20 probes pass their thresholds (is-land >= 0.990, card color >= 0.950,
   pip counts exact match >= 0.900, mana value exact match >= 0.750, mana produced >= 0.950).
3. **Given** a trained encoder with auxiliary supervision, **When** I compare price prediction
   accuracy to the feature 007 baseline, **Then** accuracy has not degraded unacceptably (manual
   judgment).

---

### User Story 2 - Correct Card Color Label Definition (Priority: P1)

As a developer, I want card color labels to follow MTG rules correctly, so that the probes and
auxiliary heads train on accurate ground truth.

**Why this priority**: Incorrect labels would make the auxiliary supervision teach wrong features,
undermining the entire retraining. This also retroactively corrects features 013 and 014.

**Independent Test**: Verify label generation against known cards: a devoid card with colored pips
gets W/U/B/R/G = 0 and C = 1; a card with only generic mana cost gets C = 1; a land with no mana
cost gets C = 1.

**Acceptance Scenarios**:

1. **Given** a card with mana cost `{2}{R}`, **When** color labels are computed, **Then** R = 1,
   W = U = B = G = C = 0.
2. **Given** a card with mana cost `{C}{C}`, **When** color labels are computed, **Then** C = 1,
   W = U = B = R = G = 0.
3. **Given** a card with mana cost `{3}` (generic only), **When** color labels are computed,
   **Then** C = 1, W = U = B = R = G = 0.
4. **Given** a card with no mana cost (e.g., a land), **When** color labels are computed, **Then**
   C = 1, W = U = B = R = G = 0.
5. **Given** a card with mana cost `{2}{R}` and the "devoid" keyword, **When** color labels are
   computed, **Then** C = 1, W = U = B = R = G = 0 (devoid overrides colored pips).
6. **Given** a card with mana cost `{G/R}` (hybrid), **When** color labels are computed, **Then**
   R = 1, G = 1, W = U = B = C = 0.

---

### User Story 3 - Tune Auxiliary Loss Weight (Priority: P2)

As a developer, I want to control the auxiliary loss weight (lambda) so that I can find the balance
between learning mana features and preserving price prediction accuracy.

**Why this priority**: The right lambda value is essential for the retraining to succeed — too low
and probes fail, too high and price accuracy degrades. But the training infrastructure (P1) must
exist before tuning can begin.

**Independent Test**: Train with different lambda values and compare probe scores and price validation
loss across runs.

**Acceptance Scenarios**:

1. **Given** a lambda value that is too low, **When** training completes and probes are run,
   **Then** some probes fail their thresholds, indicating the auxiliary signal was insufficient.
2. **Given** a lambda value that is too high, **When** training completes, **Then** price prediction
   accuracy degrades noticeably relative to the feature 007 baseline.
3. **Given** an appropriate lambda value (~0.2 as starting point), **When** training completes,
   **Then** all 20 probes pass and price accuracy is acceptable.

---

### Edge Cases

- What happens when a card has no mana cost line at all (e.g., lands)? It is colorless (C = 1),
  pip counts are all 0, and mana value is 0.
- What happens with hybrid mana pips like `{G/R}`? Each color gets 0.5 pips. The card has both
  colors (G = 1, R = 1).
- What happens with Phyrexian mana like `{W/P}`? The color gets 0.5 pips. The card has that color
  (W = 1).
- What happens with `{X}` in mana costs? X contributes 0 to mana value and does not count as any
  color.
- What happens with devoid cards? All W/U/B/R/G color labels are 0, C = 1, regardless of actual
  pips. Pip counts still reflect the actual mana cost.
- What happens when a classification head has extreme class imbalance (e.g., "is land" at ~5%
  positive)? The loss function uses pos_weight to prevent trivial all-zero predictions.
- What happens when ordinal targets span very different ranges (pip count 0–3 vs mana value 0–10)?
  Each head has its own class set; EMD loss operates on normalized CDF values so scale differences
  do not affect gradient magnitude.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST add 20 auxiliary linear heads to the encoder's pooled output during
  training: 1 is-land, 6 card-color, 6 pip-count, 1 mana-value, 6 mana-produced.
- **FR-002**: System MUST compute auxiliary labels from card text files at training time using
  existing parsers in `sealed.domain.mana_scorer`.
- **FR-003**: System MUST compute card color labels following MTG rules: a card has color W/U/B/R/G
  if it has >= 1 pip of that color; a card is colorless (C = 1) if its mana cost has no colored pips;
  cards with no mana cost are colorless; devoid cards are colorless regardless of pips.
- **FR-004**: System MUST combine the price loss and auxiliary losses as
  `L_total = L_price + lambda * sum(L_aux_i)`, where lambda is configurable.
- **FR-005**: System MUST use `BCEWithLogitsLoss` with `pos_weight` (negatives/positives) for
  classification heads to handle class imbalance.
- **FR-006**: Pip count heads (indices 7–12) and the mana value head (index 13) MUST use ordinal
  classification with Earth Mover's Distance (EMD) loss rather than MSE regression. Each such head
  outputs K logits (one per class); the loss is the L1 distance between the cumulative distribution
  of the softmax output and the cumulative distribution of the one-hot true label:
  `EMD = sum_{k=1}^{K-1} |CDF_pred(k) - CDF_true(k)|`. This penalises predictions proportionally
  to how many class boundaries they are from the truth.
- **FR-012**: Pip count classes for W, U, B, R, G MUST be:
  `{0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8}` (11 classes). Derived from a full scan of the card
  corpus — every pip count value that appears in a `mana cost:` line is represented exactly. The
  highest observed value is 8 ({G}×8 on Khalni Hydra). No overflow class is used.
- **FR-013**: Pip count classes for C MUST be: `{0, 1, 2, 2.5, 3}` (5 classes).
  Derived from corpus scan; 2.5 appears on cards with hybrid {C/W} etc. pips; 3.0 is the max
  (Echoes of Eternity, Rise of the Eldrazi). No overflow class is used.
- **FR-014**: Mana value classes MUST be:
  `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}` (17 classes). Derived from
  corpus scan — every integer mana value from 0 to 16 appears with no gaps. No overflow class
  is used.
- **FR-007**: System MUST discard the 20 auxiliary heads when saving the checkpoint. The saved
  checkpoint MUST contain the encoder and price head in the same format as the current `latest.pt`.
- **FR-008**: System MUST train from scratch (not fine-tune an existing checkpoint).
- **FR-009**: The card color label definition in FR-003 MUST be applied retroactively to any shared
  code used by features 013 and 014, as a correction toward intended behavior.
- **FR-010**: Tests that need card text as input MUST use real card files copied from
  `output/cardsfolder/` into the test fixtures folder, to guarantee the fixture format matches what
  the system will encounter at runtime.
- **FR-011**: System MUST print clear progress messages to the console for pre-training phases that
  are expected to take more than a couple of seconds (e.g., loading ~30k card files, computing
  labels, computing class weights and target statistics).
- **FR-015**: System MUST append 15 explicit mana features to the meta vector passed to the price
  prediction regression head. The meta vector MUST be 30-dimensional: 15 printing features followed
  by 15 mana features (pip counts W/U/B/R/G/C, generic count, X count, mana value, mana produced
  W/U/B/R/G/C), all normalized to approximately [0, 1].
- **FR-016**: System MUST append the same 15 normalized mana features to the card embedding stored
  in `.npz` files, after the 2×d_model transformer embedding. Total embedding dimension is
  `2*d_model + 15`. These explicit features are NOT passed through the Pool Transformer in the
  sealed module.
- **FR-017**: `encode_mana_features()` in `metadata_encoder.py` and `extract_mana_features()` /
  `normalize_mana_features()` in `mana_scorer.py` MUST use identical normalization constants so that
  the meta vector (training) and card embedding (inference) contain the same values.
- **FR-018**: The `validate-embeddings` command MUST accept `--embed-dim` to restrict probes to the
  transformer portion of the embedding, excluding the appended mana features.

### Key Entities

- **Auxiliary Head**: A single linear layer (no activation) projecting the pooled embedding to
  logits. Exists only during training. 13 are binary classifiers (1 logit each); 7 are ordinal
  classifiers (K logits each, where K = 8 for pip counts W/U/B/R/G, 4 for pip count C,
  11 for mana value).
- **Card Color Labels**: A 6-element binary vector (W/U/B/R/G/C) derived from pip counts plus
  devoid/no-mana-cost rules. Distinct from raw pip counts.
- **Lambda (auxiliary weight)**: A scalar hyperparameter controlling the strength of auxiliary
  supervision relative to the price loss. Starting point ~0.2.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 20 embedding probes pass their thresholds when run against the retrained encoder.
  Thresholds: is-land ≥ 0.990 (accuracy), card color ≥ 0.950 (accuracy), pip counts ≥ 0.900
  (exact match after rounding to nearest 0.5), mana value ≥ 0.750 (exact match after rounding to
  nearest integer), mana produced ≥ 0.950 (accuracy).
- **SC-002**: Price prediction accuracy on the validation set does not degrade unacceptably relative
  to the feature 007 baseline (manual judgment after exploring lambda values).
- **SC-003**: The saved checkpoint produces card embeddings of dimension `2*d_model + 15`, where
  the first `2*d_model` elements are the transformer pooled output and the last 15 are normalized
  mana features. The transformer portion is tested by probes with `--embed-dim <2*d_model>`.
- **SC-004**: Card color labels are correct for edge cases: devoid cards are colorless, no-mana-cost
  cards are colorless, hybrid pips grant both colors.
