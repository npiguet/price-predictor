# Feature Specification: Validate Card Embeddings

**Feature Branch**: `014-validate-embeddings`  
**Created**: 2026-04-02  
**Status**: Draft  
**Input**: User description: "Validate embeddings before starting the sealed deck picker training. This feature is described in details in specs/sealed-deck-picker.md in the Training Curriculum -> Stage 0 -> Step 1 section. This feature only covers the embedding validation, and not the embedding generation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Embedding Validation (Priority: P1)

A researcher has generated card embeddings (`.npz` files) and wants to verify they encode the features that Stage 2 training depends on before investing hours in training. They run the `validate-embeddings` command, which trains lightweight linear probes on top of the frozen embeddings and reports pass/fail for each feature. If all probes pass, the researcher proceeds to training with confidence. If any fail, they know the embeddings are insufficient and can retrain the encoder before wasting time on training that cannot converge.

**Why this priority**: This is the entire feature. Without this validation, a researcher has no way to know whether Stage 2 training will work until they've already invested significant compute in training and observed the model failing to learn.

**Independent Test**: Run `python -m sealed validate-embeddings --cards-path output/cardsfolder/` against a set of pre-generated embeddings and verify the output reports per-feature scores and an overall pass/fail result.

**Acceptance Scenarios**:

1. **Given** a cards-path directory containing `.npz` embedding files and `.txt` card text files for all cards, **When** the researcher runs `python -m sealed validate-embeddings --cards-path output/cardsfolder/`, **Then** the command trains a linear probe for each feature in the validation table (see Requirements), prints each feature's achieved score and pass threshold, reports an overall PASS or FAIL result, and exits with code 0 (pass) or 1 (fail).

2. **Given** embeddings that encode all required features above threshold, **When** the researcher runs `validate-embeddings`, **Then** every probe passes, the overall result is PASS, and the exit code is 0.

3. **Given** embeddings where one or more features score below threshold (e.g. a random encoder that doesn't capture card semantics), **When** the researcher runs `validate-embeddings`, **Then** the failing features are clearly identified with their achieved score vs required threshold, the overall result is FAIL, and the exit code is 1.

4. **Given** a cards-path with configurable threshold overrides, **When** the researcher runs `validate-embeddings --threshold-accuracy 0.99 --threshold-r2 0.90`, **Then** the overridden thresholds are used instead of the defaults.

---

### Edge Cases

- What happens when `.txt` card text files are missing for some cards? Those cards are excluded from probing. If fewer than 50 cards remain, the command fails with an error explaining that insufficient data is available for meaningful probing.
- What happens when `.npz` embedding files are missing for some cards? Same as above — cards without embeddings are excluded.
- What happens when the cards-path directory does not exist? The command fails with an informative error and exit code 2.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a `validate-embeddings` subcommand under `python -m sealed` that accepts `--cards-path`, `--threshold-accuracy`, and `--threshold-r2` parameters.

- **FR-002**: The system MUST train a separate linear probe for each feature defined in the validation table below, using card embeddings as input and ground truth extracted from card text files:

| Feature                                     | Probe type                         | Ground truth source                   | Pass threshold            |
|---------------------------------------------|------------------------------------|---------------------------------------|---------------------------|
| Is land                                     | Logistic regression (binary)       | Card types line                       | Accuracy ≥ 0.99           |
| Card color (per W/U/B/R/G/C)                | Logistic regression, one per color | `mana cost:` pip counts > 0           | Accuracy ≥ 0.95 per color |
| Pip counts (per W/U/B/R/G/C)                | Linear regression, one per color   | `mana cost:` parsed pip counts        | R² ≥ 0.85 per color       |
| Mana value                                  | Linear regression                  | Sum of generic + colored pips (X = 0) | R² ≥ 0.90                 |
| Mana produced (per W/U/B/R/G/C, lands only) | Logistic regression, one per color | `activated[N]: {T}: add` abilities    | Accuracy ≥ 0.95 per color |

- **FR-003**: Ground truth extraction MUST reuse the same parsing logic as the Stage 2 mana scorer — specifically `count_pips()` for pip counts and card color, `count_actual_sources()` for mana production, and the card type-line check from `EmbeddingAdapter.is_land()`.

- **FR-004**: Probes MUST be evaluated using cross-validation or a held-out test split (not scored on training data) to ensure reported metrics reflect generalization, not memorization.

- **FR-005**: The command MUST print each probe's feature name, achieved score, and pass threshold. Both passing and failing probes MUST be reported.

- **FR-006**: The command MUST exit with code 0 if all probes pass, code 1 if any probe fails, and code 2 for input errors (missing directory, insufficient data).

- **FR-007**: The `--threshold-accuracy` parameter MUST override the default accuracy threshold for all classification probes. The `--threshold-r2` parameter MUST override the default R² threshold for all regression probes. The mana value probe MUST use a threshold of `max(threshold-r2, 0.90)`. The is-land probe MUST use a threshold of `max(threshold-accuracy, 0.99)`.

- **FR-008**: The command MUST report the total number of cards used for probing in its output.

### Key Entities

- **Card Embedding**: A 512-dimensional vector (`.npz` file) produced by the pretrained card encoder for a single card. Input to the probe models.
- **Card Text**: The structured Oracle text (`.txt` file) for a single card, used to extract ground truth labels. Contains `name:`, `types:`, `mana cost:`, and `activated[N]:` lines.
- **Linear Probe**: A lightweight linear classifier (logistic regression) or regressor (linear regression) trained on top of frozen embeddings to test whether a specific feature is decodable from the embedding.
- **Probe Result**: The score achieved by a probe (accuracy for classification, R² for regression) along with its pass/fail status against the threshold.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: When run against the current production card encoder embeddings, all probes pass at their default thresholds (confirming the encoder captures the required features).
- **SC-002**: When run against randomized embeddings (e.g. shuffled or noise-replaced), at least 3 of the 5 probe categories fail (confirming the validation is meaningful and not a rubber stamp).
- **SC-003**: The output provides enough information for a researcher to identify which specific features are missing from the embeddings and take corrective action (retrain the encoder, adjust card text format, etc.).

## Assumptions

- Card text files (`.txt`) and embedding files (`.npz`) are colocated in the same `--cards-path` directory and share the same base filename (e.g. `Lightning-Bolt.npz` and `Lightning-Bolt.txt`).
- The card text format follows the production format documented in spec 006-card-script-parsing (lowercase, `mana cost:`, `types:`, `activated[N]:` lines).
- The mana scorer parsing functions (`count_pips`, `count_actual_sources`) from `sealed.domain.mana_scorer` are available and stable.
- The `EmbeddingAdapter.is_land()` check (or equivalent type-line parsing) is available for ground truth extraction.
- Basic lands (Plains, Island, Swamp, Mountain, Forest, Wastes) always have embeddings and text files present in `--cards-path`.
