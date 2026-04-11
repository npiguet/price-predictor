# Feature Specification: Sealed Deck Scorer

**Feature Branch**: `013-sealed-deck-scorer`
**Created**: 2026-04-11
**Status**: Draft
**Parent Spec**: [`specs/sealed-deck-picker.md`](../sealed-deck-picker.md) -- this feature implements Phase 1 (Deck Scorer) of the sealed deck picker architecture
**Input**: User description: "The feature is described in sealed-deck-picker.md in the Phase 1 - Deck Scorer chapter."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extend Card Encoding with Deterministic Features (Priority: P1)

A researcher runs the `encode-cards` command to produce card feature vectors that include both the pretrained text embedding and 32 deterministic game attributes parsed from the card's structured text. Each card's output is a 544-dimensional vector: the first 512 dimensions are the existing text embedding, and dimensions 512-543 contain deterministic features (land status, mana cost breakdown, card color, mana production, power/toughness, loyalty, and reserved padding). The full specification of each deterministic feature — indices, encoding rules, and edge-case handling — is defined in the parent spec under [Deterministic Feature Encoding (indices 512-543)](../sealed-deck-picker.md#deterministic-feature-encoding-indices-512543). Cards that already have a 544-dimensional embedding file are skipped; cards with only a 512-dimensional file are re-encoded to include the new features.

**Why this priority**: The scorer requires the full 544-dimensional feature vector. Without the deterministic features, training cannot begin -- the model needs both the learned text representation and the explicit game attributes to evaluate deck quality.

**Independent Test**: Can be tested by running `encode-cards` on a small cards folder and verifying each output file contains a 544-element vector, with the deterministic features matching expected values from manual inspection of the card text.

**Acceptance Scenarios**:

1. **Given** a cards folder with no existing embedding files, **When** the researcher runs `encode-cards`, **Then** each card's embedding file contains a 544-dimensional vector where dimensions 0-511 are the text embedding and dimensions 512-543 are the deterministic features.
2. **Given** a cards folder with existing 512-dimensional embedding files (from feature 011), **When** the researcher runs `encode-cards --clean`, **Then** all existing embedding files are deleted and every card is re-encoded to the full 544-dimensional format.
3. **Given** a cards folder where all cards already have 544-dimensional embedding files, **When** the researcher runs `encode-cards` (without `--clean`), **Then** no files are modified and the command completes quickly.
4. **Given** a card with mana cost `{2}{R}{R}`, **When** its embedding is generated, **Then** the deterministic features reflect: red pips = 2, generic mana = 2, mana value = 4, is_red = 1, and other color flags = 0.
5. **Given** a land card with an activated ability that adds `{W} or {U}`, **When** its embedding is generated, **Then** is_land = 1, produces_W = 1, produces_U = 1, mana_count = 1.
6. **Given** a card with the `devoid` keyword and `{1}{B}{G}` mana cost, **When** its embedding is generated, **Then** is_colorless = 1 and all WUBRG color flags = 0, despite having black and green pips.

---

### User Story 2 - Train Deck Scorer on Match Outcomes (Priority: P2)

A researcher runs a training command that reads match outcome data (deck pairs and game results from feature 012) and card embeddings (from P1 above), and trains a model that assigns a quality score to any given deck. The model learns from pairwise comparisons: for each recorded match, it scores both decks and adjusts so that the winning deck scores higher. Training produces a saved model checkpoint that can be used for deck evaluation and later for search-based deck building (Phase 2). The full model architecture, training objective, and normalization pipeline are defined in the parent spec under [Architecture](../sealed-deck-picker.md#architecture), [Training Objective — Bradley-Terry Pairwise Loss](../sealed-deck-picker.md#training-objective--bradley-terry-pairwise-loss), and [Feature Normalization](../sealed-deck-picker.md#feature-normalization).

**Why this priority**: The deck scorer is the core deliverable of Phase 1 -- it is the model that makes all downstream deck building possible. Without a trained scorer, there is no way to evaluate or compare decks programmatically.

**Independent Test**: Can be tested by running training on a small subset of match outcomes and verifying that training loss decreases over time, the saved checkpoint can be loaded, and the model can score an arbitrary deck.

**Acceptance Scenarios**:

1. **Given** a match outcomes file with at least 1,000 recorded matches and a cards folder with 544-dimensional embeddings, **When** the researcher starts training, **Then** training proceeds and reports loss at regular intervals.
2. **Given** training in progress, **When** training completes or is stopped, **Then** two model checkpoints are saved: a latest checkpoint (current state) and a best checkpoint (lowest validation loss seen during training). Both include all model parameters and normalization statistics, and can be loaded for scoring or resumed training.
3. **Given** a trained model checkpoint and a deck (list of card names), **When** the researcher requests a score, **Then** the model returns a single numerical score for that deck.
4. **Given** a pair of decks where one is clearly stronger (e.g., all high-impact cards vs. all filler), **When** both are scored by a sufficiently trained model, **Then** the stronger deck receives a higher score.
5. **Given** training data, **When** training begins, **Then** the deterministic features (indices 512-543) are normalized to zero mean and unit variance using statistics computed across the full training corpus, and these statistics are stored as part of the model checkpoint.
6. **Given** a match outcome, **When** the model processes both decks, **Then** both decks are scored by the same shared model (same weights), not by separate models.

---

### User Story 3 - Track Validation Metrics During Training (Priority: P3)

During training, the system automatically evaluates on a held-out set of match outcomes and reports validation metrics: validation loss and prediction accuracy (fraction of held-out matchups where the higher-scored deck actually won). This allows the researcher to detect overfitting and assess training progress.

**Why this priority**: Without validation metrics, the researcher has no way to know when training is sufficient or when the model starts overfitting. This is essential for producing a useful model.

**Independent Test**: Can be tested by running training with a validation split and verifying that validation loss and accuracy are reported periodically, and that the divergence between training and validation loss is visible when overfitting occurs.

**Acceptance Scenarios**:

1. **Given** training data, **When** training begins, **Then** a portion of the data is held out for validation (split by match, so that all games from a given match are in the same split, preventing data leakage).
2. **Given** training in progress, **When** a validation evaluation runs, **Then** the system reports both validation loss and prediction accuracy.
3. **Given** a model that has overfit, **When** validation metrics are inspected, **Then** training loss continues decreasing while validation loss has increased -- the divergence is visible in the reported metrics.

---

### User Story 4 - Evaluate Scorer Against Forge Baseline (Priority: P4)

The researcher runs an evaluation command that tests whether the scorer can build better decks than Forge's built-in deck builder. The evaluation uses a round-robin design: N pools are generated (configurable, default 12), and both builders construct a deck from each pool. Every scorer-built deck (A) plays every Forge-built deck (B) in a best-of-K match (configurable K, default 3), producing N² matches. Per-pool win rate comparisons and aggregate win rates are reported. Both builders receive the same pools, and all decks face a shared opponent field, isolating builder quality from pool quality. The full evaluation workflow is defined in the parent spec under [Evaluation Against External Baseline](../sealed-deck-picker.md#evaluation-against-external-baseline), and the greedy search procedure is described under [Phase 2 — Search-Based Deck Builder](../sealed-deck-picker.md#phase-2--search-based-deck-builder-inference).

**Why this priority**: Validation metrics (P3) measure whether the model fits its training distribution, but the Forge baseline evaluation measures whether the scorer actually produces better decks in practice. This is the ultimate quality signal -- but it requires the scorer to be trained first (P2) and meaningful to evaluate (P3).

**Independent Test**: Can be tested by running the evaluation with a trained checkpoint and verifying it produces per-pool and aggregate win rate comparisons and completes within a reasonable time.

**Acceptance Scenarios**:

1. **Given** a trained model checkpoint and a requested number of evaluation pools N, **When** the researcher runs evaluation, **Then** the Python script invokes a Java process to generate N fresh pools, builds one deck A_i per pool using the scorer-guided greedy search, and invokes a Java command-line tool to build one deck B_i per pool using Forge's SealedDeckBuilder from the same pool (returned via stdout).
2. **Given** N A-decks and N B-decks, **When** evaluation proceeds, **Then** the system writes N² validation match lines (every A deck paired with every B deck) split into per-worker files upfront, one file per Java worker.
3. **Given** a subset of matches, **When** a Java worker processes them, **Then** for each match the worker plays a best-of-K match between the two pre-built decks via the Forge AI and appends the result as `{wins_A};{wins_B}` to an outcomes file (`{input_file}-outcomes.txt`).
4. **Given** all workers have completed, **When** the Python script collects results, **Then** it reads all outcome files and prints a summary to the console including: per-deck win rates, per-pool comparison (A_i win rate vs B_i win rate), and aggregate win rate for each builder group.
5. **Given** a scorer-guided deck, **When** the deck is constructed, **Then** the greedy search starts from a random selection of 23 non-land cards from the pool, iteratively tries swapping each non-land card in the deck with each card remaining in the pool, picks the swap that most improves the score, recomputes basic lands after each swap, and repeats until no swap improves the score.
6. **Given** a Java worker that crashes mid-evaluation with some matches already played, **When** the Python script retries that worker's subset, **Then** the restarted worker skips matches whose outcomes are already in the outcomes file and only plays the remaining matches.

---

### User Story 5 - Resume Training with Embedding Fine-Tuning (Priority: P5)

After initial training with fixed card representations (Phase A), the researcher resumes training with card embeddings unfrozen at a reduced learning rate. This allows the model to refine what the card representations encode -- shifting them from generic text similarity toward deckbuilding-relevant complementarity (e.g., cards that work well together becoming nearby in embedding space, even if their text is very different). The researcher can monitor how much the embeddings drift from their original values to ensure fine-tuning is stable. The full embedding schedule rationale and drift monitoring guidance are defined in the parent spec under [Embedding Schedule](../sealed-deck-picker.md#embedding-schedule).

**Why this priority**: Embedding fine-tuning is an optimization step. The scorer must already work with frozen embeddings (P2) before this adds value. However, fine-tuning can meaningfully improve model quality by letting the representations reorganize around what the scorer actually needs.

**Independent Test**: Can be tested by resuming from a Phase A checkpoint with embeddings unfrozen, verifying that training continues, and confirming that embedding drift metrics are reported.

**Acceptance Scenarios**:

1. **Given** a trained Phase A checkpoint (frozen embeddings), **When** the researcher resumes training with embeddings unfrozen, **Then** training continues from the existing checkpoint with the scoring model's learned weights preserved.
2. **Given** training with unfrozen embeddings, **When** training progresses, **Then** the system reports average embedding drift (mean L2 distance from initial embedding values) so the researcher can monitor stability.
3. **Given** a desired embedding learning rate, **When** the researcher configures training, **Then** the embedding learning rate can be set independently from the rest of the model's learning rate (typically 10-100x lower).

---

### Edge Cases

- If a card name from the match outcomes file has no corresponding embedding file, the system logs a warning identifying the missing card(s) to stderr and skips matches containing those cards, reporting a summary of skipped matches at the end of data loading.
- If the match outcomes file is empty or missing, the system reports a clear error before attempting to train.
- Variable-length decks (different numbers of non-basic lands per deck) are handled by the scorer -- shorter decks within a batch are padded, and padding does not influence the score.
- Stale 512-dimensional embedding files from feature 011 are not auto-detected; the researcher must use `encode-cards --clean` to delete and re-encode all cards in the new 544-dimensional format (see P1, acceptance scenario 2).
- If a card has no mana cost (lands, some special cards), all mana cost features are zero.
- If a card has the `devoid` keyword, it is classified as colorless regardless of its mana cost pips.
- Power/toughness values of `*` or `X` are encoded as zero.
- Non-creature cards have power = 0 and toughness = 0; non-planeswalker cards have loyalty = 0.
- Mana production is only parsed from activated abilities containing `add` patterns; triggered or static mana production is ignored (an accepted simplification -- the text embedding already captures the full card text).

## Clarifications

### Session 2026-04-11

- Q: What algorithm produces the heuristic initial deck for the scorer-guided greedy search (evaluation P4)? → A: Random selection of 23 non-land cards from the pool.
- Q: What fraction of match outcomes data is held out for validation? → A: 80/20 split (80% training, 20% validation).
- Q: How are match outcomes mapped to training examples? → A: One training example per match line; the match winner (the deck with 2 game wins in the best-of-3) is the label. No expansion to per-game examples.
- Q: Should training automatically preserve the best checkpoint (lowest validation loss) in addition to the final checkpoint? → A: Yes — save both best-validation-loss checkpoint and latest checkpoint (two files).
- Q: Does the evaluation pipeline need worker crash recovery (monitor + restart), or simpler coordination? → A: Simple run-and-wait with retry — if a worker crashes, retry its subset. Workers skip matches whose outcomes are already recorded in the outcomes file (no replaying completed matches).

## Requirements *(mandatory)*

### Functional Requirements

**Deterministic Feature Encoding (Prerequisite Delta)**

- **FR-001**: The `encode-cards` command MUST produce 544-dimensional feature vectors per card: the first 512 dimensions from the pretrained text encoder, and dimensions 512-543 from deterministic features parsed from the card's structured text.
- **FR-002**: The 32 deterministic features MUST include, in order: is_land (1), mana cost breakdown -- per-color pip counts for W/U/B/R/G/C, generic mana, X pip count, and mana value (9), card color flags -- multi-hot W/U/B/R/G/colorless (6), mana produced -- multi-hot color flags W/U/B/R/G/C plus total mana count (7), power and toughness (2), starting loyalty (1), and zero-padding (6) -- totaling 32 features. Exact index assignments, encoding rules, and edge-case handling are defined in the parent spec under [Deterministic Feature Encoding (indices 512-543)](../sealed-deck-picker.md#deterministic-feature-encoding-indices-512543).
- **FR-003**: Cards with the `devoid` keyword MUST be classified as colorless regardless of their mana cost pips (all WUBRG color flags 0, colorless flag 1).
- **FR-004**: The existing `--clean` flag MUST be used to force a full re-encode when upgrading from the previous 512-dimensional format to the new 544-dimensional format. Without `--clean`, cards with any existing embedding file are skipped as before.
- **FR-005**: Embedding files MUST store raw (unnormalized) deterministic feature values. Normalization is performed at training time, not at encoding time, so that new cards can be encoded without recomputing global statistics.

**Deck Scorer Model**

- **FR-006**: The scorer MUST accept a variable-length unordered set of card feature vectors (representing the spells and non-basic lands of a deck) and output a single scalar quality score. See parent spec [Architecture](../sealed-deck-picker.md#architecture) and [Architecture Details](../sealed-deck-picker.md#architecture-details) for the full model design and starting hyperparameters.
- **FR-007**: The scorer MUST be permutation-invariant -- the same set of cards in any order MUST produce the identical score.
- **FR-008**: Basic land cards MUST be excluded from the scorer's input. Only spells and non-basic lands are scored, since basic lands are assigned deterministically from the selected spells and carry no additional information. See parent spec [Scorer Input — Non-Land Cards Only](../sealed-deck-picker.md#scorer-input--non-land-cards-only) for rationale and batching details.
- **FR-009**: When scoring two decks in a training pair, the scorer MUST use the same shared model instance (same weights) for both decks. The model learns a scoring function, not a comparison function.

**Training**

- **FR-010**: Training MUST use pairwise match outcomes as the learning signal. Each line in the match outcomes file is one training example; the match winner (the deck with 2 game wins in the best-of-3) is the label. The model scores both decks and is trained so that the probability of the winning deck being scored higher follows the Bradley-Terry model: `P(A beats B) = sigmoid(score_A - score_B)`. See parent spec [Training Objective — Bradley-Terry Pairwise Loss](../sealed-deck-picker.md#training-objective--bradley-terry-pairwise-loss) for the full loss formulation and rationale.
- **FR-011**: At training startup, the system MUST compute per-feature mean and standard deviation for the deterministic features (indices 512-543) across the full training corpus and normalize those features to zero mean and unit variance in memory. See parent spec [Feature Normalization](../sealed-deck-picker.md#feature-normalization) for the full normalization pipeline and rationale for normalizing at training time rather than encoding time.
- **FR-012**: The normalization statistics (per-feature mean and standard deviation vectors) MUST be stored as part of the model checkpoint so they are available at inference time without recomputation and cannot fall out of sync with the model.
- **FR-013**: Training data MUST be split 80/20 into training and validation sets by match (each line in the match outcomes file stays intact in one split), preventing data leakage between splits.
- **FR-014**: Training MUST report training loss and validation metrics (validation loss, prediction accuracy) at regular intervals throughout training.
- **FR-015**: Training MUST support two embedding modes: frozen (card embeddings have zero learning rate and do not change during training) and unfrozen (card embeddings train with a configurable, independently-set learning rate). See parent spec [Embedding Schedule](../sealed-deck-picker.md#embedding-schedule) for the Phase A/B rationale and learning rate guidance.
- **FR-016**: When training with unfrozen embeddings, the system MUST report average embedding drift (mean L2 distance of embeddings from their initial values) to help the researcher monitor fine-tuning stability.
- **FR-017**: Training MUST save two model checkpoints: (1) the **latest** checkpoint, overwritten after each validation evaluation, and (2) the **best** checkpoint, overwritten only when validation loss improves. Both can be loaded for inference, evaluation, or resumed training.

**Evaluation**

- **FR-018**: The evaluation pipeline MUST use a round-robin design: generate N pools (configurable, default 12), build one scorer deck (A_i) and one Forge deck (B_i) from each pool, then play N² cross-group matches (every A deck vs every B deck) as best-of-K (configurable K, default 3). Pool generation and Forge deck building are performed by invoking Java processes from Python. The N² match pairings are split into per-worker files upfront. If any worker crashes, the Python script retries the failed worker's subset; workers MUST skip matches whose outcomes are already recorded in their outcomes file (no replaying completed matches).
- **FR-018a**: Forge decks (B_i) MUST be built from the same pools as the scorer decks (A_i), using Forge's SealedDeckBuilder invoked via a Java command-line tool that returns the built deck via stdout. Both builders receive identical pools.
- **FR-018b**: Each Java match worker MUST play a best-of-K match between two pre-built decks (both A and B are fully specified 40-card decks) via the Forge AI and write the outcome as `{wins_A};{wins_B}` to a companion outcomes file (`{input_file}-outcomes.txt`). Before playing a match, the worker MUST check the outcomes file and skip any match line whose outcome is already recorded.
- **FR-018c**: After all workers complete, the Python script MUST read all outcome files and print a summary including: per-deck win rates, per-pool comparison (A_i win rate vs B_i win rate), and aggregate win rate for each builder group (all A decks vs all B decks).
- **FR-019**: The scorer-guided greedy search used in evaluation MUST: start from a random selection of 23 non-land cards from the pool, iteratively evaluate all possible single-card swaps (one non-land card out, one pool card in), apply the best-improving swap, recompute basic lands after each swap, and repeat until no swap improves the score. See parent spec [Phase 2 — Search-Based Deck Builder](../sealed-deck-picker.md#phase-2--search-based-deck-builder-inference) for the full search procedure, non-basic land handling, and local optima considerations.
- **FR-020**: Validation prediction accuracy MUST be computed as the fraction of held-out matchups where the deck that scored higher actually won the match.

**CLI**

- **FR-021**: Training MUST be invocable from the command line with configurable parameters including: paths to match outcomes and card embeddings folder, model checkpoint output path, training hyperparameters, and embedding mode (frozen or unfrozen with separate learning rate).
- **FR-022**: Evaluation MUST be invocable from the command line with configurable parameters including: model checkpoint path, number of evaluation pools, games per matchup (best-of-K), and number of Java worker processes.

### Key Entities

- **Card Feature Vector**: A 544-dimensional numerical representation of a single card, combining the 512-dimensional text embedding with 32 deterministic game attributes. Stored as a named file per card in the cards folder.
- **Match Outcome**: A recorded game result consisting of two deck lists (card names) and the win count for each deck in a best-of-3 match. Read from the match outcomes file produced by feature 012.
- **Deck Score**: A single scalar value assigned by the scorer to a complete deck (spells and non-basic lands only). Used to rank decks by quality -- higher score means better deck. Scores are relative (only differences matter), not calibrated win rates.
- **Model Checkpoint**: A saved snapshot of the trained scorer model, including all learned parameters, normalization statistics, and (when unfrozen) the fine-tuned card embeddings. Self-contained for inference.
- **Normalization Statistics**: Per-feature mean and standard deviation vectors for the 32 deterministic features, computed from the training corpus and stored with the model checkpoint.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Running `encode-cards` on a cards folder produces 544-dimensional feature vectors for all cards, with deterministic features matching expected values when spot-checked against the source card text (at least 20 cards verified manually).
- **SC-002**: Training on at least 10,000 match outcomes converges -- training loss decreases during early training and validation loss stabilizes rather than increasing.
- **SC-003**: Validation prediction accuracy exceeds 55%, meaningfully above the 50% random baseline (accounting for inherent noise in individual game outcomes where the weaker deck sometimes wins on draws alone).
- **SC-004**: The scorer assigns higher scores to Forge-built decks than to randomly-assembled decks from the same pool in at least 80% of comparisons, demonstrating it has learned basic deck quality.
- **SC-005**: A trained model checkpoint can be saved, loaded in a fresh session, and used to score decks -- producing identical scores for the same input across sessions.
- **SC-006**: The Forge baseline round-robin evaluation completes and reports per-pool and aggregate win rates. The scorer-built decks achieve a higher aggregate win rate than the Forge-built decks across the shared opponent field, demonstrating competitive deck quality.

## Assumptions

- Features 011 (`encode-cards`, `generate-pools`) and 012 (`match-outcomes`) are complete and functional before this feature begins.
- A sufficient volume of match outcome data (10,000+ matches) has been generated by feature 012 before training begins.
- The pretrained price predictor encoder and vocabulary are available at their default paths.
- Card scripts in the cards folder follow the Forge converted-text format described in spec 006, including parseable `types:`, `mana cost:`, `keyword:`, `activated:`, `power toughness:`, and `loyalty:` lines.
- The Forge connector (from features 002/011/012) is available and provides the deck-building and game-playing infrastructure needed for the Forge baseline evaluation (P4).
- The existing `python -m sealed` CLI entry point will be extended with new subcommands for training and evaluation, alongside the existing `encode-cards`, `generate-pools`, and `match-outcomes` subcommands.
- The `encode-cards` command currently produces 512-dimensional vectors; this feature extends it to 544 dimensions while maintaining the same CLI interface (including the existing `--clean` flag), atomic write behavior, and incremental processing logic.
- The parent spec recommends a Set Transformer architecture for the scorer. Specific architecture choices will be made during the planning phase based on the functional requirements defined here.
- Each line in the match outcomes file represents one match from a unique pair of pools, so splitting by match inherently prevents pool-level data leakage.
