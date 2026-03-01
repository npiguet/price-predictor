# Feature Specification: Card Price Predictor

**Feature Branch**: `001-card-price-predictor`
**Created**: 2026-02-26
**Status**: Draft
**Input**: User description: "Uses AI/machine learning techniques to guess the price of a card based on its characteristics (mana cost, card types, permanent types, oracle text, PT, abilities, etc.). The goal is to take the description of a card as input, and output a decent price estimate."

## Clarifications

### Session 2026-03-01

- Q: Which representative price to use for cards with multiple printings? → A: Cheapest across all printings (including foil and special). Use the overall lowest price regardless of printing type.
- Q: Should rarity be used as a model input feature? → A: No. Rarity is excluded — the model predicts from game characteristics only (what's visible on the card). Rarity is a printing attribute, not a game attribute, and made-up cards have no rarity.
- Q: What currency and price source should the model use? → A: EUR from Cardmarket prices as exposed by MTGJSON files. The model is trained natively on EUR prices. No currency conversion needed. Cardmarket is the authoritative price source as it is less volatile than alternatives. MTGJSON remains the data file source.
- Q: How should the system handle colorless mana ({C}) vs generic mana ({1}–{N})? → A: Colorless mana and generic mana are distinct concepts and must be tracked separately. Generic mana (denoted {1} to {N} in card text) appears only in mana costs and means "any color or colorless mana can pay this." Colorless mana (denoted {C}) is a specific type of mana — it is not a color, but some costs require specifically colorless mana (e.g., mana produced by Sol Ring). Colored mana cannot pay colorless costs. The system must distinguish colorless mana pips from generic mana in mana cost parsing and feature engineering.
- Q: Should the system provide progress logging during long-running operations? → A: Yes. Training and evaluation involve multiple slow stages (parsing 32,000+ card scripts, loading large JSON files, building mappings, training the model). The user must see human-readable progress messages on the console so they know the system is working and can estimate how long it will take.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Predict Price from Card Attributes (Priority: P1)

A user provides the characteristics of a Magic: The Gathering card — such as mana cost, card type(s), permanent type(s), oracle text, power/toughness, and keyword abilities — and receives an EUR market price estimate in return. The card does not need to exist in any real set; predicting the price of hypothetical or custom-designed cards is the primary use case for this project.

This is the core value proposition: given a card description (real or imaginary), produce a meaningful price prediction.

**Why this priority**: This is the fundamental capability the system exists to deliver. Without single-card prediction, nothing else matters.

**Independent Test**: Can be fully tested by submitting a known card's attributes and verifying the system returns a numeric price estimate within a reasonable range of the card's actual market price.

**Acceptance Scenarios**:

1. **Given** a complete set of card attributes (mana cost, at least one card type, oracle text), **When** the user submits the attributes for prediction, **Then** the system returns an EUR price estimate as a positive numeric value.
2. **Given** a card with minimal attributes (e.g., only card type and mana cost, no oracle text), **When** the user submits the partial attributes, **Then** the system still returns a price estimate using the available information.
3. **Given** a set of attributes matching a well-known expensive card (e.g., characteristics similar to a mythic rare staple), **When** the user submits these attributes, **Then** the predicted price is meaningfully higher than the prediction for a typical common card.
4. **Given** a completely made-up card that does not exist in any real MTG set (e.g., a custom-designed card with novel oracle text), **When** the user submits its attributes, **Then** the system still returns a price estimate based on the learned patterns from real cards.

---

### User Story 2 - Train Model on Historical Card Data (Priority: P2)

The system assembles its training dataset by reading card attributes from MTG Forge card scripts (the canonical source of card game data) and pairing them with EUR Cardmarket prices from a frozen MTGJSON AllPricesToday.json snapshot. It then trains or retrains the prediction model so that future predictions reflect real-world pricing patterns.

**Why this priority**: The prediction model must be trained before it can produce accurate estimates. This is a prerequisite for the quality of P1, but is separated because the training workflow is a distinct user journey from prediction.

**Independent Test**: Can be tested by providing a training dataset, triggering the training process, and verifying that a trained model artifact is produced and that predictions change compared to an untrained baseline.

**Acceptance Scenarios**:

1. **Given** Forge card scripts and a frozen price data file are available, **When** the user initiates model training, **Then** the system reads card attributes from Forge scripts, matches them to prices, and produces a trained model.
2. **Given** a previously trained model exists, **When** the user retrains with updated data files, **Then** the new model replaces the old one and subsequent predictions reflect the updated data.
3. **Given** some card scripts lack a matching price entry (or vice versa), **When** the user initiates training, **Then** the system reports which cards were skipped and completes training on the successfully matched data.
4. **Given** some card scripts contain malformed or unparseable data, **When** the user initiates training, **Then** the system reports parsing errors and continues with the valid cards.

---

### User Story 3 - Evaluate Model Accuracy (Priority: P3)

A user wants to understand how accurate the prediction model is. They provide a held-out test set of cards with known prices, and the system reports accuracy metrics — showing how close predictions are to actual prices.

**Why this priority**: Accuracy evaluation is essential for trust and iterative improvement, but the system can deliver value (P1) and be trained (P2) without a formal evaluation workflow.

**Independent Test**: Can be tested by providing a test dataset with known prices, running evaluation, and verifying that accuracy metrics (e.g., mean absolute error, median error percentage) are returned.

**Acceptance Scenarios**:

1. **Given** a trained model and a test dataset of cards with known prices, **When** the user runs model evaluation, **Then** the system returns accuracy metrics including mean absolute error and median percentage error.
2. **Given** a test dataset, **When** the evaluation completes, **Then** the system provides a per-card breakdown showing predicted vs. actual price for each card in the test set.

---

### User Story 4 - See Progress During Long-Running Operations (Priority: P4)

A user initiates model training or evaluation and wants to see that the system is actively working. Long-running operations — parsing thousands of card scripts, loading large JSON files, building name-to-UUID mappings, and training the model — should display human-readable progress messages on the console so the user is never left staring at a blank screen wondering if the system is stuck.

**Why this priority**: Without progress feedback, the user has no way to distinguish "working" from "frozen." This is a usability concern, not a functional one — the system works correctly without it, but the experience is poor. Depends on P2 and P3 operations existing first.

**Independent Test**: Can be tested by running the train or evaluate command and verifying that progress messages appear on the console before the final JSON result is printed.

**Acceptance Scenarios**:

1. **Given** the user initiates model training, **When** each major stage begins (parsing card scripts, loading price data, building name-to-UUID mapping, matching cards to prices, training the model, saving the model), **Then** the system prints a human-readable progress message to the console identifying the current stage.
2. **Given** the user initiates model training, **When** a stage that processes many items is in progress (e.g., parsing 32,000+ card scripts or loading a large JSON file), **Then** the system provides periodic progress updates (e.g., item counts or completion indicators) so the user can estimate how long the operation will take.
3. **Given** the user initiates model evaluation, **When** each major stage begins, **Then** progress messages are displayed in the same manner as training.
4. **Given** the user initiates a single-card prediction (which is fast), **When** prediction completes, **Then** progress messages are not required because the operation is near-instantaneous.
5. **Given** the user runs training or evaluation, **When** progress messages are printed, **Then** the messages appear on stderr (not stdout) so they do not interfere with the structured JSON output on stdout.

---

### Edge Cases

- What happens when the user provides card attributes that don't correspond to any realistic card (e.g., a 0-mana planeswalker with power/toughness)?
- How does the system handle oracle text containing unusual keywords or mechanics not seen in training data?
- What happens when all optional attributes are omitted and only card type is provided?
- How does the system handle cards with extreme price outliers in the training data (e.g., cards worth €500+ alongside cards worth €0.05)?
- What happens when the training dataset is very small (e.g., fewer than 100 cards)?
- What happens when a Forge card script has no matching price record (e.g., a newly added card not yet in MTGJSON)?
- How does the system handle multi-face cards (transform, split, adventure) where a single Forge script defines multiple faces with different attributes?
- How does the system distinguish paper-only cards from digital-only cards when filtering the training set?
- What happens when a user submits a made-up card with attributes far outside the range seen in training data (e.g., a creature with power 100)?
- What happens if progress logging is redirected or piped — does it still appear on the console (stderr) without corrupting the JSON output (stdout)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept card attributes as input, including: mana cost (with distinct handling of generic, colored, and colorless mana components), card type(s), permanent type(s), oracle text, power, toughness, and keyword abilities.
- **FR-002**: System MUST return a numeric EUR price estimate for any valid combination of input attributes.
- **FR-003**: System MUST handle partial input gracefully — not all attributes are required for every card (e.g., non-creature cards have no power/toughness).
- **FR-004**: System MUST support training on a dataset of paper cards with known attributes and market prices. Digital-only cards (e.g., MTG Arena exclusives, Alchemy cards) MUST be excluded from the training set.
- **FR-004a**: System MUST exclude any card that has no available price data from the training set. Only cards with a confirmed paper market price may be used for training.
- **FR-004b**: System MUST accept prediction requests for any combination of valid card attributes, including cards that do not exist in any real set. The model MUST NOT reject input solely because it does not match a known card.
- **FR-005**: System MUST produce reproducible predictions — the same input attributes with the same model version MUST yield the same price estimate.
- **FR-006**: System MUST validate all input attributes at the boundary (e.g., mana cost format, numeric power/toughness values, non-empty card type).
- **FR-007**: System MUST report training errors and data quality issues (skipped rows, invalid entries) without silently dropping data.
- **FR-008**: System MUST support model evaluation against a test dataset, reporting mean absolute error and median percentage error.
- **FR-009**: System MUST treat the trained model as a versioned artifact — each training run produces a distinct model that can be identified.
- **FR-010**: System MUST display human-readable progress messages during long-running operations (training and evaluation) so the user can see that the system is actively working.
- **FR-011**: System MUST display progress messages on stderr, separate from the structured JSON result on stdout, so that piping or redirecting output does not mix progress text with machine-readable results.
- **FR-012**: System MUST identify each major processing stage in its progress messages (e.g., parsing cards, loading prices, building mappings, training, saving). For stages that process many items, periodic progress updates (e.g., counts or completion indicators) MUST be provided.

### Key Entities

- **Card**: A Magic: The Gathering card described by its game attributes. Key attributes: mana cost (string, e.g., "{2}{W}{W}" or "{3}{C}{C}"), card types (list, e.g., ["Creature", "Enchantment", "Scheme", "Plane", "Conspiracy", "Vanguard", "Phenomenon"]), permanent types (list, e.g., ["Angel"]), oracle text (string), power (numeric or null), toughness (numeric or null), keyword abilities (list, e.g., ["Flying", "Vigilance"]). Mana costs may contain generic mana ({1}–{N}, payable by any color or colorless), colored mana ({W}, {U}, {B}, {R}, {G}), and colorless mana ({C}, payable only by colorless mana — distinct from generic).
- **Price Estimate**: The system's predicted EUR market price for a given card. Attributes: predicted price (numeric, EUR), model version used (identifier).
- **Training Dataset**: A collection of paper-only cards with known market prices used to train the model. Each entry pairs a Card's attributes with its actual EUR Cardmarket price. Cards without a price and digital-only cards are excluded.
- **Trained Model**: The prediction model artifact produced by training. Attributes: model version/identifier, training date, training dataset summary (number of cards, price range).

## Assumptions

- Cards are from Magic: The Gathering (MTG). The terminology (mana cost, oracle text, P/T) is MTG-specific.
- **Paper cards only.** Digital-only cards (MTG Arena exclusives, Alchemy, and other formats without a secondary paper market) are excluded from training because they have no meaningful paper price. The MTGJSON price data's Cardmarket `paper` section is the authoritative filter.
- **Made-up cards are a first-class use case.** The system's primary purpose is to predict what a card *would* cost based on its characteristics. The prediction input does not need to match any real card.
- "Price" means the current EUR paper market price from Cardmarket, as provided by MTGJSON data files.
- Input is structured card attributes (individual fields), not freeform natural language text that needs parsing.
- **Colorless mana ({C}) is distinct from generic mana ({1}–{N}).** Generic mana in a cost means "pay with any type of mana." Colorless mana in a cost means "pay only with colorless mana" (such as mana produced by Sol Ring). The system must track these separately — colorless mana pips contribute to the mana value but do not add any color to the card.
- **Card attribute data** is sourced from MTG Forge card scripts — plain text files stored in the local Forge repository checkout at `../forge`. Each card is a `.txt` file containing key-value pairs (Name, ManaCost, Types, PT, keywords, oracle text, abilities, etc.). There are 32,000+ card scripts available.
- **Price data** is sourced from MTGJSON's AllPricesToday.json, downloaded once and frozen in the resources folder. This file is keyed by UUID, so a name-to-UUID mapping is needed to join card attributes to prices.
- Forge card scripts are oracle-level data (one file per card, not per printing). They do NOT contain rarity, set code, or other printing-specific attributes. Rarity is explicitly excluded as a model input — the model predicts from game characteristics only. This keeps the model consistent with the made-up card use case where rarity is undefined.
- For cards with multiple printings at different prices, the system uses the cheapest price across all printings (including foil and special printings). The overall lowest price is used regardless of printing type.
- The system does not scrape or fetch prices from external services at runtime.
- A "decent" price estimate means the model should be meaningfully better than random guessing — specific accuracy thresholds are defined in Success Criteria.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system produces a price estimate for any valid card input in under 2 seconds.
- **SC-002**: On a held-out test set, the model achieves a median percentage error of 50% or less (i.e., for the median card, the prediction is within 50% of the actual price).
- **SC-003**: The model correctly distinguishes price tiers — cards predicted in the top 20% by price should overlap with at least 60% of actual top-20% cards.
- **SC-004**: Training on a dataset of 10,000+ cards completes within 10 minutes on a standard workstation.
- **SC-005**: Predictions are reproducible — running the same input through the same model version yields identical results 100% of the time.
- **SC-006**: During training, the user sees at least one progress message within the first 5 seconds of execution — before any long-running stage has completed — confirming the system has started working.
- **SC-007**: Progress messages are written to stderr only and never appear in stdout, so that `python -m price_predictor train ... > result.json` captures clean JSON while the user sees progress on the console.
