# Feature Specification: CardMarket EUR Pricing

**Feature Branch**: `004-cardmarket-eur-pricing`
**Created**: 2026-02-27
**Status**: Draft
**Input**: User description: "Prices come from the MTGJSON dataset and use the CardMarket price in EUR."
**Refines**: `001-card-price-predictor` (replaces USD/TCGPlayer assumption with EUR/CardMarket)

## Clarifications

### Session 2026-03-01

- Q: FR-007/US3-AC3 says €0.00 cards get a 0.00 label, but feature 003 clamps all prices below €0.01 to €0.01. Which takes precedence? → A: Feature 003's €0.01 floor takes precedence. Zero-priced cards are *included* in training (not excluded as missing), but the final training label is €0.01 after the floor is applied.
- Q: The codebase already uses CardMarket EUR prices and ignores TCGPlayer. Is this feature primarily formalization of existing behavior plus FR-008 exclusion count? → A: Yes. FR-001–FR-006, FR-009, FR-010 formalize existing behavior. FR-008 (exclusion count reporting) is the only new code change.
- Q: Should FR-008's exclusion count use stderr INFO log (like feature 003), WARNING, or both? → A: Stderr INFO log in `build_price_map()`, consistent with feature 003's logging pattern.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Training Data Uses CardMarket EUR Prices (Priority: P1)

When the system builds its training dataset, it sources card prices from the MTGJSON dataset, specifically using CardMarket prices denominated in EUR. Each card's training label is the CardMarket EUR price (subject to the cheapest-printing rule from feature 003 when multiple printings exist). This means the model learns European market pricing patterns and all predictions are in EUR.

This replaces the original assumption in feature 001 that prices would be in USD from TCGPlayer or a similar US-focused aggregator.

**Why this priority**: The choice of price source and currency is foundational to the entire training pipeline and prediction output. Every other price-related decision depends on this. The model cannot be trained until the system knows which prices to extract from the MTGJSON data.

**Independent Test**: Can be tested by processing the MTGJSON price data file, extracting prices for a sample of cards, and verifying that the extracted prices match the CardMarket EUR values in the source data (not TCGPlayer, not USD).

**Acceptance Scenarios**:

1. **Given** the MTGJSON price data contains both CardMarket (EUR) and TCGPlayer (USD) prices for a card, **When** the system extracts the training price, **Then** the CardMarket EUR price is used and the TCGPlayer USD price is ignored.
2. **Given** a card has a CardMarket EUR price of €2.50 in the MTGJSON data, **When** the system prepares the training label, **Then** the price is recorded as 2.50 EUR.
3. **Given** the MTGJSON price data for a card contains CardMarket prices for multiple date snapshots, **When** the system extracts the price, **Then** it uses the most recent available snapshot.

---

### User Story 2 - Predictions Output in EUR (Priority: P2)

When a user requests a price prediction for a card (real or hypothetical), the system returns the estimate in EUR, consistent with the CardMarket-based training data. The user sees a EUR value, not USD or any other currency. No currency conversion is performed — the prediction is natively in EUR because the model was trained on EUR prices.

**Why this priority**: Prediction output currency follows directly from the training data currency. Once P1 establishes EUR as the training currency, predictions are inherently in EUR. This story makes the EUR output explicit as a user-facing requirement.

**Independent Test**: Can be tested by submitting card attributes for prediction and verifying the returned price is labeled as EUR and is consistent in magnitude with CardMarket EUR market prices (not inflated/deflated as if converted from another currency).

**Acceptance Scenarios**:

1. **Given** the model is trained on CardMarket EUR prices, **When** a user requests a price prediction, **Then** the returned estimate is denominated in EUR.
2. **Given** a card whose actual CardMarket price is approximately €5.00, **When** the user requests a prediction, **Then** the prediction is in the same order of magnitude (EUR), not in a different currency's range (e.g., not $5.50 USD).

---

### User Story 3 - Cards Without CardMarket Prices Are Excluded (Priority: P3)

Some cards in the MTGJSON dataset may have prices from other vendors (e.g., TCGPlayer) but no CardMarket price. Since the system exclusively uses CardMarket EUR prices, these cards cannot be assigned a training label and are excluded from the training set. The system reports how many cards were excluded for this reason.

**Why this priority**: Data completeness and exclusion reporting support training data quality. The system can train without this reporting (P1 handles the core extraction), but transparency about excluded cards helps users understand the effective training set size.

**Independent Test**: Can be tested by including cards in the price data that have only TCGPlayer prices (no CardMarket entry), running training data preparation, and verifying those cards are excluded and counted in the exclusion report.

**Acceptance Scenarios**:

1. **Given** a card has a TCGPlayer USD price but no CardMarket EUR price in the MTGJSON data, **When** the system prepares training data, **Then** the card is excluded from the training set.
2. **Given** some cards lack CardMarket prices, **When** training data preparation completes, **Then** the system reports the count of cards excluded due to missing CardMarket prices.
3. **Given** a card has a CardMarket price of €0.00 (explicitly zero), **When** the system prepares training data, **Then** the card is included in the training set (zero is a valid price, not the same as missing) with a training label of €0.01 after the price floor from feature 003 is applied.

---

### Edge Cases

- What happens when a card has a CardMarket entry but the price value is null or empty? The card is treated as having no CardMarket price and is excluded from training.
- What happens when the MTGJSON data contains CardMarket prices in a nested structure with both "retail" and "buylist" prices? The system uses the retail price (the price a buyer would pay), not the buylist price (what a store offers to buy it for).
- What happens when the MTGJSON data format changes and the CardMarket price field moves or is renamed? The system fails clearly with a descriptive error indicating the expected price field was not found.
- What happens when all cards in the dataset lack CardMarket prices (e.g., using a TCGPlayer-only price file)? The system reports that zero cards have valid training prices and does not produce a model.
- How does this interact with the cheapest-printing rule (feature 003)? The cheapest-printing comparison is performed across CardMarket EUR prices only — prices from other vendors are never considered.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST source all training prices from the CardMarket section of the MTGJSON price dataset.
- **FR-002**: All training prices MUST be in EUR as provided by CardMarket. No currency conversion is performed.
- **FR-003**: The system MUST ignore prices from other vendors in the MTGJSON dataset (e.g., TCGPlayer, CardKingdom) for training purposes.
- **FR-004**: When the MTGJSON data contains multiple CardMarket price snapshots, the system MUST use the most recent available snapshot.
- **FR-005**: When CardMarket provides both retail and buylist prices, the system MUST use the retail price.
- **FR-006**: Cards with no CardMarket EUR price (missing, null, or absent from the CardMarket section) MUST be excluded from the training set.
- **FR-007**: Cards with a CardMarket EUR price of exactly 0.00 MUST be included in the training set (zero is a valid price, not treated as missing). The €0.01 price floor from feature 003 applies — the final training label is €0.01, not €0.00.
- **FR-008**: The system MUST report the count of cards excluded due to missing CardMarket prices during training data preparation, as an INFO-level log message to stderr in `build_price_map()`, consistent with feature 003's logging pattern.
- **FR-009**: All price predictions output by the system MUST be denominated in EUR, consistent with the training data currency.
- **FR-010**: The system MUST NOT perform any currency conversion at any stage — training, prediction, or evaluation.

### Key Entities

- **CardMarket Price**: The retail price of a card on the CardMarket marketplace, denominated in EUR. Sourced from the MTGJSON dataset's CardMarket section. This is the authoritative price for training and prediction.
- **MTGJSON Price Dataset**: The frozen AllPricesToday.json file from MTGJSON (as referenced in feature 001). Contains price data from multiple vendors; only the CardMarket EUR section is used.
- **Excluded Card**: A card present in the card attribute data but absent from the CardMarket price data, making it ineligible for training.

## Assumptions

- The MTGJSON AllPricesToday.json file contains a CardMarket section with EUR prices. This is consistent with MTGJSON's documented data structure, which includes CardMarket as a standard vendor alongside TCGPlayer.
- CardMarket prices in MTGJSON represent the "trend" or average retail price on the CardMarket marketplace (cardmarket.com), the dominant European MTG secondary market.
- The EUR price from CardMarket is used as-is. No adjustment for shipping, fees, or regional tax differences is applied.
- This decision means the model is trained on European market pricing, which may differ from US (TCGPlayer) pricing for the same cards. This is an intentional choice — the system targets the European market.
- The MTGJSON data structure nests prices under vendor > card UUID > price type. The system navigates this structure to extract CardMarket retail prices.
- This feature supersedes the feature 001 assumption that prices are "USD paper market price (e.g., TCGPlayer market price or similar)." The definitive source is CardMarket in EUR.
- The existing codebase already implements CardMarket EUR price extraction (FR-001–FR-006, FR-009, FR-010). This feature formalizes that behavior as explicit requirements. The only new code change is FR-008 (exclusion count reporting).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of training labels are sourced from CardMarket EUR prices — zero training labels come from any other vendor or currency.
- **SC-002**: When verified against a manual sample of 20 cards, the extracted CardMarket EUR price matches the value in the raw MTGJSON data file with zero discrepancies.
- **SC-003**: All price predictions are denominated in EUR with zero instances of USD or other currency values in any output.
- **SC-004**: The count of cards excluded due to missing CardMarket prices is reported accurately, with zero discrepancies when verified against a manual count of cards lacking CardMarket entries.
- **SC-005**: At least 70% of cards in the MTGJSON dataset have CardMarket EUR prices available, ensuring a sufficiently large training set (estimated 20,000+ cards).
