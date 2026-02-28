# Feature Specification: Cheapest Printing Price for Training

**Feature Branch**: `003-cheapest-printing-price`
**Created**: 2026-02-27
**Status**: Draft
**Input**: User description: "if a card has multiple printings (with different set, foiling, rarity, whatever) then only the price of the cheapest version is used for training."
**Refines**: `001-card-price-predictor` (replaces the vague "representative price" assumption with a concrete rule)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use Cheapest Printing Price for Training (Priority: P1)

When the system builds its training dataset, many cards exist across multiple printings — different sets, foil vs. non-foil, different rarities, promo versions, etc. Each printing may have a different market price. To keep training simple and consistent, the system selects the single cheapest available price across all versions of a card as the training label for that card.

This rule means the model learns to predict the floor price of a card based on its game attributes. Since Forge card scripts represent a card at the oracle level (one entry per unique card, not per printing), pairing each card with its cheapest printing price is the most natural and conservative mapping.

**Why this priority**: This is the core business rule this feature defines. The entire training data pipeline for multi-printing cards depends on this decision. Without it, the system has no deterministic way to assign a price to cards with multiple printings.

**Independent Test**: Can be fully tested by providing a card that has multiple printings with different prices in the price data source, running the training data preparation, and verifying that only the cheapest price is selected as the training label.

**Acceptance Scenarios**:

1. **Given** a card exists with 3 printings priced at $1.50, $3.00, and $8.00, **When** the system prepares training data for this card, **Then** the training label is $1.50 (the cheapest).
2. **Given** a card exists with both foil ($12.00) and non-foil ($2.00) printings, **When** the system prepares training data, **Then** the training label is $2.00 (the cheapest, regardless of foiling).
3. **Given** a card exists with printings across 5 different sets at varying prices, **When** the system prepares training data, **Then** the cheapest price across all sets is selected.
4. **Given** a card has only one printing with one price, **When** the system prepares training data, **Then** that single price is used (trivial case — no selection needed).

---

### User Story 2 - Training Data Transparency for Price Selection (Priority: P2)

When the system processes multi-printing cards during training data preparation, it reports which price was selected and how many alternative prices were available. This allows the user to verify the cheapest-price rule is working correctly and to identify cards where the price spread is unusually wide (which may indicate data quality issues or unusual market conditions).

**Why this priority**: Transparency supports debugging and trust in the training data. The system can function without it (P1 delivers the core behavior), but it significantly helps the user validate that the price selection rule produces sensible results.

**Independent Test**: Can be tested by running training data preparation and checking that the output includes a summary or log showing, for cards with multiple printings, the selected price and the number of alternatives.

**Acceptance Scenarios**:

1. **Given** a card has 4 printings with prices $0.25, $1.00, $5.00, and $15.00, **When** training data is prepared, **Then** the system reports that $0.25 was selected from 4 available prices.
2. **Given** all cards in the dataset have only one printing, **When** training data is prepared, **Then** no multi-printing selection events are reported.
3. **Given** training data preparation completes, **When** the user reviews the output, **Then** a summary shows how many cards had multiple printings and required price selection.

---

### Edge Cases

- What happens when a card has multiple printings but all printings have the same price? The system selects that price (trivially the cheapest); no special handling needed.
- What happens when the cheapest printing has a price of $0.00 or effectively free? The system uses $0.00 as the training label. This is a valid data point (some cards are genuinely near-worthless).
- What happens when one printing's price is missing or null but others have valid prices? The system ignores printings with missing prices and selects the cheapest among valid prices.
- What happens when ALL printings of a card have missing/null prices? The card is excluded from the training set entirely (consistent with feature 001's FR-004a: only cards with a confirmed price are used).
- What happens when a card has an extremely cheap promo printing (e.g., $0.01) that is far below all other printings? The system still uses $0.01 as the cheapest price. The rule is unconditional — no outlier filtering is applied at the price selection step.
- How does the system handle the same card name appearing with both paper and digital-only printings? Digital-only printings are already excluded from the price data (per feature 001's paper-only rule). The cheapest-price selection applies only to paper printings.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: When a card has multiple printings with different prices, the system MUST select the single cheapest price across all available versions as the training label for that card.
- **FR-002**: The cheapest-price selection MUST consider all printing variants, including different sets, foil/non-foil, different rarities, and promo versions. No variant type is excluded from comparison.
- **FR-003**: When a card has only one printing (or all printings share the same price), the system MUST use that price without any special handling.
- **FR-004**: Printings with missing or null prices MUST be ignored during price selection. The cheapest valid price is used.
- **FR-005**: If all printings of a card have missing or null prices, the card MUST be excluded from the training set (no training label can be assigned).
- **FR-006**: The cheapest-price rule MUST apply unconditionally — no outlier filtering or minimum price threshold is applied during price selection.
- **FR-007**: The system MUST report, during training data preparation, how many cards had multiple printings and required price selection.
- **FR-008**: For each card where price selection occurred, the system MUST make the selected price and the number of alternative prices available for inspection (e.g., in a log or summary report).

### Key Entities

- **Printing**: A specific version of a card tied to a set, foiling, and rarity. Each printing has its own market price. A single card may have many printings.
- **Training Label**: The price assigned to a card for model training purposes. For cards with multiple printings, this is the cheapest available price. For single-printing cards, this is the only available price.
- **Price Selection Record**: A record documenting which price was chosen for a multi-printing card, the number of alternative prices, and the price range. Used for transparency and debugging.

## Assumptions

- "Cheapest" means the lowest numeric USD paper market price across all printings. Ties are resolved arbitrarily (any printing with the minimum price is acceptable).
- This rule applies only to the training data pipeline. It does not affect how the model makes predictions at inference time — predictions are based on card attributes, not printing information.
- The paper-only filter from feature 001 is applied before this rule. Digital-only printings are not considered.
- The price data source (MTGJSON AllPricesToday.json) provides per-printing prices keyed by UUID. Multiple UUIDs may map to the same card name; the cheapest price among those UUIDs is selected.
- Foil and non-foil printings are treated equally — a foil's price is compared on the same basis as a non-foil's price.
- This rule supersedes the vague "representative price (e.g., cheapest normal printing or median across printings)" assumption in feature 001. The definitive rule is: always use the cheapest.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every card in the training set that has multiple printings, the assigned training price equals the minimum price across all available printings (100% compliance with the cheapest-price rule).
- **SC-002**: The training data preparation summary correctly reports the count of cards that had multiple printings and required price selection, with zero discrepancies when verified against a manual sample of 20 cards.
- **SC-003**: Cards where all printings have missing prices are excluded from the training set with zero exceptions.
- **SC-004**: The cheapest-price selection step adds no more than 30 seconds to the total training data preparation time (compared to a baseline with no multi-printing logic).
