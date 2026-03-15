# Feature Specification: Printing Data Fields in Training Data

**Feature Branch**: `009-printing-metadata`
**Created**: 2026-03-15
**Status**: Draft
**Input**: User description: "Add printing metadata to card training data and prediction input. Enrich card text with isReserved, rarity, printings count, set code, and format legalities. Auto-fill for known cards, optional with defaults for unknown cards."
**Depends on**: `008-model-harmonization` (current model architecture, training pipeline, and card text format)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enrich Training Data with Printing Data Fields (Priority: P1)

A model developer wants to retrain both sklearn and transformer models with printing data fields included in the card text, so that the models can learn price-relevant signals beyond the card's oracle text (such as reserve list status, rarity, reprint count, set identity, and format legality).

The system reads AllPrintings.json during data loading and, for each card matched to a price, appends five printing data fields to the card text: whether the card is on the reserve list, the rarity of the cheapest printing, the total number of sets the card has been printed in, the set code of the cheapest printing, and the list of constructed formats in which the card is legal. The enriched text is then used for training both models.

**Why this priority**: Without enriched training data, the models cannot learn from printing metadata. This is the foundation — prediction and API changes are meaningless until the models are trained on the new input format.

**Independent Test**: Can be tested by running the data loading pipeline and verifying that the output card text for a known card (e.g., Black Lotus) includes the correct printing data fields with values matching AllPrintings.json.

**Acceptance Scenarios**:

1. **Given** AllPrintings.json and AllPricesToday.json are available, **When** the data loader produces training tuples, **Then** each card's text includes a `reserved:` field with value `true` or `false`.
2. **Given** a card with multiple printings at different rarities, **When** the cheapest printing is identified, **Then** the `rarity:` field reflects the rarity of that specific cheapest printing (not another printing).
3. **Given** a card printed in 5 different sets, **When** the card text is generated, **Then** the `printings:` field contains the value `5`.
4. **Given** a card whose cheapest printing is from set "UMA", **When** the card text is generated, **Then** the `set:` field contains `uma` (lowercased).
5. **Given** a card that is legal in Commander and Modern but banned in Standard, **When** the card text is generated, **Then** the `legalities:` field lists `commander, modern` (and any other legal formats) but not `standard`.
6. **Given** a card on the reserve list, **When** the card text is generated, **Then** the `reserved:` field is `true`.
7. **Given** a card not on the reserve list (or where the field is absent in AllPrintings), **When** the card text is generated, **Then** the `reserved:` field is `false`.

---

### User Story 2 - Predict Known Cards with Auto-Filled Metadata (Priority: P2)

A user wants to predict the price of an existing MTG card through the API. When they provide only the card name (or card text without the five printing data fields), the system automatically looks up the card in AllPrintings.json and fills in all five fields from the cheapest printing's data before running the prediction.

**Why this priority**: This is the primary prediction use case — most API calls will be for known cards. Auto-filling metadata ensures predictions use the richest available information without requiring the client to look up printing data themselves.

**Independent Test**: Can be tested by calling the prediction API with a known card name and verifying the response uses the correct printing data values from AllPrintings.

**Acceptance Scenarios**:

1. **Given** a known card name is submitted to the prediction API with card text that does not contain the five printing data fields, **When** the prediction runs, **Then** the system auto-fills all five fields from the cheapest printing and uses the enriched text for prediction.
2. **Given** a known card that is on the reserve list, **When** the prediction is made, **Then** the auto-filled `reserved` field is `true`.
3. **Given** a known card printed in 12 sets, **When** the prediction is made, **Then** the auto-filled `printings` field is `12`.

---

### User Story 3 - Predict Unknown Cards with Optional Metadata (Priority: P3)

A user wants to predict the price of an unknown card (a spoiler for a future set, or a custom card). The card is not in AllPrintings.json, so the system cannot auto-fill the printing data fields. The user may optionally include some or all five fields inline in the card text. Any fields not present in the card text receive sensible defaults.

**Why this priority**: This is a secondary use case — predicting prices for cards not yet in the database. It requires the default value logic and API parameter support, which build on the auto-fill mechanism from P2.

**Independent Test**: Can be tested by calling the prediction API with card text that has no match in AllPrintings, with and without optional printing data fields, and verifying defaults are applied correctly.

**Acceptance Scenarios**:

1. **Given** an unknown card submitted with card text containing none of the five printing data fields, **When** the prediction runs, **Then** the system applies defaults: `reserved: false`, `rarity: rare`, `printings: 1`, `set: ukn`, `legalities: standard, pioneer, modern, brawl, legacy, vintage, pauper, commander, penny, oathbreaker`.
2. **Given** an unknown card submitted with card text that includes `rarity: mythic` and `legalities: commander, legacy` but not the other three fields, **When** the prediction runs, **Then** the provided values are used for rarity and legalities, and defaults are applied for the remaining three fields.
3. **Given** an unknown card submitted with all five printing data fields present in the card text, **When** the prediction runs, **Then** all provided values are used and no defaults are applied.

---

### Edge Cases

- What happens when a card exists in AllPrintings but has no price in AllPricesToday (so no "cheapest printing" can be determined)? The card is excluded from training data (existing behavior). For prediction, the system applies defaults (same as unknown cards) since no cheapest printing can be determined.
- What happens when a card has `isReserved` absent in AllPrintings? Treat it as `false` (MTGJSON only includes the field when true).
- What happens when a card is legal in a format not in our list of 10 (e.g., an online-only format)? That format is ignored; only the 10 recognized constructed formats are included.
- What happens when a card is not legal in any of the 10 constructed formats (e.g., banned everywhere)? The legalities field is empty.
- What happens when a card has only one printing and that printing has no rarity information? Default to `rare` as fallback.
- What happens when the client includes printing data fields in the card text for a known card? The values present in the card text override the auto-filled values, allowing the client to ask "what if" questions (e.g., "what would this card be worth if it were mythic?").

## Clarifications

### Session 2026-03-15

- Q: What should happen if retrained models show worse accuracy after adding metadata? → A: Informational only — log the comparison but ship regardless; metadata fields are kept no matter the accuracy impact.
- Q: Should existing API clients that send no metadata fields continue to work without changes? → A: Fully backward-compatible — existing requests work unchanged, metadata auto-filled or defaulted silently.
- Q: Are the five fields separate API parameters or inline card text data? → A: They are card text data fields, not separate API parameters. When provided, they appear inline within the card text body (same format as training data). The system parses them from the card text if present; if absent, it auto-fills or defaults them.
- Q: How should prediction auto-fill determine the "cheapest printing" for known cards? → A: Use AllPricesToday.json at prediction time (same logic as training).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The data loader MUST extract five printing data fields from AllPrintings.json for each card matched to a price: reserve list status (boolean), rarity (of the cheapest printing), total number of distinct printings, set code (of the cheapest printing), and format legalities (list of legal formats).
- **FR-002**: The printing data fields MUST be appended to the card text in a consistent format that both sklearn and transformer models can consume. The fields are lowercased and use the same line-based format as existing card text fields.
- **FR-003**: The legalities field MUST include only the following 10 constructed formats: Standard, Pioneer, Modern, Brawl, Legacy, Vintage, Pauper, Commander, Penny, Oathbreaker. Online-only formats MUST be excluded.
- **FR-004**: For known cards (found in AllPrintings), the prediction pipeline MUST auto-fill all five printing data fields from the cheapest printing's data (determined by current AllPricesToday.json prices, same logic as training) when they are not present in the submitted card text.
- **FR-005**: For unknown cards (not in AllPrintings) or when specific fields are not present in the submitted card text, the system MUST apply these defaults: `reserved: false`, `rarity: rare`, `printings: 1`, `set: ukn`, `legalities: standard, pioneer, modern, brawl, legacy, vintage, pauper, commander, penny, oathbreaker`.
- **FR-006**: The five printing data fields are part of the card text body, not separate API parameters. The prediction API MUST parse these fields from the submitted card text if present. Client-provided values within the card text MUST override auto-filled or default values. The API MUST remain fully backward-compatible: existing requests with card text that does not contain these fields MUST continue to work unchanged, with fields auto-filled (known cards) or defaulted (unknown cards) silently.
- **FR-007**: The same enriched card text format (with printing data fields) MUST be used consistently across training, evaluation, and prediction to ensure the models see the same input format at all stages.
- **FR-008**: Both sklearn and transformer models MUST be retrained on the enriched card text format after this change. Models trained without printing data fields MUST NOT be used with enriched input.

### Key Entities

- **Printing Data Fields**: Five card text fields describing the printing context of a card: reserve list status, rarity, reprint count, set code, and format legalities. These are inline data within the card text body (not separate API parameters). Derived from AllPrintings.json for known cards, parsed from client-provided card text if present, or filled with defaults for unknown cards.
- **Cheapest Printing**: The specific printing of a card that has the lowest price in AllPricesToday.json. Metadata fields (rarity, set code) are taken from this printing, not from other printings of the same card.

## Assumptions

- The existing data loader already identifies the cheapest printing per card when matching prices. The metadata extraction builds on this existing logic.
- AllPrintings.json contains `isReserved`, `rarity`, `printings` (list of set codes), `setCode`, and `legalities` fields for all cards. The `isReserved` field is only present when true (boolean, absent means false).
- The 10 constructed formats listed are stable and do not change frequently. If new formats are added to MTGJSON, they are ignored until explicitly added to the recognized list.
- The set code is the 3-letter (or occasionally 2-4 letter) code as provided by MTGJSON, lowercased.
- Retraining both models is required after this change and is part of the normal workflow (not a separate feature).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All cards in the training dataset have all five printing data fields present in their card text (100% coverage).
- **SC-002**: For any known card, the auto-filled printing data matches the values in AllPrintings.json for the cheapest printing (100% accuracy).
- **SC-003**: Predictions for unknown cards with no metadata provided use the correct default values for all five fields.
- **SC-004**: The retrained models are evaluated against the models trained without metadata (measured by existing evaluation metrics: MAE, median percentage error, top-20% overlap). This comparison is **informational only** — accuracy regression does not block the feature; the metadata fields are retained regardless of impact on prediction accuracy.
