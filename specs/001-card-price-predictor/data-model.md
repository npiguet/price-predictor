# Data Model: Card Price Predictor

**Date**: 2026-02-26
**Feature**: 001-card-price-predictor

## Domain Entities

### Card

Represents a Magic: The Gathering card's game-relevant attributes,
independent of any data source or file format.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| uuid | string | yes | Unique identifier for this printing |
| name | string | yes | Card name |
| mana_cost | string or null | no | Mana cost notation, e.g., "{2}{W}{W}" |
| mana_value | float | yes | Converted mana cost as a number |
| colors | list[string] | yes | Colors from cost/indicator (W, U, B, R, G) |
| color_identity | list[string] | yes | Full color identity |
| types | list[string] | yes | Card types (Creature, Instant, etc.) |
| supertypes | list[string] | yes | Supertypes (Legendary, Basic, etc.) |
| subtypes | list[string] | yes | Subtypes (Human, Wizard, Aura, etc.) |
| oracle_text | string or null | no | Rules text |
| keywords | list[string] | yes | Keyword abilities (Flying, Trample, etc.) |
| power | string or null | no | Power (string to handle "*", "X") |
| toughness | string or null | no | Toughness (string to handle "*", "X") |
| loyalty | string or null | no | Planeswalker loyalty |
| rarity | string | yes | common, uncommon, rare, mythic |
| layout | string | yes | Card layout (normal, transform, split, etc.) |

**Validation rules**:
- `uuid` must be non-empty
- `mana_value` must be >= 0
- `types` must contain at least one entry
- `rarity` must be one of: common, uncommon, rare, mythic, special, bonus
- `power`/`toughness` if present must be a number, "*", or "X"

### PriceEstimate

The system's output for a single prediction.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| predicted_price_usd | float | yes | Predicted USD market price |
| model_version | string | yes | Identifier of the model used |

**Validation rules**:
- `predicted_price_usd` must be >= 0
- `model_version` must be non-empty

### TrainingExample

A Card paired with its known market price, used for model training.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| card | Card | yes | The card's attributes |
| actual_price_usd | float | yes | Known USD market price |

**Validation rules**:
- `actual_price_usd` must be > 0

### TrainedModel

Metadata about a trained prediction model (the model artifact itself
is an infrastructure concern — serialized file on disk).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_version | string | yes | Unique identifier (e.g., timestamp-based) |
| training_date | string | yes | ISO 8601 date of training |
| card_count | int | yes | Number of cards in training set |
| price_range_min | float | yes | Lowest price in training set |
| price_range_max | float | yes | Highest price in training set |
| metrics | EvaluationMetrics or null | no | If evaluation was run at training time |

### EvaluationMetrics

Accuracy metrics from model evaluation.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| mean_absolute_error | float | yes | Mean absolute error in USD |
| median_percentage_error | float | yes | Median percentage error (0-100+) |
| top_20_overlap | float | yes | Fraction overlap of predicted vs actual top-20% |
| sample_count | int | yes | Number of cards in the evaluation set |

## Relationships

```
TrainingExample ──contains──► Card
TrainingExample ──has──► actual_price_usd

TrainedModel ──produces──► PriceEstimate (at prediction time)
TrainedModel ──optionally has──► EvaluationMetrics

Card ──input to──► prediction function ──returns──► PriceEstimate
```

## Value Objects

### ManaCost

Parsed representation of a mana cost string. Extracts:
- Total mana value (numeric)
- Generic mana component
- Per-color mana counts (W, U, B, R, G)
- Presence of X, hybrid, or phyrexian mana

### CardFeatureVector

The numeric representation of a Card after feature engineering.
This is an application-layer concern (the transformation from domain
entity to model input), not stored or persisted.
