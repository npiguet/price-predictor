# Data Model: Card Price Predictor

**Date**: 2026-03-01 (revised)
**Feature**: 001-card-price-predictor

## Domain Entities

### Card

Represents a Magic: The Gathering card's game-relevant attributes,
parsed from a Forge card script. This is oracle-level data (one entry
per card, not per printing). Contains no printing-specific fields
(rarity, set code, UUID).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | yes | Card name (from Forge `Name:` field) |
| mana_cost | ManaCost or null | no | Parsed mana cost |
| types | list[string] | yes | Card types (Creature, Instant, etc.) |
| supertypes | list[string] | yes | Supertypes (Legendary, Basic, etc.) |
| subtypes | list[string] | yes | Subtypes (Human, Wizard, Aura, etc.) |
| oracle_text | string or null | no | Rules text (from `Oracle:` field) |
| keywords | list[string] | yes | Keyword abilities (from `K:` lines) |
| power | string or null | no | Power (string to handle "*", "X") |
| toughness | string or null | no | Toughness (string to handle "*", "X") |
| loyalty | string or null | no | Planeswalker starting loyalty |
| layout | string | yes | Card layout (normal, doublefaced, split, adventure, modal, flip) |
| ability_count | int | yes | Number of defined abilities (A:, T:, S: lines) |

**Validation rules**:
- `name` must be non-empty
- `types` must contain at least one entry
- `power`/`toughness` if present must be a number, "*", or "X"
- `layout` must be one of: normal, doublefaced, split, adventure,
  modal, flip

**Not included** (by design):
- `uuid` — printing-level, not available in Forge scripts
- `rarity` — excluded per spec (printing attribute, not game attribute)
- `setCode` — not available, not needed
- `colors` — derived from ManaCost value object instead
- `color_identity` — derived from ManaCost + rules text analysis

### PriceEstimate

The system's output for a single prediction.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| predicted_price_eur | float | yes | Predicted EUR market price |
| model_version | string | yes | Identifier of the model used |

**Validation rules**:
- `predicted_price_eur` must be >= 0
- `model_version` must be non-empty

### TrainingExample

A Card paired with its known EUR market price, used for model training.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| card | Card | yes | The card's attributes |
| actual_price_eur | float | yes | Cheapest EUR Cardmarket price across all printings |

**Validation rules**:
- `actual_price_eur` must be > 0

### TrainedModel

Metadata about a trained prediction model (the model artifact itself
is an infrastructure concern — serialized file on disk).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_version | string | yes | Unique identifier (ISO timestamp) |
| training_date | string | yes | ISO 8601 date of training |
| card_count | int | yes | Number of cards in training set |
| price_range_min_eur | float | yes | Lowest price in training set (EUR) |
| price_range_max_eur | float | yes | Highest price in training set (EUR) |
| metrics | EvaluationMetrics or null | no | If evaluation was run at training time |

### EvaluationMetrics

Accuracy metrics from model evaluation.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| mean_absolute_error_eur | float | yes | Mean absolute error in EUR |
| median_percentage_error | float | yes | Median percentage error (0-100+) |
| top_20_overlap | float | yes | Fraction overlap of predicted vs actual top-20% |
| sample_count | int | yes | Number of cards in the evaluation set |

## Relationships

```
TrainingExample ──contains──► Card
TrainingExample ──has──► actual_price_eur

TrainedModel ──produces──► PriceEstimate (at prediction time)
TrainedModel ──optionally has──► EvaluationMetrics

Card ──input to──► prediction function ──returns──► PriceEstimate
```

## Value Objects

### ManaCost

Parsed representation of a mana cost string in Forge format
(space-separated shards like `2 W W`). Extracts:
- Total mana value (numeric CMC)
- Generic mana component
- Per-color mana counts (W, U, B, R, G)
- Color count (number of distinct colors)
- Presence of X cost
- Presence of hybrid mana
- Presence of phyrexian mana

**Parsing rules** (Forge mana format):
- `W`, `U`, `B`, `R`, `G` → colored mana (1 each)
- `C` → colorless mana (1 CMC, no color)
- `1`, `2`, ... `N` → generic mana
- `X` → variable (0 CMC, flag X-present)
- `WU`, `WB`, etc. → hybrid (0.5 each color for counting, 1 CMC)
- `WP`, `UP`, etc. → phyrexian (1 color, 1 CMC)
- `W2`, `U2`, etc. → two-or-colored hybrid (1 CMC)
- `S` → snow mana (1 CMC, no color)
- `no cost` → null ManaCost (card cannot be cast)

### CardFeatureVector

The numeric representation of a Card after feature engineering.
This is an application-layer concern (the transformation from domain
entity to model input), not stored or persisted.
