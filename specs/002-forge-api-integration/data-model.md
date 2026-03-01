# Data Model: Forge API Integration

**Date**: 2026-03-01
**Feature**: 002-forge-api-integration

This feature does not introduce new domain entities. It reuses
`Card` and `PriceEstimate` from feature 001 and adds infrastructure-
level types for the REST service and Java connector.

## Reused Domain Entities (feature 001)

### Card

Unchanged. See `specs/001-card-price-predictor/data-model.md`.

### PriceEstimate

Unchanged. Fields: `predicted_price_eur` (float), `model_version`
(string).

## HTTP Request/Response (infrastructure)

### Evaluation Request

| Field | Type | Location | Description |
|-------|------|----------|-------------|
| body | string | HTTP body | Forge card script text |
| content-type | string | HTTP header | Must be `text/plain` |

**Validation rules**:
- Body must be non-empty
- Body must be valid UTF-8
- Body must contain at least one parseable `Types:` line

### Evaluation Response (success)

```json
{
  "predicted_price_eur": 12.45,
  "model_version": "20260301-143000"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| predicted_price_eur | float | yes | EUR price estimate (>= 0, 2 decimals) |
| model_version | string | yes | Model identifier (timestamp format) |

### Evaluation Response (error)

```json
{
  "error": "Failed to parse card script: no Types line found"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| error | string | yes | Human-readable error description |

**HTTP status codes**:
- 200: Successful prediction
- 400: Invalid or unparseable input
- 500: Internal server error (model failure)

## Java Connector Types

### CardAttributes (builder pattern)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | String | no | Card name |
| manaCost | String | no | Mana cost in Forge format (e.g., "2 W W") |
| types | List\<String\> | yes | Card types (Creature, Instant, etc.) |
| supertypes | List\<String\> | no | Supertypes (Legendary, Basic, etc.) |
| subtypes | List\<String\> | no | Subtypes (Human, Wizard, etc.) |
| oracleText | String | no | Rules text |
| keywords | List\<String\> | no | Keyword abilities |
| power | String | no | Power (number, "*", "X") |
| toughness | String | no | Toughness (number, "*", "X") |
| loyalty | String | no | Planeswalker starting loyalty |

**Validation rules** (enforced by builder):
- `types` must be non-null and non-empty (at least one card type)
- All other fields are optional (nullable)

### PriceEstimate (Java record)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| predictedPriceEur | double | yes | EUR price estimate |
| modelVersion | String | yes | Model version identifier |

### PricePredictorException

| Field | Type | Description |
|-------|------|-------------|
| message | String | Error description |
| cause | Throwable | Underlying exception (timeout, I/O, parse) |

Subtypes:
- `ServiceUnavailableException` — connection refused or timeout
- `InvalidResponseException` — server returned non-200 or unparseable JSON

## Relationships

```
CardAttributes ──serialized to──► Forge script text ──HTTP POST──► Server
Server ──parses──► Card (domain entity) ──predicts──► PriceEstimate
Server ──returns──► JSON response ──parsed by──► PriceEstimate (Java)
```
