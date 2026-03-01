# Data Model: Card Evaluation Endpoints

**Feature**: 005-card-eval-endpoints
**Date**: 2026-03-01

## Existing Entities (No Changes)

### Card (domain entity)

Defined in `src/price_predictor/domain/entities.py`. Frozen dataclass representing a parsed MTG card.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| name | str | Yes | Non-empty |
| types | list[str] | Yes | At least one recognized card type |
| supertypes | list[str] | No | Default empty |
| subtypes | list[str] | No | Default empty |
| mana_cost | ManaCost \| None | No | None for lands / "no cost" cards |
| oracle_text | str \| None | No | Card rules text |
| keywords | list[str] | No | Default empty |
| power | str \| None | No | Creature power |
| toughness | str \| None | No | Creature toughness |
| loyalty | str \| None | No | Planeswalker starting loyalty |
| layout | str | No | Default "normal" |
| ability_count | int | No | Default 0 |

### PriceEstimate (domain entity)

Defined in `src/price_predictor/domain/entities.py`. Frozen dataclass for prediction output.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| predicted_price_eur | float | Yes | >= 0, rounded to 2 decimal places |
| model_version | str | Yes | Non-empty identifier |

## Endpoint Data Shapes (Implicit — Not Formal Entities)

These are JSON shapes used at the HTTP boundary. They are not domain entities and do not require dedicated classes. They are documented here for contract clarity.

### Evaluation Request

HTTP `POST /api/v1/evaluate` with `Content-Type: text/plain`.

**Body**: Raw Forge card script text (UTF-8 string). Example:
```
Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target.
```

### Evaluation Response (Success — 200)

```json
{
  "predicted_price_eur": 2.35,
  "model_version": "20260301-143000"
}
```

| Field | Type | Notes |
|-------|------|-------|
| predicted_price_eur | float | >= 0, 2 decimal places |
| model_version | string | Non-empty |

### Evaluation Response (Error — 400/500)

```json
{
  "error": "Failed to parse card script: No Name field found in card script"
}
```

| Field | Type | Notes |
|-------|------|-------|
| error | string | Human-readable error description |

## Structured Log Entry (FR-012)

Not a domain entity. A JSON object written to stderr by the endpoint for each request.

### Success Log

| Field | Type | Notes |
|-------|------|-------|
| event | string | Always `"evaluate_request"` |
| timestamp | string | ISO 8601 with microseconds |
| status_code | int | HTTP status code (200, 400, 500) |
| latency_ms | float | Request processing time in milliseconds |
| card_name | string | Parsed card name |
| card_types | list[string] | Parsed card types |
| card_mana_cost | string \| null | Raw mana cost string or null |
| predicted_price_eur | float | The prediction result |
| model_version | string | Model version used |

### Error Log

| Field | Type | Notes |
|-------|------|-------|
| event | string | Always `"evaluate_request"` |
| timestamp | string | ISO 8601 with microseconds |
| status_code | int | HTTP status code (400 or 500) |
| latency_ms | float | Request processing time in milliseconds |
| error | string | Error description |

## CLI Output Format

The `eval` subcommand writes to stdout (human-readable). Format:

```
Predicted price: €2.35
Model version:   20260301-143000
```

Error output goes to stderr:
```
Error: File not found: path/to/card.txt
Error: Could not connect to prediction service at http://localhost:8000/api/v1/evaluate
Error: Prediction service returned error: No Name field found in card script
```

## State Transitions

No state machines in this feature. The request/response cycle is stateless:

```
Forge script text → [parse] → Card entity → [predict] → PriceEstimate → JSON response
```

The CLI adds:
```
File path → [read file] → Forge script text → [HTTP POST] → JSON response → [display] → stdout
```
