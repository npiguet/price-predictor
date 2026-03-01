# API Contract: POST /api/v1/evaluate

**Feature**: 005-card-eval-endpoints
**Version**: 1.0 (established in feature 002, documented here)
**Date**: 2026-03-01

## Endpoint

```
POST /api/v1/evaluate
```

No authentication required. Designed for local/internal use.

## Request

| Property | Value |
|----------|-------|
| Method | POST |
| Path | /api/v1/evaluate |
| Content-Type | text/plain |
| Encoding | UTF-8 |
| Body | Raw Forge card script text |

### Request Body

The body is a Forge card script in plain text. The format uses colon-separated key-value pairs, one per line.

**Minimum viable input** (must include `Name` and at least one recognized `Types`):

```
Name:Test Card
Types:Instant
```

**Complete input example**:

```
Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target.
```

**Complex input example** (multi-face card — only front face evaluated):

```
Name:Delver of Secrets
ManaCost:U
Types:Creature Human Wizard
PT:1/1
K:Prowess
AlternateMode:DoubleFaced
Oracle:At the beginning of your upkeep, look at the top card of your library.

ALTERNATE

Name:Insectile Aberration
ManaCost:no cost
Types:Creature Human Insect
PT:3/2
K:Flying
Oracle:Flying
```

### Recognized Fields

| Field | Required | Description |
|-------|----------|-------------|
| Name | Yes | Card name |
| Types | Yes | Space-separated supertypes, card types, and subtypes |
| ManaCost | No | Forge mana cost format (e.g., `2 U U`, `R`, `1 G`) |
| PT | No | Power/toughness as `P/T` (e.g., `2/2`, `*/4`) |
| Oracle | No | Rules text (`\n` for line breaks) |
| Loyalty | No | Planeswalker starting loyalty |
| K: | No | Keyword ability lines (one per keyword) |
| A:/T:/S: | No | Ability lines (counted for ability_count feature) |
| AlternateMode | No | Layout hint (DoubleFaced, Split, Adventure, Modal, Flip) |

All other fields are ignored.

## Response

### Success (200 OK)

```json
{
  "predicted_price_eur": 2.35,
  "model_version": "20260301-143000"
}
```

| Field | Type | Description |
|-------|------|-------------|
| predicted_price_eur | number | Predicted EUR price, >= 0, 2 decimal places |
| model_version | string | Identifier of the model that produced the estimate |

Content-Type: `application/json`

### Client Error (400 Bad Request)

Returned when the input cannot be parsed into a valid card.

```json
{
  "error": "Failed to parse card script: No Name field found in card script"
}
```

| Field | Type | Description |
|-------|------|-------------|
| error | string | Human-readable error description |

Common error causes:
- Empty request body
- Missing `Name` field
- Missing or unrecognized `Types` field
- Completely unparseable content

### Server Error (500 Internal Server Error)

Returned when parsing succeeds but prediction fails unexpectedly.

```json
{
  "error": "Prediction failed: <details>"
}
```

## Behavioral Notes

1. **Partial input**: Missing optional fields are treated as absent. Prediction proceeds with available attributes.
2. **Multi-face cards**: Only the front face (before `ALTERNATE` marker) is evaluated.
3. **Extra fields**: Fields not listed above are silently ignored.
4. **Made-up cards**: Hypothetical card descriptions are supported — the model predicts based on attributes.
5. **Encoding**: UTF-8 assumed. Other encodings may produce garbled text and parsing errors.
6. **Single card**: One card per request. If multiple card definitions appear, only the first is used.

## Consumers

- **CLI eval subcommand**: `python -m price_predictor eval card.txt` (this feature)
- **Java connector**: `forge-connector/` PricePredictorClient (feature 002)
- **Direct HTTP clients**: curl, Postman, etc.
