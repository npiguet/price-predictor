# REST API Contract: Prediction Service

**Date**: 2026-03-01
**Feature**: 002-forge-api-integration
**Shared with**: 005-card-eval-endpoints

## Overview

The prediction service exposes a single REST endpoint for card price
evaluation. This contract is shared between features 002 (service
architecture + connector) and 005 (endpoint + CLI tool). Both features
implement against the same endpoint.

## Endpoint

### `POST /api/v1/evaluate` — Evaluate a card's price

**Request**:
```
POST /api/v1/evaluate HTTP/1.1
Host: localhost:8000
Content-Type: text/plain

Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target.
```

**Response (200 OK)**:
```json
{
  "predicted_price_eur": 2.35,
  "model_version": "20260301-143000"
}
```

**Response (400 Bad Request)**:
```json
{
  "error": "Failed to parse card script: no Types line found"
}
```

**Response (500 Internal Server Error)**:
```json
{
  "error": "Prediction failed: unexpected model error"
}
```

## Request Format

- **Method**: POST
- **Path**: `/api/v1/evaluate`
- **Content-Type**: `text/plain; charset=utf-8`
- **Body**: Raw Forge card script text (key:value lines)

Supported fields in the body:

| Line format | Example | Maps to |
|-------------|---------|---------|
| `Name:value` | `Name:Lightning Bolt` | Card name |
| `ManaCost:value` | `ManaCost:2 W W` | Mana cost (Forge format) |
| `Types:value` | `Types:Legendary Creature - Human Wizard` | Supertypes + types + subtypes |
| `Oracle:value` | `Oracle:Deals 3 damage...` | Oracle/rules text |
| `K:value` | `K:Flying` | Keyword ability (one per line) |
| `PT:value` | `PT:3/4` | Power/toughness |
| `Loyalty:value` | `Loyalty:3` | Planeswalker loyalty |

All fields are optional except at least one card type must be
derivable from the `Types:` line.

## Response Format

- **Content-Type**: `application/json`
- **Success body**: `{"predicted_price_eur": <float>, "model_version": "<string>"}`
- **Error body**: `{"error": "<string>"}`

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Prediction returned successfully |
| 400 | Invalid input (empty body, unparseable script, no card types) |
| 500 | Server-side error (model failure) |

## Versioning

The `/v1/` path segment versions the API. Breaking changes require
a new version path (e.g., `/v2/`). Non-breaking additions (new
response fields) are permitted within the same version.

## Server Configuration

- **Default host**: `0.0.0.0` (all interfaces)
- **Default port**: `8000`
- **Start command**: `python -m price_predictor serve`
- **Optional flags**: `--port`, `--host`, `--model-path`
