# Contract: POST /api/v1/evaluate

**Feature**: 007-transformer-model-arch
**Breaking change**: Yes — response field names change.

## Request

**Method**: POST
**Path**: `/api/v1/evaluate`
**Content-Type**: `text/plain`
**Body**: Forge card script as plain text.

```
Name:Lightning Bolt
ManaCost:R
Types:Instant
A:SP$ DealDamage | ValidTgts$ Creature,Player | TgtPrompt$ Select any target | NumDmg$ 3 | SpellDescription$ CARDNAME deals 3 damage to any target.
Oracle:Lightning Bolt deals 3 damage to any target.
```

## Response — Success (200)

**Content-Type**: `application/json`

### Before (features 001–006)

```json
{
  "predicted_price_eur": 2.35,
  "model_version": "20260301-143601"
}
```

### After (feature 007)

```json
{
  "sklearn": {
    "predicted_price_eur": 2.35,
    "model_version": "20260301-143601"
  },
  "transformer": {
    "predicted_price_eur": 2.18,
    "model_version": "transformer-v1"
  }
}
```

| Field | Type | Nullable | Description |
|---|---|---|---|
| `sklearn` | object | No | Sklearn model results. Always present. |
| `sklearn.predicted_price_eur` | float | No | Sklearn model prediction (EUR). |
| `sklearn.model_version` | string | No | Sklearn model version identifier. |
| `transformer` | object \| null | Yes | Transformer model results. `null` if model artifact is unavailable. |
| `transformer.predicted_price_eur` | float | No (within object) | Transformer model prediction (EUR). |
| `transformer.model_version` | string | No (within object) | Transformer model version identifier. |

### Graceful Degradation

If `models/transformer/model.pt` does not exist or fails to load:

```json
{
  "sklearn": {
    "predicted_price_eur": 2.35,
    "model_version": "20260301-143601"
  },
  "transformer": null
}
```

The endpoint never fails due to a missing transformer model — the sklearn prediction is always returned.

## Response — Client Error (400)

Returned when the card script cannot be parsed.

```json
{
  "error": "Failed to parse card script: <reason>"
}
```

## Response — Server Error (500)

Returned when prediction fails unexpectedly (both models).

```json
{
  "error": "Prediction failed: <reason>"
}
```

## Structured Log Entry

Each request emits a single structured JSON log line:

```json
{
  "event": "evaluate_request",
  "timestamp": "2026-03-14T10:30:00.000000+00:00",
  "status_code": 200,
  "latency_ms": 45.123,
  "card_name": "Lightning Bolt",
  "card_types": ["Instant"],
  "card_mana_cost": "R",
  "sklearn_predicted_price_eur": 2.35,
  "sklearn_model_version": "20260301-143601",
  "transformer_predicted_price_eur": 2.18,
  "transformer_model_version": "transformer-v1"
}
```

## CLI Consumer: `eval` Subcommand

The `eval` CLI subcommand displays both predictions:

```
sklearn:
  Predicted price: €2.35
  Model version:   20260301-143601
transformer:
  Predicted price: €2.18
  Model version:   transformer-v1
```

If transformer is unavailable:

```
sklearn:
  Predicted price: €2.35
  Model version:   20260301-143601
transformer:
  not available
```
