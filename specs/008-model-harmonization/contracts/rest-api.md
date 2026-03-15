# REST API Contract: 008 Model Harmonization

## POST /api/v1/predict

### Request

**Content-Type**: `text/plain`

**Body**: Multiline string in converted card text format.

```text
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
```

### Response (200 OK)

```json
{
  "sklearn": {
    "predicted_price_eur": 2.45,
    "model_version": "20260315-103000"
  },
  "transformer": {
    "predicted_price_eur": 2.50,
    "model_version": "transformer-v1"
  }
}
```

- `transformer` is `null` if no transformer model is loaded.
- `sklearn` is always present (required model).

### Response (400 Bad Request)

```json
{
  "detail": "Failed to parse converted card text: missing required field 'name'"
}
```

### Change from previous version

- **Before**: Body was a raw Forge card script (uppercase `Name:`, `ManaCost:`, `Types:`, `A:` ability lines).
- **After**: Body is a converted card text (lowercase `name:`, `mana cost:`, `types:`, `spell[N]:` ability lines).
- Response format is **unchanged**.
