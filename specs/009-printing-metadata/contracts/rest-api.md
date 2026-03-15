# REST API Contract: 009 Printing Data Fields

## POST /api/v1/predict

### Request

**Content-Type**: `text/plain`

**Body**: Multiline string in converted card text format. The five printing data fields are optional — if absent, they are auto-filled (known cards) or defaulted (unknown cards).

#### Without metadata (backward-compatible — auto-filled or defaulted)

```text
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
```

#### With metadata (client-provided values used as-is)

```text
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
reserved: false
rarity: uncommon
printings: 23
set: 2xm
legalities: modern, legacy, vintage, pauper, commander, penny, oathbreaker
```

#### Partial metadata (mixed auto-fill and client-provided)

```text
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
rarity: mythic
```

In this case, `rarity: mythic` overrides the auto-filled value. The other four fields are auto-filled from AllPrintings (if known) or defaulted (if unknown).

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
- Response format is **unchanged** from 008.

### Response (400 Bad Request)

```json
{
  "error": "Failed to parse converted card text: missing required field 'name'"
}
```

### Metadata Auto-Fill Behavior

| Card found in AllPrintings? | Client provides field? | Value used |
|----------------------------|----------------------|------------|
| Yes | No | Auto-filled from cheapest printing (AllPricesToday.json prices) |
| Yes | Yes | Client-provided value (override) |
| No | No | Default value (see below) |
| No | Yes | Client-provided value |

### Default Values (FR-005)

| Field | Default |
|-------|---------|
| `reserved` | `false` |
| `rarity` | `rare` |
| `printings` | `1` |
| `set` | `ukn` |
| `legalities` | `standard, pioneer, modern, brawl, legacy, vintage, pauper, commander, penny, oathbreaker` |

### Change from previous version (008)

- **Before**: Card text body had no printing data fields. Prediction used only game text.
- **After**: Card text body may include 5 printing data fields (inline, same key:value format). Auto-fill/default logic ensures backward compatibility — no client changes required.
- Response format is **unchanged**.

### Server Startup Change

The `serve` command now requires access to `resources/AllPrintings.json` and `resources/AllPricesToday.json` at startup to build the metadata lookup for auto-fill. New CLI options:

```
price_predictor serve --printings-path resources/AllPrintings.json
                      --prices-path resources/AllPricesToday.json
```

Both default to `resources/AllPrintings.json` and `resources/AllPricesToday.json` respectively (matching existing train/evaluate defaults).
