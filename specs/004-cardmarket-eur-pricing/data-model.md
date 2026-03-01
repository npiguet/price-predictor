# Data Model: CardMarket EUR Pricing

**Feature**: 004-cardmarket-eur-pricing | **Date**: 2026-03-01

## Entity Impact Analysis

This feature formalizes existing behavior. **No entity changes are required.**

### Existing Entities (unchanged)

| Entity | Location | EUR Fields | Impact |
|--------|----------|------------|--------|
| TrainingExample | `domain/entities.py` | `actual_price_eur` | None — already EUR |
| Card | `domain/entities.py` | (no price field) | None |
| PredictionResult | `application/predict.py` | `predicted_price_eur` | None — already EUR |
| EvaluationMetrics | `application/evaluate.py` | `mean_absolute_error_eur` | None — already EUR |

### Price Extraction Path (existing, formalized)

```text
AllPricesToday.json
  └── data[uuid]
        └── paper
              └── cardmarket
                    └── retail
                          ├── normal[latest_date] → price (EUR)
                          └── foil[latest_date] → price (EUR)
```

This path is already implemented in `mtgjson_loader.py` lines 65–76 (`get_cheapest_price`) and 101–112 (`build_price_map`). Feature 004 formalizes this as the authoritative price extraction path.

### Excluded Card (conceptual, not a persisted entity)

Cards present in `name_to_uuids` but absent from the price map result are "excluded cards." They are not tracked as an entity — only their count is logged (FR-008).

| Attribute | Type | Source |
|-----------|------|--------|
| count | int | `len(name_to_uuids) - len(result)` |

## Behavioral Changes

| Function | Current Behavior | New Behavior | FR |
|----------|-----------------|--------------|-----|
| `build_price_map()` | Logs cards with prices only | Additionally logs exclusion count | FR-008 |

All other behaviors (FR-001–FR-007, FR-009, FR-010) are already implemented and unchanged.
