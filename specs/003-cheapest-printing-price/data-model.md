# Data Model: Cheapest Printing Price for Training

**Feature**: 003-cheapest-printing-price
**Date**: 2026-03-01

## Overview

This feature does not introduce new persistent entities or data structures. It refines the behavior of the existing price-loading infrastructure. The data model impact is limited to:

1. A behavioral change to how prices are collected and normalized (floor clamping)
2. Transient per-card statistics used only for logging during training data preparation

## Existing Entities (Unchanged)

### Card
- **Location**: `src/price_predictor/domain/entities.py`
- **Impact**: None. Card represents oracle-level attributes and has no price field.

### TrainingExample
- **Location**: `src/price_predictor/domain/entities.py`
- **Fields**: `card: Card`, `actual_price_eur: float`
- **Validation**: `actual_price_eur > 0`
- **Impact**: None. The €0.01 floor guarantees `actual_price_eur >= 0.01`, which satisfies the existing `> 0` validation. Note: this entity is defined but not currently instantiated by the training pipeline.

### Price Map (runtime data structure)
- **Location**: `src/price_predictor/infrastructure/mtgjson_loader.py`
- **Type**: `dict[str, float]` (card name → cheapest EUR price)
- **Impact**: Values may now include prices clamped to €0.01 (previously, prices at €0.00 were excluded entirely). The type and structure remain unchanged.

## Behavioral Changes

### Price Collection Rule

| Aspect | Before (feature 001) | After (feature 003) |
|--------|----------------------|---------------------|
| Price filter | `price and price > 0` (excludes None, 0.0) | `price is not None and price >= 0` (excludes only None) |
| Floor | None | `max(selected_price, 0.01)` applied after `min()` |
| Zero-price printings | Excluded from comparison | Included in comparison, then clamped to €0.01 |
| Negative prices | Excluded (by `> 0`) | Excluded (by `>= 0`) |

### Logging Data (transient, not persisted)

During `build_price_map()` execution, the following transient data is tracked per card:

| Field | Type | Description |
|-------|------|-------------|
| `card_name` | `str` | The card being processed |
| `all_prices` | `list[float]` | All valid prices found across printings/finishes |
| `selected_price` | `float` | `max(min(all_prices), 0.01)` |
| `alternative_count` | `int` | `len(all_prices) - 1` |

This data is used exclusively for FR-007/FR-008 logging and is not returned or persisted.

### Summary Statistics (transient, not persisted)

After processing all cards, the following summary is logged:

| Field | Type | Description |
|-------|------|-------------|
| `multi_printing_count` | `int` | Cards where `len(all_prices) > 1` |
| `total_cards_with_prices` | `int` | Cards in the resulting price map |

## Validation Rules

| Rule | Location | Description |
|------|----------|-------------|
| Price non-null | `mtgjson_loader.py` | `price is not None` — null/missing prices are ignored |
| Price non-negative | `mtgjson_loader.py` | `price >= 0` — negative prices (data errors) are ignored |
| Price floor | `mtgjson_loader.py` | `max(price, 0.01)` — final price is at least €0.01 |
| Training price positive | `entities.py` | `actual_price_eur > 0` — unchanged, satisfied by floor |

## State Transitions

Not applicable. This feature modifies a stateless data transformation step (price loading). There are no entity state machines or lifecycle transitions.
