# CLI Contract: Card Price Predictor

**Date**: 2026-02-26
**Feature**: 001-card-price-predictor

## Overview

The system exposes its functionality as a Python CLI tool. All three
user stories (predict, train, evaluate) are accessible as subcommands.

## Commands

### `predict` — Predict price for a card

**Usage**:
```
python -m price_predictor predict [OPTIONS]
```

**Input options** (all optional except at least one card type):
```
--mana-cost TEXT       Mana cost, e.g., "{2}{W}{W}"
--mana-value FLOAT    Converted mana cost
--types TEXT           Card types, comma-separated (e.g., "Creature,Enchantment")
--supertypes TEXT      Supertypes, comma-separated (e.g., "Legendary")
--subtypes TEXT        Subtypes, comma-separated (e.g., "Human,Wizard")
--oracle-text TEXT     Oracle/rules text
--keywords TEXT        Keywords, comma-separated (e.g., "Flying,Vigilance")
--power TEXT           Power value (number, "*", or "X")
--toughness TEXT       Toughness value (number, "*", or "X")
--loyalty TEXT         Planeswalker loyalty
--rarity TEXT          Rarity: common, uncommon, rare, mythic
--colors TEXT          Colors, comma-separated (e.g., "W,U")
--model-path PATH     Path to trained model (default: models/latest.joblib)
```

**Output** (stdout, JSON):
```json
{
  "predicted_price_usd": 12.45,
  "model_version": "20260226-143000"
}
```

**Exit codes**:
- 0: Success
- 1: Invalid input (validation error)
- 2: Model not found or not loadable

**Errors** (stderr):
```
Error: No card types provided. At least one --types value is required.
Error: Model file not found at models/latest.joblib
```

---

### `train` — Train model on card data

**Usage**:
```
python -m price_predictor train [OPTIONS]
```

**Input options**:
```
--prices-path PATH     Path to AllPricesToday.json (default: resources/AllPricesToday.json)
--cards-path PATH      Path to AllIdentifiers.json (default: resources/AllIdentifiers.json)
--output-path PATH     Path to save trained model (default: models/)
--test-split FLOAT     Fraction for test set (default: 0.2)
--random-seed INT      Random seed for reproducibility (default: 42)
```

**Output** (stdout, JSON):
```json
{
  "model_version": "20260226-143000",
  "model_path": "models/20260226-143000.joblib",
  "cards_used": 45231,
  "cards_skipped": 1203,
  "skipped_reasons": {
    "no_price": 890,
    "invalid_data": 213,
    "filtered_out": 100
  },
  "price_range": {"min": 0.01, "max": 425.00}
}
```

**Exit codes**:
- 0: Success
- 1: Input file not found or unreadable
- 2: Training failed (insufficient valid data)

---

### `evaluate` — Evaluate model accuracy

**Usage**:
```
python -m price_predictor evaluate [OPTIONS]
```

**Input options**:
```
--model-path PATH      Path to trained model (default: models/latest.joblib)
--prices-path PATH     Path to AllPricesToday.json (default: resources/AllPricesToday.json)
--cards-path PATH      Path to AllIdentifiers.json (default: resources/AllIdentifiers.json)
--test-split FLOAT     Fraction for test set (default: 0.2)
--random-seed INT      Random seed (default: 42) — must match training seed
--output-csv PATH      Optional: save per-card results to CSV
```

**Output** (stdout, JSON):
```json
{
  "model_version": "20260226-143000",
  "mean_absolute_error_usd": 3.42,
  "median_percentage_error": 38.5,
  "top_20_overlap": 0.64,
  "sample_count": 9046
}
```

**Exit codes**:
- 0: Success
- 1: Input file or model not found
- 2: Evaluation failed
