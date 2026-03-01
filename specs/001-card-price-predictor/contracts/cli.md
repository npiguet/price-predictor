# CLI Contract: Card Price Predictor

**Date**: 2026-03-01 (revised)
**Feature**: 001-card-price-predictor

## Overview

The system exposes its functionality as a Python CLI tool. All three
user stories (predict, train, evaluate) are accessible as subcommands.
All prices are in EUR (Cardmarket).

## Output Channels

- **stdout**: Structured JSON result only. Clean for piping/redirection.
- **stderr**: Human-readable progress messages during long-running
  operations (train, evaluate). Error messages for all commands.

This separation ensures `python -m price_predictor train ... > result.json`
captures clean JSON while progress is visible on the console.

## Commands

### `predict` — Predict price for a card

**Usage**:
```
python -m price_predictor predict [OPTIONS]
```

**Input options** (all optional except at least one card type):
```
--mana-cost TEXT       Mana cost in Forge format, e.g., "2 W W"
--types TEXT           Card types, comma-separated (e.g., "Creature,Enchantment")
--supertypes TEXT      Supertypes, comma-separated (e.g., "Legendary")
--subtypes TEXT        Subtypes, comma-separated (e.g., "Human,Wizard")
--oracle-text TEXT     Oracle/rules text
--keywords TEXT        Keywords, comma-separated (e.g., "Flying,Vigilance")
--power TEXT           Power value (number, "*", or "X")
--toughness TEXT       Toughness value (number, "*", or "X")
--loyalty TEXT         Planeswalker loyalty
--colors TEXT          Colors, comma-separated (e.g., "W,U") — override if
                       different from mana cost colors
--model-path PATH     Path to trained model (default: models/latest.joblib)
```

**Output** (stdout, JSON):
```json
{
  "predicted_price_eur": 12.45,
  "model_version": "20260301-143000"
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
--forge-cards-path PATH   Path to Forge cardsfolder directory
                          (default: ../forge/forge-gui/res/cardsfolder)
--prices-path PATH        Path to AllPricesToday.json
                          (default: resources/AllPricesToday.json)
--printings-path PATH     Path to AllPrintings.json
                          (default: resources/AllPrintings.json)
--output-path PATH        Path to save trained model (default: models/)
--test-split FLOAT        Fraction for test set (default: 0.2)
--random-seed INT         Random seed for reproducibility (default: 42)
```

**Output** (stdout, JSON):
```json
{
  "model_version": "20260301-143000",
  "model_path": "models/20260301-143000.joblib",
  "cards_used": 24531,
  "cards_skipped": 7569,
  "skipped_reasons": {
    "no_price": 6200,
    "parse_error": 169,
    "no_printings_match": 1200
  },
  "price_range_eur": {"min": 0.02, "max": 425.00}
}
```

**Progress messages** (stderr):
```
Parsing Forge card scripts...
Parsed 32145 cards (169 parse errors)
Loading AllPrintings.json — building name-to-UUID mapping...
Built name-to-UUID mapping (28500 card names)
Loading AllPricesToday.json — building price map...
Loaded price data (22000 cards with prices)
Matched 24531 cards to prices, skipped 7614
Feature engineering complete (576 features)
Training model...
Model training complete
Model saved: 20260301-143000
```

**Exit codes**:
- 0: Success
- 1: Input path not found or unreadable
- 2: Training failed (insufficient valid data)

---

### `evaluate` — Evaluate model accuracy

**Usage**:
```
python -m price_predictor evaluate [OPTIONS]
```

**Input options**:
```
--model-path PATH         Path to trained model
                          (default: models/latest.joblib)
--forge-cards-path PATH   Path to Forge cardsfolder directory
                          (default: ../forge/forge-gui/res/cardsfolder)
--prices-path PATH        Path to AllPricesToday.json
                          (default: resources/AllPricesToday.json)
--printings-path PATH     Path to AllPrintings.json
                          (default: resources/AllPrintings.json)
--test-split FLOAT        Fraction for test set (default: 0.2)
--random-seed INT         Random seed (default: 42) — must match
                          training seed
--output-csv PATH         Optional: save per-card results to CSV
```

**Output** (stdout, JSON):
```json
{
  "model_version": "20260301-143000",
  "mean_absolute_error_eur": 2.87,
  "median_percentage_error": 38.5,
  "top_20_overlap": 0.64,
  "sample_count": 6132
}
```

**Progress messages** (stderr): Same stage messages as `train`
(parsing, mapping, pricing, matching), plus evaluation-specific:
```
Loading model from models/latest.joblib...
Computing predictions on test set (6132 cards)...
Evaluation complete
```

**Exit codes**:
- 0: Success
- 1: Input file or model not found
- 2: Evaluation failed
