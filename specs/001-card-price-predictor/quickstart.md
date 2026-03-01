# Quickstart: Card Price Predictor

## Prerequisites

- Python 3.14+
- pip
- MTG Forge repository checkout at `../forge` (sibling directory)

## Setup

```bash
# 1. Clone and enter the project
cd price-predictor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Download MTGJSON data files (one-time)
mkdir -p resources
# Download AllPricesToday.json from https://mtgjson.com/api/v5/AllPricesToday.json
# Download AllPrintings.json from https://mtgjson.com/api/v5/AllPrintings.json
# Place both in resources/

# 5. Ensure Forge checkout exists
ls ../forge/forge-gui/res/cardsfolder/
# Should show alphabetical subdirectories (a/ through z/)
```

## Train the model

```bash
python -m price_predictor train \
  --forge-cards-path ../forge/forge-gui/res/cardsfolder \
  --prices-path resources/AllPricesToday.json \
  --printings-path resources/AllPrintings.json
```

Expected output: a trained model saved in `models/` and a JSON summary
showing how many cards were used, skipped, and the price range.

## Predict a card price

```bash
python -m price_predictor predict \
  --types "Creature" \
  --supertypes "Legendary" \
  --subtypes "Human,Wizard" \
  --mana-cost "1 U R" \
  --power "2" \
  --toughness "2" \
  --keywords "Prowess" \
  --oracle-text "Whenever you cast a noncreature spell, draw a card."
```

Returns a JSON object with `predicted_price_eur` and `model_version`.

## Evaluate model accuracy

```bash
python -m price_predictor evaluate \
  --model-path models/latest.joblib
```

## Run tests

```bash
pytest
```

Fast unit tests run by default. For integration tests:

```bash
pytest tests/integration/
```

## Validation checklist

- [ ] `train` command completes without errors
- [ ] `predict` command returns a JSON price estimate in EUR
- [ ] `evaluate` reports median percentage error <= 50%
- [ ] `pytest` passes all tests
- [ ] Predictions are reproducible (same input → same output)
