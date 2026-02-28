# Quickstart: Card Price Predictor

## Prerequisites

- Python 3.11+
- pip

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
# Download AllIdentifiers.json from https://mtgjson.com/api/v5/AllIdentifiers.json
# Place both in resources/
```

## Train the model

```bash
python -m price_predictor train \
  --prices-path resources/AllPricesToday.json \
  --cards-path resources/AllIdentifiers.json
```

Expected output: a trained model saved in `models/` and a summary
showing how many cards were used.

## Predict a card price

```bash
python -m price_predictor predict \
  --types "Creature" \
  --supertypes "Legendary" \
  --subtypes "Human,Wizard" \
  --mana-cost "{1}{U}{R}" \
  --power "2" \
  --toughness "2" \
  --keywords "Prowess" \
  --oracle-text "Whenever you cast a noncreature spell, draw a card." \
  --rarity "mythic" \
  --colors "U,R"
```

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
- [ ] `predict` command returns a JSON price estimate
- [ ] `evaluate` reports median percentage error <= 50%
- [ ] `pytest` passes all tests
- [ ] Predictions are reproducible (same input → same output)
