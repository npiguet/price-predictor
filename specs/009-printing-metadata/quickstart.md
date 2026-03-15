# Quickstart: 009 Printing Data Fields

## Prerequisites

- Python 3.14+ with venv activated
- `pip install -e .` (or dependencies from pyproject.toml)
- `resources/AllPrintings.json` and `resources/AllPricesToday.json` present
- Converted card texts in `output/` (from `price_predictor convert`)

## Retrain models with enriched data

Both models must be retrained after this feature to learn from the new metadata fields.

```bash
cd src

# Train sklearn model (enriches card text with metadata during training)
python -m price_predictor train sklearn

# Train transformer model (enriches card text with metadata during training)
python -m price_predictor train transformer
```

## Evaluate retrained models

```bash
cd src

# Evaluate sklearn
python -m price_predictor evaluate sklearn

# Evaluate transformer
python -m price_predictor evaluate transformer
```

## Predict (CLI)

```bash
cd src

# Known card — metadata auto-filled from AllPrintings
python -m price_predictor predict sklearn -f ../output/l/lightning_bolt.txt

# Unknown card with inline metadata
python -m price_predictor predict sklearn -c "name: custom creature
mana cost: {2}{G}
types: creature beast
power toughness: 4/4
reserved: false
rarity: rare
printings: 1
set: ukn
legalities: standard, pioneer, modern, legacy, vintage, commander"
```

## Predict (API)

```bash
# Start the server (now loads metadata for auto-fill)
cd src
python -m price_predictor serve

# Known card — no metadata needed (auto-filled)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: text/plain" \
  -d "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target."

# Unknown card with metadata
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: text/plain" \
  -d "name: custom mythic
mana cost: {3}{B}{B}
types: creature demon
power toughness: 6/6
reserved: false
rarity: mythic
printings: 1
set: ukn
legalities: standard, modern, commander"

# "What if" — known card with overridden rarity
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: text/plain" \
  -d "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
rarity: mythic"
```

## Run tests

```bash
cd src
pytest                        # All tests
pytest -m "not integration"   # Fast unit tests only
ruff check .                  # Lint
```
