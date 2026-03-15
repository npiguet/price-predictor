# Quickstart: 008 Model Harmonization

## Prerequisites

- Converted card text files in `./output/` (run `convert` first)
- MTGJSON data files: `resources/AllPricesToday.json`, `resources/AllPrintings.json`

## Train a model

```bash
# Train sklearn model
python -m price_predictor train sklearn

# Train transformer model
python -m price_predictor train transformer --epochs 20 --batch-size 64
```

Models are saved to `./models/sklearn/` and `./models/transformer/` respectively.

## Predict a card price

```bash
# From a file
python -m price_predictor predict sklearn --file output/l/lightning_bolt.txt

# From inline text
python -m price_predictor predict transformer --card "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target."
```

## Evaluate a model

```bash
python -m price_predictor evaluate sklearn
python -m price_predictor evaluate transformer
```

## REST API

```bash
# Start server
python -m price_predictor serve

# Predict via API (now accepts converted card text format)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: text/plain" \
  -d "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target."
```
