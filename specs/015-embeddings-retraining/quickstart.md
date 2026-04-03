# Quickstart: Embeddings Retraining with Auxiliary Supervision

**Feature**: 015-embeddings-retraining  
**Date**: 2026-04-03

## Prerequisites

- Python 3.14+ with venv activated
- CUDA-capable GPU
- Vocabulary file built: `models/price-predictor/transformer/vocab.txt`
- Converted card text files in: `output/cardsfolder/`
- Price data: `resources/AllPrices.json`
- Printings data: `resources/AllPrintings.json`

## Training with Auxiliary Supervision

```bash
cd src

# Train from scratch with auxiliary heads (lambda=0.2 as starting point)
python -m price_predictor train transformer \
  --vocab-path models/price-predictor/transformer/vocab.txt \
  --aux-lambda 0.2

# If probes fail after training, increase lambda:
python -m price_predictor train transformer \
  --vocab-path models/price-predictor/transformer/vocab.txt \
  --aux-lambda 0.5

# If price accuracy degrades too much, decrease lambda:
python -m price_predictor train transformer \
  --vocab-path models/price-predictor/transformer/vocab.txt \
  --aux-lambda 0.1
```

The `--aux-lambda` flag controls the weight of auxiliary losses relative to the price
loss. When set to 0 (default), training proceeds without auxiliary supervision (legacy
behavior).

## Validation

After training, re-encode all cards and validate embeddings:

```bash
cd src

# Re-encode cards with the new model
python -m sealed encode-cards --clean

# Run all 20 embedding probes
python -m sealed validate-embeddings --cards-path output/cardsfolder/
```

All 20 probes must pass. If any fail, retrain with a higher lambda value.

## Expected Output

During training with `--aux-lambda > 0`, expect additional console output:

```
Computing auxiliary labels for ~30000 cards...
Computing class weights and target statistics...
  is_land: pos_weight=19.2 (1523 positive / 29277 negative)
  card_color_W: pos_weight=3.1 (7412 positive / 23388 negative)
  ...
  pip_count_W: mean=0.34, std=0.72
  ...
Epoch 1/100 — train_loss: 0.482, val_loss: 0.391, aux_loss: 0.832, 12.3s
Epoch 2/100 — train_loss: 0.445, val_loss: 0.372, aux_loss: 0.614, 12.1s
...
```

## Running Tests

```bash
cd src
pytest tests/unit/sealed/domain/test_embedding_probe.py -v
pytest tests/unit/infrastructure/test_transformer_model.py -v
pytest tests/unit/application/test_train_transformer.py -v
```
