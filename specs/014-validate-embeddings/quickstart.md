# Quickstart: Validate Card Embeddings

## Prerequisites

- Card embeddings (`.npz`) and card text files (`.txt`) generated via `python -m sealed encode-cards`
- Python environment with scikit-learn installed

## Basic Usage

```bash
cd src
python -m sealed validate-embeddings --cards-path output/cardsfolder/
```

This trains 21 linear probes on top of frozen embeddings and reports pass/fail for each.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All probes passed — embeddings are good for Stage 2 training |
| 1 | One or more probes failed — review output to identify weak features |
| 2 | Input error (missing directory, insufficient cards) |

## Custom Thresholds

Override default accuracy (classification) or R² (regression) thresholds:

```bash
python -m sealed validate-embeddings \
    --cards-path output/cardsfolder/ \
    --threshold-accuracy 0.99 \
    --threshold-r2 0.90
```

- `--threshold-accuracy` applies to all classification probes (is-land, card color, mana produced). Default: 0.95. The is-land probe uses `max(threshold, 0.99)`.
- `--threshold-r2` applies to all regression probes (pip counts, mana value). Default: 0.85. The mana value probe uses `max(threshold, 0.90)`.

## Interpreting Results

**PASS**: All probes meet their thresholds. Proceed to Stage 2 training.

**FAIL**: Check which probes failed. Common causes:
- Low accuracy on "is land" → encoder doesn't distinguish card types
- Low R² on pip counts → encoder doesn't capture mana cost structure
- Low accuracy on mana produced → encoder doesn't understand land abilities

Action: retrain the card encoder (spec 010) with improved tokenization or longer training before attempting Stage 2.

## What It Tests

The validation checks whether frozen embeddings encode the 5 feature categories that the Stage 2 mana scorer depends on:

1. **Is land** — Can a linear model tell lands from non-lands?
2. **Card color** — Can it tell which colors a card requires?
3. **Pip counts** — Can it predict how many pips of each color?
4. **Mana value** — Can it predict total mana cost?
5. **Mana produced** — Can it tell which colors a land produces?

Each probe is a simple linear model (logistic or linear regression) trained with 5-fold cross-validation. If a linear model can decode the feature from the embedding, it means the encoder has learned to represent that information.
