# Quickstart: CardMarket EUR Pricing

**Feature**: 004-cardmarket-eur-pricing | **Date**: 2026-03-01

## What This Feature Does

This feature formalizes that the price predictor uses **CardMarket EUR prices** from MTGJSON as the sole price source for training and prediction. It also adds transparency logging for cards excluded due to missing CardMarket prices.

## New Log Output (FR-008)

During training data preparation, `build_price_map()` now reports excluded cards:

```text
INFO  Loading AllPricesToday.json — building price map...
INFO    Grizzly Bears: selected €0.10 from 3 prices
INFO    Lightning Bolt: selected €1.50 from 3 prices
INFO    ...
INFO  Price selection summary: 7 of 14 cards had multiple price points
INFO  Price exclusion: 2 of 16 cards excluded (no CardMarket price available)
INFO  Loaded price data (14 cards with prices)
```

The new line is `Price exclusion: X of Y cards excluded (no CardMarket price available)` where:
- **X** = number of cards in `name_to_uuids` that had no valid CardMarket price
- **Y** = total number of cards in `name_to_uuids` (before exclusion)

## Existing Behavior (formalized, not changed)

The following behaviors already exist and are now explicitly documented as requirements:

1. **Price source**: CardMarket EUR retail prices from `paper → cardmarket → retail` in MTGJSON
2. **Currency**: EUR throughout — training labels, predictions, evaluation metrics
3. **Vendor exclusivity**: TCGPlayer, CardKingdom, and other vendors are ignored
4. **Date selection**: Most recent available snapshot used
5. **Price type**: Retail (not buylist)
6. **Zero prices**: Included in training (clamped to €0.01 by feature 003's floor)
7. **Missing prices**: Cards without any CardMarket price are excluded from training

## Running Training

No changes to the training command:

```bash
cd src
python -m price_predictor.application.train \
  --allprintings resources/AllPrintings.json \
  --allprices resources/AllPricesToday.json \
  --forge-dir resources/cardsfolder \
  --output models/model.joblib
```

The exclusion count log appears automatically on stderr during execution.
