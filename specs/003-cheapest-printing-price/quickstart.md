# Quickstart: Cheapest Printing Price for Training

**Feature**: 003-cheapest-printing-price

## What Changed

When a Magic card has multiple printings (different sets, foil vs. non-foil, different rarities, promos), each printing may have a different market price. The system now:

1. **Selects the cheapest price** across all English paper printings as the training label for that card.
2. **Applies a €0.01 price floor** — if the cheapest price is below €0.01 (including €0.00), it is clamped to €0.01 to prevent log-domain errors during model training.
3. **Logs transparency information** about price selection to stderr during training.

## Usage

No new commands or flags. The behavior is automatic during training:

```bash
python -m price_predictor train \
  --forge-cards /path/to/forge/cardsfolder \
  --prices /path/to/AllPricesToday.json \
  --printings /path/to/AllPrintings.json \
  --output /path/to/models
```

## New Log Output

During training, you will see additional INFO-level messages on stderr:

```
Loading AllPricesToday.json — building price map...
  Lightning Bolt: selected €1.50 from 4 prices
  Sol Ring: selected €0.80 from 3 prices
  ...
Price selection summary: 8432 of 14210 cards had multiple price points
Loaded price data (14210 cards with prices)
```

- **Per-card lines** appear only for cards where multiple prices were compared.
- **Summary line** shows the total count of cards that required price selection.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Single printing | Price used directly, no selection log |
| All printings same price | Cheapest is that price, still logged as multi-price |
| Cheapest price is €0.00 | Clamped to €0.01 |
| Some printings have no price | Ignored; cheapest among valid prices used |
| All printings have no price | Card excluded from training set |
