# Quickstart: MTG Custom Tokenizer

## Prerequisites

- Converted card corpus in `./output/` (run `python -m price_predictor convert` first)
- Existing project setup (`.venv` activated, dependencies installed)

## Step 1: Build the vocabulary

```bash
python -m price_predictor vocabulary
```

Expected output:
```json
{
  "vocab_path": "models/transformer/vocab.txt",
  "vocab_size": 5847,
  "domain_token_count": 312,
  "freq_threshold_token_count": 5535,
  "coverage_pct": 98.1,
  "unk_pct": 1.9
}
```

The file `models/transformer/vocab.txt` is created. Inspect it:
```bash
head -20 models/transformer/vocab.txt
# Should show: [PAD], [UNK], cardname, then domain terms
wc -l models/transformer/vocab.txt
# Should be < 10000
```

Verify a known keyword is present:
```bash
grep "^first_strike$" models/transformer/vocab.txt   # multi-word keyword
grep "^flying$" models/transformer/vocab.txt          # single keyword
grep "^{w}$" models/transformer/vocab.txt             # mana symbol
```

## Step 2: Train the transformer

`vocab.txt` is picked up automatically from the default path:

```bash
python -m price_predictor train transformer --epochs 20 --batch-size 64
```

The model is saved to `models/transformer/`. Verify `vocab_size` in the saved config is not 30522:
```python
import torch
ckpt = torch.load("models/transformer/latest.pt", map_location="cpu", weights_only=True)
print(ckpt["config"]["vocab_size"])  # Should be ~5000-8000, not 30522
```

## Step 3: Evaluate

```bash
python -m price_predictor evaluate transformer
```

## Step 4: Predict

```bash
python -m price_predictor predict transformer --card "name: lightning bolt
mana cost: {r}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
reserved: false
rarity: uncommon
printings: 23
set: 2xm
legalities: commander, legacy, modern, pauper, vintage, penny, oathbreaker"
```

## Custom vocab path

If vocab.txt lives somewhere other than the default:

```bash
python -m price_predictor vocabulary --output-dir /custom/path/
python -m price_predictor train transformer --vocab-path /custom/path/vocab.txt
python -m price_predictor evaluate transformer --vocab-path /custom/path/vocab.txt
python -m price_predictor predict transformer --vocab-path /custom/path/vocab.txt
python -m price_predictor serve --vocab-path /custom/path/vocab.txt
```

## Troubleshooting

**"Vocabulary file not found"**: Run `python -m price_predictor vocabulary` first.

**vocab_size still 30522 after training**: Old `latest.pt` was not overwritten. Check that training completed and wrote a new checkpoint.

**Coverage below 95%**: Try lowering `--freq-threshold` to 3. This will increase vocab size but improve coverage.
