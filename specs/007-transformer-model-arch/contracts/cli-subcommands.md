# Contract: CLI Subcommands

**Feature**: 007-transformer-model-arch

## New Subcommand: `train-transformer`

```
usage: price_predictor train-transformer [-h]
    [--output-dir PATH]
    [--prices-path PATH]
    [--printings-path PATH]
    [--forge-cards-path PATH]
    [--model-output PATH]
    [--batch-size INT]
    [--epochs INT]
    [--lr FLOAT]
    [--patience INT]
    [--random-seed INT]
```

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `output/` | Directory containing converted card script `.txt` files |
| `--prices-path` | `resources/AllPricesToday.json` | MTGJSON prices file |
| `--printings-path` | `resources/AllPrintings.json` | MTGJSON printings file |
| `--forge-cards-path` | `../forge/forge-gui/res/cardsfolder/` | Forge card scripts directory |
| `--model-output` | `models/transformer/` | Output directory for model artifact |
| `--batch-size` | `64` | Training batch size |
| `--epochs` | `20` | Maximum training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--random-seed` | `42` | Random seed for reproducibility |

**Exit codes**: 0 = success, 1 = input error, 2 = runtime error

**Stdout — per-epoch progress**:
```
Epoch 1/20 — train_loss: 0.312, val_loss: 0.285, 14.1s
Epoch 2/20 — train_loss: 0.198, val_loss: 0.192, 13.8s
...
```

**Stdout — final summary**:
```
Training complete. Best epoch: 8/20, val_loss: 0.142. Early stopping triggered at epoch 13.
Model saved to models/transformer/model.pt
```

**Stdout — auto-evaluation**:
```
Evaluation on validation set (4032 cards):
  Mean absolute error: €1.23
  Median percentage error: 42.5%
```

## New Subcommand: `evaluate-transformer`

```
usage: price_predictor evaluate-transformer [-h]
    [--model-path PATH]
    [--output-dir PATH]
    [--prices-path PATH]
    [--printings-path PATH]
    [--forge-cards-path PATH]
    [--random-seed INT]
    [--output-csv PATH]
```

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `models/transformer/` | Path to model artifact directory |
| `--output-dir` | `output/` | Directory containing converted card script `.txt` files |
| `--prices-path` | `resources/AllPricesToday.json` | MTGJSON prices file |
| `--printings-path` | `resources/AllPrintings.json` | MTGJSON printings file |
| `--forge-cards-path` | `../forge/forge-gui/res/cardsfolder/` | Forge card scripts directory |
| `--random-seed` | `42` | Random seed for reproducibility (must match training) |
| `--output-csv` | (none) | Optional path to write per-card CSV breakdown |

**Exit codes**: 0 = success, 1 = model not found, 2 = runtime error

**Stdout**:
```json
{
  "model_path": "models/transformer/model.pt",
  "mean_absolute_error_eur": 1.23,
  "median_percentage_error": 42.5,
  "sample_count": 4032
}
```

## Modified Subcommand: `serve`

No argument changes. The server now loads both models:

1. Sklearn model from `--model-path` (default: `models/latest.joblib`) — **required**
2. Transformer model from `models/transformer/model.pt` — **optional** (graceful degradation)

## Modified Subcommand: `eval`

No argument changes. Output format changes to show both predictions (see evaluate-endpoint.md).
