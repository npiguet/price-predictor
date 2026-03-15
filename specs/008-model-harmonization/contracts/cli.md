# CLI Contract: 008 Model Harmonization

## Commands

### train {model}

```text
price-predictor train sklearn [options]
price-predictor train transformer [options]
```

**Shared options** (all models):
- `--output-dir` — Converted card text directory (default: `./output`)
- `--prices-path` — AllPricesToday.json path (default: `resources/AllPricesToday.json`)
- `--printings-path` — AllPrintings.json path (default: `resources/AllPrintings.json`)
- `--model-output` — Model artifact output directory (default: `./models/<model>/`)
- `--test-split` — Train/test split ratio (default: `0.2`)
- `--random-seed` — Random seed (default: `42`)

**Transformer-specific options**:
- `--batch-size` (default: `64`)
- `--epochs` (default: `20`)
- `--lr` (default: `1e-4`)
- `--patience` (default: `5`)

**Output**: JSON to stdout with model metadata (version, path, card count, metrics).

### predict {model}

```text
price-predictor predict sklearn (--file PATH | --card TEXT)
price-predictor predict transformer (--file PATH | --card TEXT)
```

**Options**:
- `--file`, `-f` — Path to a converted card text file (mutually exclusive with `--card`)
- `--card`, `-c` — Inline multiline converted card text (mutually exclusive with `--file`)

**Output**: JSON to stdout with `predicted_price_eur` and `model_version`.

### evaluate {model}

```text
price-predictor evaluate sklearn [options]
price-predictor evaluate transformer [options]
```

**Shared options**:
- `--output-dir` — Converted card text directory (default: `./output`)
- `--prices-path`, `--printings-path` — Same as train
- `--model-path` — Model artifact path (default: `./models/<model>/`)
- `--test-split`, `--random-seed` — Same as train

**Sklearn-specific options**:
- `--output-csv` — Optional CSV output path for per-card results

**Output**: JSON to stdout with evaluation metrics.

### Unchanged commands

- `serve` — Unchanged (starts REST API server)
- `convert` — Unchanged (converts Forge scripts to text)
- `check-convert` — Unchanged (validates converted files)

## Removed commands

- `train-transformer` (replaced by `train transformer`)
- `evaluate-transformer` (replaced by `evaluate transformer`)
- `predict` old-style (replaced by `predict {model}`)
- `eval` (replaced by `predict {model}`)
- `evaluate` old-style (replaced by `evaluate {model}`)
