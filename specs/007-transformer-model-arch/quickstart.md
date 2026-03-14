# Quickstart: Transformer Model Architecture

**Feature Branch**: `007-transformer-model-arch`

## Prerequisites

- Python 3.14+
- NVIDIA GPU with ≥8 GB VRAM (CUDA) for training
- Converted card scripts in `output/` (from `price_predictor convert`)
- MTGJSON data files in `resources/` (`AllPrintings.json`, `AllPricesToday.json`)
- Forge card scripts at `../forge/forge-gui/res/cardsfolder/`

## Install Dependencies

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
```

This installs CUDA-enabled `torch` and `transformers` (added as project dependencies for this feature) alongside the existing dependencies. The `--extra-index-url` is required to get the GPU build of PyTorch — the default PyPI index only has the CPU-only build.

## Train the Transformer Model

```bash
python -m price_predictor train-transformer
```

This will:
1. Load Forge cards and match to MTGJSON prices (same pipeline as sklearn training)
2. Load converted card texts from `output/` and tokenize with BERT WordPiece
3. Analyze token length distribution and set `max_seq_len`
4. Split 80/20 train/validation (seed 42, same strategy as sklearn)
5. Train for up to 20 epochs with early stopping (patience 5)
6. Print per-epoch summaries: `Epoch 3/20 — train_loss: 0.142, val_loss: 0.158, 12.3s`
7. Save best checkpoint to `models/transformer/model.pt`
8. Auto-evaluate on validation set and print metrics

**Key options**:
```bash
--model-output models/transformer/   # Where to save the model
--batch-size 64                      # Batch size (fits 8GB VRAM)
--epochs 20                          # Maximum training epochs
--lr 1e-4                            # Learning rate
--patience 5                         # Early stopping patience
--random-seed 42                     # Reproducibility seed
```

## Evaluate the Transformer Model

```bash
python -m price_predictor evaluate-transformer
```

Re-evaluates a saved model against the validation set. Defaults to `models/transformer/` but supports `--model-path` override.

**Output** (printed to CLI):
```json
{
  "mean_absolute_error_eur": 1.23,
  "median_abs_error_log": 0.19,
  "sample_count": 4000
}
```

## Run the Prediction Service

```bash
python -m price_predictor serve
```

The server loads both the sklearn model (`models/latest.joblib`) and the transformer model (`models/transformer/model.pt`). Both models run on every evaluation request.

**Evaluate a card**:
```bash
python -m price_predictor eval path/to/card_script.txt
```

**Response includes both predictions**:
```json
{
  "sklearn": {
    "predicted_price_eur": 2.35,
    "model_version": "20260301-143601"
  },
  "transformer": {
    "predicted_price_eur": 2.18,
    "model_version": "transformer-v1"
  }
}
```

If the transformer model is not trained yet, `"transformer"` is `null` and the sklearn prediction is still returned.

## Verify GPU Availability

```python
import torch
print(torch.cuda.is_available())        # True if CUDA GPU detected
print(torch.cuda.get_device_name(0))     # e.g., "NVIDIA GeForce RTX 3060 Ti"
print(torch.cuda.mem_get_info(0))        # (free, total) in bytes
```

## File Layout After Training

```
models/
├── latest.joblib                    # Sklearn model (existing)
├── 20260301-143601.joblib           # Sklearn model (timestamped)
└── transformer/
    └── model.pt                     # Transformer model artifact
```
