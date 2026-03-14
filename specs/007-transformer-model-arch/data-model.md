# Data Model: Transformer Model Architecture

**Feature Branch**: `007-transformer-model-arch`
**Date**: 2026-03-14

## Entities

### TransformerConfig

Immutable configuration for reconstructing the transformer model architecture. Stored inside the model artifact.

| Field | Type | Description |
|---|---|---|
| `d_model` | int | Embedding and hidden dimension (128) |
| `n_layers` | int | Number of transformer encoder layers (4) |
| `n_heads` | int | Number of attention heads (4) |
| `ff_dim` | int | Feed-forward inner dimension (512) |
| `max_seq_len` | int | Maximum input token count including `[CLS]` and `[PAD]` (TBD, ~256) |
| `vocab_size` | int | Tokenizer vocabulary size (30,522 for BERT base uncased) |
| `dropout` | float | Dropout rate (0.1) |

**Validation rules**:
- All integer fields > 0
- `d_model` must be divisible by `n_heads` (head_dim = d_model / n_heads)
- `dropout` must be in [0.0, 1.0)

### CardPriceTransformerModel (nn.Module)

The PyTorch model implementing the encoder-only transformer for price regression.

**Architecture layers** (in forward pass order):

| Layer | Input Shape | Output Shape | Description |
|---|---|---|---|
| Token embedding | (batch, seq_len) int | (batch, seq_len, d_model) | `nn.Embedding(vocab_size, d_model)` |
| Positional embedding | (batch, seq_len) int | (batch, seq_len, d_model) | `nn.Embedding(max_seq_len, d_model)` |
| Sum + dropout | (batch, seq_len, d_model) | (batch, seq_len, d_model) | Token emb + pos emb, then dropout |
| TransformerEncoderLayer ×4 | (batch, seq_len, d_model) | (batch, seq_len, d_model) | Multi-head attention + FFN + residual + LayerNorm |
| CLS extraction | (batch, seq_len, d_model) | (batch, d_model) | Select position 0 (the `[CLS]` token) |
| Dropout | (batch, d_model) | (batch, d_model) | Dropout before output head |
| Linear projection | (batch, d_model) | (batch, 1) | `nn.Linear(d_model, 1)` → scalar |

**Forward method signature**:
```python
def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Args:
        input_ids: (batch_size, seq_len) — token IDs from BERT tokenizer
        attention_mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding

    Returns:
        (batch_size,) — predictions in shifted-log-price space
    """
```

### TransformerModelArtifact

The on-disk representation of a trained transformer model. Stored as a single `.pt` file.

| Field | Type | Description |
|---|---|---|
| `state_dict` | dict | PyTorch model state dictionary (all weight tensors) |
| `config` | dict | TransformerConfig fields as a plain dict |

**File path**: `models/transformer/model.pt`

### TransformerTrainResult

Result of a transformer training run. Returned by the training use case.

| Field | Type | Description |
|---|---|---|
| `model_path` | Path | Path to saved `.pt` artifact |
| `best_epoch` | int | Epoch number with lowest validation loss |
| `best_val_loss` | float | Lowest validation loss achieved |
| `stopped_early` | bool | Whether early stopping triggered |
| `final_epoch` | int | Last epoch trained (may be < max epochs if early stopped) |
| `card_count` | int | Number of training cards |
| `cards_skipped` | int | Number of cards skipped (no price or no output file) |
| `price_range_min_eur` | float | Minimum price in training set |
| `price_range_max_eur` | float | Maximum price in training set |
| `max_seq_len` | int | Empirically determined sequence length used |

### TransformerEvalResult

Result of a transformer evaluation run.

| Field | Type | Description |
|---|---|---|
| `model_path` | Path | Path to the evaluated model |
| `mean_absolute_error_eur` | float | MAE in EUR |
| `median_abs_error_log` | float | Median absolute error in shifted-log price space |
| `sample_count` | int | Number of test cards evaluated |
| `per_card` | list[dict] | Optional per-card breakdown |

### TransformerTrainingDataset (torch.utils.data.Dataset)

PyTorch dataset wrapping tokenized card texts paired with shifted-log prices.

| Field | Type | Description |
|---|---|---|
| `input_ids` | Tensor (N, max_seq_len) | Tokenized card texts (padded/truncated) |
| `attention_masks` | Tensor (N, max_seq_len) | 1 for real tokens, 0 for padding |
| `targets` | Tensor (N,) | Shifted-log prices: `log(price + 2)` |

**Construction**: Built from matched (card_name, text_content, price_eur) tuples.

## Relationships

```
TransformerConfig ──> CardPriceTransformerModel (defines architecture)
                  ──> TransformerModelArtifact (stored alongside state_dict)

TransformerTrainingDataset ──> CardPriceTransformerModel (feeds training)

TrainTransformerUseCase ──> TransformerTrainResult
                        ──> TransformerModelArtifact (saves)

EvaluateTransformerUseCase ──> TransformerModelArtifact (loads)
                           ──> TransformerEvalResult

PredictPriceUseCase ──> TransformerModelArtifact (loads for inference)
                    ──> PriceEstimate (returns prediction)

server.create_app() ──> sklearn model artifact (existing)
                    ──> TransformerModelArtifact (new, optional)
```

## State Transitions

### Model Artifact Lifecycle

```
[Not exists] ──(train-transformer)──> [Saved to models/transformer/model.pt]
                                         │
                                  (retrain) overwrites
                                         │
                                         ▼
                           [Updated models/transformer/model.pt]
```

### Training Epoch Lifecycle

```
[Start] ──> [Warmup epochs 1-2] ──> [Full LR epochs 3+]
                                        │
                              ┌─────────┴─────────┐
                              │                     │
                       [Val loss improves]    [Val loss stagnates]
                       Save checkpoint         Increment patience
                              │                     │
                              └─────────┬─────────┘
                                        │
                              [Patience exhausted OR epoch 20]
                                        │
                                        ▼
                              [Training complete]
                              Best checkpoint = final artifact
```
