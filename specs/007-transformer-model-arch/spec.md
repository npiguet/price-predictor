C# Feature Specification: Transformer Model Architecture

**Feature Branch**: `007-transformer-model-arch`
**Created**: 2026-03-01
**Status**: Draft
**Input**: User description: "The AI model should be built based on a transformer model with attention, following the architecture of modern LLMs. Training must be possible on a GeForce RTX 3060 Ti with 8GB VRAM."
**Depends on**: `001-card-price-predictor` (prediction requirements and accuracy targets), `006-card-script-parsing` (card text input)

## Clarifications

### Session 2026-03-14

- Q: Does the transformer model replace the existing sklearn model or coexist alongside it? → A: Coexist — both models are available; the prediction service runs both.
- Q: Where should the trained model artifact (`.pt` file) be saved? → A: `models/transformer/` — under the existing `models/` directory, separated by model type.
- Q: Should training and inference support CPU fallback when no GPU is available? → A: GPU required for training, CPU fallback for inference.
- Q: How should the training process be invoked? → A: Subcommand added to an existing CLI entry point.
- Q: When the API evaluation endpoint is invoked, which models run? → A: Both models always run. The response includes the predicted price from both the sklearn model and the transformer model. There is no model selection parameter — both predictions are always returned.
- Q: Should model evaluation run automatically after training or be a separate CLI subcommand? → A: Both — auto-evaluate after training (metrics printed at end of run), plus a separate CLI subcommand for re-evaluation against any saved model.
- Q: Should `torch` and `transformers` be added as required project dependencies? → A: Yes — both `torch` and `transformers` (Hugging Face) are required project dependencies for this feature.
- Q: When retraining, what happens to the existing model artifact? → A: Overwrite — new training always replaces the existing `.pt` file in `models/transformer/`.
- Q: How does the evaluation subcommand specify which model to evaluate? → A: Default path (`models/transformer/`) with optional `--model-path` override.
- Q: Is `max_seq_len = 256` final, or should it be determined empirically? → A: Empirical — analyze tokenized card lengths first during implementation, then finalize (may differ from 256).
- Q: What training progress information should the CLI display? → A: Per-epoch summary line (epoch number, train loss, val loss, elapsed time) plus a final summary with best epoch and early-stop status. No progress bars.

### Session 2026-03-12

- Q: What text representation of a card is fed to the tokenizer? → A: The full converted card script text from feature 006 (the `.txt` files in `./output/`). Each file contains a structured text representation with fields like `name:`, `mana cost:`, `types:`, `static:`, `activated:`, `triggered:`, `spell:`, etc. The entire file content is used as-is as the tokenizer input.
- Q: What training/validation split strategy should be used? → A: Reuse feature 001's 80/20 random split to keep evaluation comparable across models.
- Q: What loss function should be used for training? → A: MSE (mean squared error) on shifted-log prices. The shifted-log transform already tames outliers, so a robust loss is unnecessary.
- Q: Should this feature reuse feature 001's data loading infrastructure for building the training dataset? → A: Yes — reuse `forge_parser.py` and `mtgjson_loader.py` for card loading and price matching. The converted card scripts from `./output/` are matched to prices via the existing pipeline.
- Q: Which specific tokenizer should the model use? → A: BERT WordPiece tokenizer (`bert-base-uncased`, ~30K vocab). Pre-trained, no custom training step needed, good English coverage for card text.
- Q: What training hyperparameters should be used? → A: AdamW optimizer, lr=1e-4, batch size 64, 20 epochs, linear warmup over first 2 epochs.
- Q: Should training use checkpointing and/or early stopping? → A: Save best model checkpoint (by validation loss) + early stopping with patience of 5 epochs.

### Session 2026-03-01

- Q: What format should the trained model artifact be saved in? → A: PyTorch native (`.pt` file with state dict + config)
- Q: Should the model input include special tokens? → A: Yes — `[CLS]` token at sequence start for aggregation (BERT-style) and `[PAD]` for batching. No `[SEP]` token (single card input, no segment pairs).
- Q: Should the model predict raw EUR prices or log-transformed prices? → A: Shifted-log-transformed prices — model predicts `log(price + 2)` during training, output is converted back via `exp(x) - 2` for display. The shift of 2 EUR (the community "bulk" threshold) compresses price differences below ~2 EUR, making the model focus its accuracy on cards that are worth distinguishing by price. Standard approach adapted for the MTG price distribution.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Model on Consumer Hardware (Priority: P1)

A model developer wants to train the card price prediction model on their local workstation equipped with a GeForce RTX 3060 Ti (8GB VRAM). They run the training stage (from feature 007), and the model trains to completion without running out of GPU memory. The training process uses the transformer architecture with attention mechanisms to learn patterns from tokenized card data and their associated prices.

**Why this priority**: The 8GB VRAM constraint is a hard requirement that fundamentally shapes the model's size and training approach. If training cannot complete on the target hardware, the model is unusable. This must be validated before anything else.

**Independent Test**: Can be fully tested by running the training stage on a machine with a GeForce RTX 3060 Ti (8GB VRAM) and verifying that training completes without out-of-memory errors while producing a model artifact.

**Acceptance Scenarios**:

1. **Given** a prepared training dataset and token vocabulary, **When** the user starts model training on a system with 8GB VRAM, **Then** training runs to completion without exceeding available GPU memory.
2. **Given** training is in progress, **When** the user monitors GPU memory usage, **Then** peak VRAM consumption stays below 8GB throughout the entire training run.
3. **Given** the training completes, **When** the user inspects the output, **Then** a trained model artifact is saved to disk that uses the transformer architecture with attention.
4. **Given** training is in progress, **When** the user observes CLI output, **Then** training loss decreases over time, indicating the model is learning from the data.

---

### User Story 2 - Predict Card Prices from Tokenized Input (Priority: P2)

A model developer or end user wants to use the trained transformer model to predict card prices. They provide tokenized card data, and the model outputs a numeric price prediction. The model uses attention mechanisms to weigh the relationships between different parts of the card description (e.g., how keyword abilities interact with mana cost and card types to influence price).

**Why this priority**: Price prediction is the model's purpose. This validates that the architecture can actually learn the relationship between card attributes and prices, not just that it fits in memory.

**Independent Test**: Can be tested by loading a trained model, feeding it tokenized representations of known cards, and verifying that it produces numeric price predictions that correlate with actual prices.

**Acceptance Scenarios**:

1. **Given** a trained model and a tokenized card representation, **When** the model runs inference, **Then** it produces a single numeric price prediction (a positive value in the expected currency unit).
2. **Given** the same trained model and the same tokenized input, **When** inference is run multiple times, **Then** the same price prediction is returned every time (deterministic output).
3. **Given** tokenized inputs for two cards where one is known to be significantly more valuable (e.g., a mythic rare multi-format staple vs. a bulk common), **When** both are run through the model, **Then** the expensive card receives a meaningfully higher predicted price.
4. **Given** a tokenized input with fewer tokens (e.g., a simple vanilla creature), **When** inference is run, **Then** the model handles the shorter input gracefully and still produces a prediction.

---

### User Story 3 - Evaluate Model Quality (Priority: P3)

A model developer wants to evaluate how well the transformer architecture performs on the card price prediction task. They run the trained model against the validation dataset and review accuracy metrics to understand whether the attention-based architecture captures meaningful price-relevant patterns in card data.

**Why this priority**: Evaluation confirms the architecture choice is sound for this specific task. It depends on having a working model (P1) that produces predictions (P2).

**Independent Test**: Can be tested by running the trained model against the held-out validation dataset and verifying that accuracy metrics meet the thresholds defined in feature 001.

**Acceptance Scenarios**:

1. **Given** a trained model and the validation dataset, **When** evaluation is run, **Then** the system reports accuracy metrics including mean absolute error and median percentage error.
2. **Given** the evaluation results, **When** compared to the accuracy targets from feature 001, **Then** the model meets or exceeds the target of median percentage error of 50% or less.
3. **Given** the evaluation results, **When** the per-card breakdown is reviewed, **Then** the model's predictions show a meaningful correlation with actual prices (not random).

---

### Edge Cases

- What happens when a card's tokenized representation is very short (e.g., a vanilla creature with just name, cost, types, and P/T)? The model must handle variable-length sequences and still produce meaningful predictions for short inputs.
- What happens when a card's tokenized representation is unusually long (e.g., a card with extensive oracle text and many abilities)? The model must handle long inputs up to its maximum sequence length without errors, and truncate gracefully if the input exceeds the limit.
- What happens when the training dataset has highly skewed prices (most cards are cheap, few are expensive)? The model must still learn to predict across the full price range, not just collapse to predicting the median price for everything.
- What happens when the model is asked to predict a price for a card with attributes very different from the training distribution (e.g., a hypothetical card with unusual ability combinations)? The model produces a best-effort prediction; it may be less accurate for out-of-distribution inputs but must not crash or return invalid output.
- What happens if VRAM is partially consumed by other processes when training starts? The system should detect available memory and report an error if insufficient VRAM is available, rather than crashing mid-training.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The prediction model MUST use a transformer architecture with attention mechanisms, following the general design patterns of modern language models.
- **FR-002**: The model MUST be trainable on a GPU with 8GB VRAM (GeForce RTX 3060 Ti). Peak GPU memory consumption during training MUST NOT exceed 8GB.
- **FR-003**: The model MUST accept tokenized card data (a sequence of token identifiers) as input, prepended with a `[CLS]` token and padded with `[PAD]` tokens for batching, and produce a single numeric price prediction as output. The input text is the full converted card script from feature 006 (`./output/*.txt` files), tokenized using the BERT WordPiece tokenizer (`bert-base-uncased`, ~30K vocab) from the Hugging Face `transformers` library. Internally, the model predicts in shifted-log-price space (trained on `log(price + 2)` where 2 EUR is the bulk threshold); the output is converted back to EUR via `exp(x) - 2` for the final prediction. The prediction is derived from the transformer's output at the `[CLS]` position.
- **FR-004**: The model MUST handle variable-length input sequences. Cards with different amounts of text produce token sequences of different lengths, and the model must process all of them without requiring manual padding decisions from the user.
- **FR-005**: The model MUST define a maximum input sequence length. Inputs exceeding this length MUST be truncated rather than causing errors.
- **FR-006**: The model MUST use an embedding layer to convert token identifiers into dense vector representations that the transformer layers process. The embedding layer is trained alongside the model using the vocabulary defined by the chosen standard tokenizer.
- **FR-007**: Inference (producing a prediction from a single card) MUST be fast enough to support the prediction service response time requirements from feature 005 (price estimate within 3 seconds).
- **FR-008**: The model MUST produce deterministic predictions — the same input with the same model version MUST always yield the same output.
- **FR-009**: The trained model MUST be saveable to and loadable from disk as a PyTorch native `.pt` file containing the model state dict and all configuration needed to reconstruct the model and run inference. The model artifact MUST be saved to `models/transformer/`. Retraining overwrites the existing artifact; no versioning or history is maintained.
- **FR-012**: The transformer model MUST coexist alongside the existing sklearn model from feature 001. When the API evaluation endpoint (`POST /api/v1/evaluate`) is invoked, BOTH models MUST be run and the response MUST include the predicted price from each model. There is no model selection parameter — both predictions are always returned. The response format MUST include separate fields for each model's prediction (e.g., `predicted_price_eur_sklearn` and `predicted_price_eur_transformer`). If the transformer model is not available (no trained artifact), its prediction field MUST be `null` and the sklearn prediction MUST still be returned (graceful degradation).
- **FR-013**: Inference MUST support CPU fallback when no GPU is available. Training requires GPU.
- **FR-014**: Training MUST be invocable as a subcommand added to an existing CLI entry point (not a standalone script). During training, the CLI MUST print a per-epoch summary line containing epoch number, train loss, validation loss, and elapsed time (e.g., `Epoch 3/20 — train_loss: 0.142, val_loss: 0.158, 12.3s`). After training completes, a final summary MUST be printed showing the best epoch and early-stopping status. No progress bars.
- **FR-015**: Evaluation MUST run automatically after training completes, printing metrics (MAE, median percentage error) to the CLI. A separate evaluation subcommand MUST also be available to re-evaluate any saved model against the validation dataset independently. The evaluation subcommand defaults to loading the model from `models/transformer/` but accepts an optional `--model-path` argument to evaluate a model at a different location.
- **FR-010**: The model MUST be capable of learning price-relevant patterns from the training data, as evidenced by decreasing training loss (MSE on shifted-log prices) and meeting the accuracy targets defined in feature 001 (median percentage error ≤ 50%).
- **FR-011**: Training MUST save the best model checkpoint (lowest validation loss) and apply early stopping with a patience of 5 epochs. The final saved model artifact is the best checkpoint, not the last epoch's weights.

### Key Entities

- **Transformer Model**: The core prediction model. Uses attention mechanisms to process sequences of card tokens and produce a price prediction. Sized to fit within the 8GB VRAM constraint during training. Saved as a model artifact after training.
- **Embedding Layer**: Converts token identifiers into dense vector representations. Trained alongside the transformer model. Uses the BERT `bert-base-uncased` vocabulary (30,522 tokens), which already includes `[CLS]` and `[PAD]` special tokens.
- **Attention Mechanism**: The component within each transformer layer that allows the model to weigh relationships between different tokens in the input sequence (e.g., how a keyword ability relates to the mana cost to influence predicted price).
- **Model Artifact**: The complete trained model saved to disk as a PyTorch native `.pt` file in `models/transformer/`. Contains all weights (embeddings, transformer layers, output layer) as a state dict, plus the model configuration needed to reconstruct the architecture and run inference. Produced by the training stage, consumed by the prediction stage. Coexists with the sklearn model artifacts in `models/`.

### Network Architecture

The model is an encoder-only transformer sized to fit comfortably within 8GB VRAM during training while avoiding overfitting on ~20,000 training examples. The architecture follows this structure:

```
Input token IDs (from standard tokenizer)
  → Token embedding (vocab_size × 128)
  → + Learned positional encoding (max_seq_len × 128)
  → Transformer encoder layer ×4:
      → Multi-head self-attention (4 heads, head_dim = 32)
      → Residual connection + LayerNorm
      → Feed-forward network (128 → 512 → 128, ReLU activation)
      → Residual connection + LayerNorm
  → Extract [CLS] position output (128-dim vector)
  → Dropout (0.1)
  → Linear projection (128 → 1) → scalar in shifted-log-price space
  → exp(x) − 2 → EUR price
```

**Dimensions**:

| Parameter | Value | Rationale |
|---|---|---|
| `d_model` | 128 | Embedding and hidden dimension. Kept small — 20K training examples cannot support larger representations without overfitting. |
| `n_layers` | 4 | Transformer encoder layers. Enough depth for multi-hop attention patterns (e.g., keyword + mana cost → price) without excess capacity. |
| `n_heads` | 4 | Attention heads (head_dim = 32 each). Allows the model to attend to different relationship types in parallel. |
| `ff_dim` | 512 | Feed-forward inner dimension (4× d_model, the standard transformer ratio). |
| `max_seq_len` | TBD (starting estimate: 256) | Maximum input token count including `[CLS]` and `[PAD]`. To be determined empirically by analyzing the tokenized card length distribution during implementation. |
| `dropout` | 0.1 | Applied after positional encoding and before the output head. Primary regularization against overfitting. |
| `vocab_size` | 30,522 | BERT `bert-base-uncased` WordPiece vocabulary size (fixed). |

**Size estimate**: ~2M parameters. Training memory footprint with AdamW optimizer (fp32): ~350 MB for batch size 64 — well within the 8GB VRAM budget.

**Training hyperparameters**:

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for transformer training; weight decay helps regularize. |
| Learning rate | 1e-4 | Conservative starting point for small transformers; avoids divergence. |
| Batch size | 64 | Fits comfortably in 8GB VRAM with this model size. |
| Epochs | 20 | ~16K training examples × 20 passes provides sufficient gradient updates for convergence. |
| LR schedule | Linear warmup (2 epochs) | Stabilizes early training before full learning rate is applied. |
| Early stopping | Patience = 5 epochs | Stop training if validation loss does not improve for 5 consecutive epochs; prevents overfitting. |
| Checkpointing | Best model (by val loss) | Save the model state dict whenever validation loss reaches a new minimum; final artifact is the best checkpoint. |

## Assumptions

- The model is an encoder-only (or encoder-style) transformer, since the task is regression (input sequence → single number), not text generation. The architecture does not need a decoder or autoregressive generation capabilities.
- The tokenizer is the BERT WordPiece tokenizer (`bert-base-uncased`, vocab size 30,522) from the Hugging Face `transformers` library. Both `torch` and `transformers` are required project dependencies for this feature. The model pipeline prepends a `[CLS]` token, truncates to the maximum sequence length if needed, and pads with `[PAD]` tokens for batching. These special tokens are already present in the BERT vocabulary.
- The 8GB VRAM budget must accommodate the model parameters, the batch of training examples, gradients, and optimizer state simultaneously during training. The model size must be chosen accordingly.
- The maximum input sequence length will be determined empirically during implementation by analyzing the distribution of tokenized card lengths in the dataset. The starting estimate is 256 tokens, but the final value may differ. Typical cards are expected to produce sequences of a few dozen to a few hundred tokens.
- Training uses AdamW optimizer (lr=1e-4, batch size 64, 20 epochs) with linear warmup over the first 2 epochs. Mixed precision training and gradient accumulation are acceptable additional techniques to fit within the VRAM budget, but these are implementation choices — the requirement is simply that training completes within 8GB.
- The model internally predicts shifted-log-transformed prices (`log(price + 2)` where 2 EUR is the community "bulk" threshold). The output layer maps the transformer's `[CLS]` representation to a single scalar in shifted-log-price space. During inference, the output is converted back to EUR via `exp(x) - 2`. This shifted-log approach compresses price differences below ~2 EUR (treating all bulk cards as roughly equivalent) while preserving proportional sensitivity in the mid-to-high price range where distinctions matter.
- Training uses an 80/20 random split (same strategy as feature 001) to keep evaluation comparable across models. The validation dataset is used to monitor overfitting and report accuracy.
- The transformer model coexists with the existing sklearn model from feature 001. When the evaluation endpoint is called, both models are always run and both predictions are returned. If the transformer model artifact is not available, its prediction is returned as `null` while the sklearn prediction is still provided.
- Inference supports CPU fallback (automatic device detection) so the prediction service does not require GPU. Training requires GPU (CUDA).
- Training is invoked as a subcommand on an existing CLI entry point, not as a standalone script.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model training completes on a GeForce RTX 3060 Ti (8GB VRAM) without out-of-memory errors, using the full training dataset.
- **SC-002**: Peak VRAM usage during training stays below 8GB as measured by GPU monitoring tools.
- **SC-003**: The trained model achieves a median percentage error of 50% or less on the held-out validation dataset (per feature 001 SC-002).
- **SC-004**: Single-card inference completes in under 100 milliseconds, well within the 3-second endpoint requirement from feature 005.
- **SC-005**: Training on the full dataset (20,000+ cards) completes within 10 minutes on the target hardware (per feature 001 SC-004).
- **SC-006**: Training loss consistently decreases over the course of training, demonstrating that the model learns from the data rather than producing random predictions.
