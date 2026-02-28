# Feature Specification: Transformer Model Architecture

**Feature Branch**: `009-transformer-model-arch`
**Created**: 2026-03-01
**Status**: Draft
**Input**: User description: "The AI model should be built based on a transformer model with attention, following the architecture of modern LLMs. Training must be possible on a GeForce RTX 3060 Ti with 8GB VRAM."
**Depends on**: `001-card-price-predictor` (prediction requirements and accuracy targets), `008-mtg-custom-tokenizer` (tokenized input format)

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

A model developer or end user wants to use the trained transformer model to predict card prices. They provide tokenized card data (produced by the custom tokenizer from feature 008), and the model outputs a numeric price prediction. The model uses attention mechanisms to weigh the relationships between different parts of the card description (e.g., how keyword abilities interact with mana cost and card types to influence price).

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
- **FR-003**: The model MUST accept tokenized card data (a sequence of token identifiers from the custom tokenizer, feature 008) as input and produce a single numeric price prediction as output.
- **FR-004**: The model MUST handle variable-length input sequences. Cards with different amounts of text produce token sequences of different lengths, and the model must process all of them without requiring manual padding decisions from the user.
- **FR-005**: The model MUST define a maximum input sequence length. Inputs exceeding this length MUST be truncated rather than causing errors.
- **FR-006**: The model MUST use the embedding layer trained alongside the model (using the custom vocabulary from feature 008) to convert token identifiers into dense vector representations that the transformer layers process.
- **FR-007**: Inference (producing a prediction from a single card) MUST be fast enough to support the prediction service response time requirements from feature 005 (price estimate within 3 seconds).
- **FR-008**: The model MUST produce deterministic predictions — the same input with the same model version MUST always yield the same output.
- **FR-009**: The trained model MUST be saveable to and loadable from disk as a self-contained artifact, including all weights and configuration needed to run inference.
- **FR-010**: The model MUST be capable of learning price-relevant patterns from the training data, as evidenced by decreasing training loss and meeting the accuracy targets defined in feature 001 (median percentage error ≤ 50%).

### Key Entities

- **Transformer Model**: The core prediction model. Uses attention mechanisms to process sequences of card tokens and produce a price prediction. Sized to fit within the 8GB VRAM constraint during training. Saved as a model artifact after training.
- **Embedding Layer**: Converts token identifiers from the custom vocabulary into dense vector representations. Trained alongside the transformer model. The vocabulary size is determined by the custom tokenizer (feature 008).
- **Attention Mechanism**: The component within each transformer layer that allows the model to weigh relationships between different tokens in the input sequence (e.g., how a keyword ability relates to the mana cost to influence predicted price).
- **Model Artifact**: The complete trained model saved to disk. Contains all weights (embeddings, transformer layers, output layer) and the configuration needed to load and run inference. Produced by the training stage, consumed by the prediction stage.

## Assumptions

- The model is an encoder-only (or encoder-style) transformer, since the task is regression (input sequence → single number), not text generation. The architecture does not need a decoder or autoregressive generation capabilities.
- The custom tokenizer (feature 008) produces token sequences that are the direct input to the model. No additional preprocessing is needed between tokenization and model input beyond padding/truncation to the maximum sequence length.
- The 8GB VRAM budget must accommodate the model parameters, the batch of training examples, gradients, and optimizer state simultaneously during training. The model size must be chosen accordingly.
- The maximum input sequence length will be determined during implementation based on the distribution of tokenized card lengths in the dataset. Typical cards are expected to produce sequences of a few dozen to a few hundred tokens.
- Mixed precision training and gradient accumulation are acceptable techniques to fit within the VRAM budget and improve effective batch size, but these are implementation choices — the requirement is simply that training completes within 8GB.
- The model output is a single scalar value (predicted price). The output layer maps the transformer's representation to this scalar.
- Training uses the training dataset from feature 007's dataset preparation stage. The validation dataset is used to monitor overfitting and report accuracy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model training completes on a GeForce RTX 3060 Ti (8GB VRAM) without out-of-memory errors, using the full training dataset.
- **SC-002**: Peak VRAM usage during training stays below 8GB as measured by GPU monitoring tools.
- **SC-003**: The trained model achieves a median percentage error of 50% or less on the held-out validation dataset (per feature 001 SC-002).
- **SC-004**: Single-card inference completes in under 100 milliseconds, well within the 3-second endpoint requirement from feature 005.
- **SC-005**: Training on the full dataset (20,000+ cards) completes within 10 minutes on the target hardware (per feature 001 SC-004).
- **SC-006**: Training loss consistently decreases over the course of training, demonstrating that the model learns from the data rather than producing random predictions.
