# Feature Specification: Model Harmonization

**Feature Branch**: `008-model-harmonization`
**Created**: 2026-03-15
**Status**: Draft
**Input**: User description: "Harmonization of the application across models. Get rid of accumulated cruft and inconsistencies, and make way for future addition of more training models without overloading the CLI or REST API."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Unified Model Training (Priority: P1)

A user wants to train either the sklearn or transformer model. Instead of remembering two different commands with different argument patterns (`train` vs `train-transformer`), they use a single `train` command and specify which model to train. Both models now train from the same converted card text files in the `./output` folder, establishing a single canonical input format for all training pipelines.

**Why this priority**: Training is the foundational operation. Unifying the command and input format is the highest-impact change because it eliminates the most significant divergence between the two model pipelines and establishes the pattern that all future models will follow.

**Independent Test**: Can be fully tested by running `train sklearn` and `train transformer` and verifying both produce valid model artifacts from converted card files.

**Acceptance Scenarios**:

1. **Given** converted card text files exist in `./output`, **When** the user runs `train sklearn`, **Then** the sklearn model is trained using only converted card texts and the trained model artifact is saved to the `./models/sklearn/` subdirectory.
2. **Given** converted card text files exist in `./output`, **When** the user runs `train transformer`, **Then** the transformer model is trained using converted card texts and the trained model artifact is saved to the `./models/transformer/` subdirectory.
3. **Given** no model argument is provided, **When** the user runs `train`, **Then** a clear error message is shown listing the available model choices.
4. **Given** the user runs `train sklearn`, **When** training completes, **Then** the output format (metrics, file paths) is consistent with the current training output.

---

### User Story 2 - Unified Prediction Command (Priority: P1)

A user wants to predict the price of a card. Instead of choosing between three overlapping commands (`predict`, `eval`, `evaluate`) with different interfaces and behaviors, they use a single `predict` command, specify which model to use, and provide the card in converted text format via `--file` or `--card`. The prediction runs locally without requiring the REST service.

**Why this priority**: Equal priority to training because prediction is the primary user-facing operation. Collapsing three commands into one with a clean interface removes the biggest source of confusion in the CLI.

**Independent Test**: Can be tested by running `predict sklearn --file path/to/card.txt` and `predict transformer --card "<card text>"` and verifying both return a price estimate.

**Acceptance Scenarios**:

1. **Given** a trained sklearn model exists in `./models/sklearn/`, **When** the user runs `predict sklearn --file path/to/card.txt`, **Then** the system loads the model from the sklearn subdirectory, reads the converted card text file, runs prediction locally, and outputs the predicted price.
2. **Given** a trained transformer model exists in `./models/transformer/`, **When** the user runs `predict transformer --card "<multiline card text>"`, **Then** the system loads the model from the transformer subdirectory, parses the inline card text, runs prediction locally, and outputs the predicted price.
3. **Given** no model argument is provided, **When** the user runs `predict`, **Then** a clear error message is shown listing the available model choices.
4. **Given** neither `--file` nor `--card` is provided, **When** the user runs `predict sklearn`, **Then** a clear error message indicates that one of `--file` or `--card` is required.
5. **Given** both `--file` and `--card` are provided, **When** the user runs `predict sklearn --file x --card y`, **Then** a clear error message indicates that only one input method should be used.

---

### User Story 3 - REST API Accepts Converted Card Format (Priority: P2)

An API consumer sends a card in the converted text format to the prediction endpoint. The endpoint no longer accepts raw Forge card script syntax; it accepts the same converted text format used by the CLI and training pipelines, establishing a single card representation across the entire application.

**Why this priority**: Lower than the CLI changes because the REST API is a secondary interface. However, it is essential for consistency: the entire application should speak one card format.

**Independent Test**: Can be tested by sending a POST request with a converted card text body and verifying the response contains a price prediction.

**Acceptance Scenarios**:

1. **Given** the REST service is running with trained models, **When** a consumer POSTs a multiline string in converted card text format to the prediction endpoint, **Then** the response contains predicted price(s).
2. **Given** an invalid or unparseable card text is sent, **When** the consumer POSTs to the prediction endpoint, **Then** the response is a 400 error with a clear message.

---

### Edge Cases

- What happens when the user provides a model name that doesn't exist (e.g., `train xgboost`)? The system must reject it with a clear error listing valid model names.
- What happens when `--file` points to a nonexistent file? The system must report a clear file-not-found error.
- What happens when the converted card text is malformed (missing required fields like `name:` or `types:`)? The system must report which fields are missing or invalid.
- What happens when no trained model artifact exists for the selected model? The system must report that the model has not been trained yet.
- What happens when the `./output` folder is empty or missing during training? The system must report that no training data is available and suggest running `convert` first.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The CLI MUST provide a single `train` command that accepts a positional model argument with allowed values (currently: `sklearn`, `transformer`).
- **FR-002**: The `train` command MUST use converted card text files from the `./output` folder as the sole training input for all models. Raw Forge card scripts MUST NOT be used directly for training.
- **FR-003**: The `train` command MUST save each model's output artifacts to a dedicated subdirectory under `./models/` named after the model identifier (e.g., `./models/sklearn/`, `./models/transformer/`).
- **FR-004**: The `predict` command MUST default to loading the model from the corresponding `./models/<model>/` subdirectory when no explicit model path is provided.
- **FR-005**: The `train` command MUST retain all model-specific tuning parameters (e.g., batch size, epochs, learning rate for transformer; test split, random seed for sklearn) as optional arguments.
- **FR-006**: The CLI MUST provide a single `predict` command that accepts a positional model argument with the same allowed values as `train`.
- **FR-007**: The `predict` command MUST accept exactly one of two mutually exclusive input methods: `--file` / `-f` (path to a converted card text file) or `--card` / `-c` (inline multiline string in converted card text format).
- **FR-008**: The `predict` command MUST run prediction locally by calling the relevant model's inference logic directly, without requiring or contacting the REST service.
- **FR-009**: The old `predict`, `eval`, `evaluate`, `evaluate-transformer`, and `train-transformer` commands MUST be removed from the CLI.
- **FR-010**: The REST API prediction endpoint MUST accept a multiline string in converted card text format (instead of raw Forge card script format).
- **FR-011**: The system MUST provide clear, actionable error messages when: an invalid model name is given, required arguments are missing, input files don't exist, no trained model is found, or training data is unavailable.
- **FR-012**: The `serve` and `convert` and `check-convert` commands MUST remain unchanged.

### Key Entities

- **Converted Card Text**: A multiline plain-text representation of a card using the format produced by the `convert` command (fields like `name:`, `mana cost:`, `types:`, ability lines). This becomes the single canonical card input format for training, prediction, and the REST API.
- **Model Identifier**: A string key (`sklearn`, `transformer`) that selects which model pipeline to use. Designed to be extensible for future model types. Also determines the subdirectory name under `./models/` where the model's artifacts are stored and loaded from by default.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The CLI exposes exactly two model-related commands (`train`, `predict`) instead of the current five (`train`, `train-transformer`, `predict`, `eval`, `evaluate`, `evaluate-transformer`), reducing command count by more than half.
- **SC-002**: All model training and prediction operations use the same card input format (converted card text), eliminating format divergence between model pipelines.
- **SC-003**: Price predictions via the CLI do not require the REST service to be running; they execute locally and return results within the same time frame as the current implementation.
- **SC-004**: Adding a new model type in the future requires only: implementing the model's train/predict logic and registering the model name as a valid choice. No new CLI commands or REST endpoints are needed.
