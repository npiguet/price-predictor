# Feature Specification: ML Pipeline CLI Stages

**Feature Branch**: `007-pipeline-cli`
**Created**: 2026-02-28
**Status**: Draft
**Input**: User description: "The application has several different stages in its lifecycle. All stages can be invoked by the user manually via the CLI: (1) Dataset preparation, (2) Pre-Training, (3) Training, (4) Prediction."
**Depends on**: `001-card-price-predictor` (model and training concepts), `006-card-script-parsing` (card script extraction)

## Clarifications

### Session 2026-03-01

- Q: How should model artifacts be handled on retrain? → A: Timestamped artifacts — each training run saves with a timestamp. A "latest" pointer is updated. Prediction loads "latest" by default.
- Q: Should external paths (Forge dir, price data URL) be configurable? → A: Yes, via CLI flags with sensible defaults. Each stage accepts optional path overrides (e.g., `--forge-dir`, `--price-url`) with the current assumed paths as defaults.
- Q: What train/validation split ratio should be used? → A: 80/20 — 80% training, 20% validation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare Dataset (Priority: P1)

A user wants to prepare the data needed for model training. They run a CLI command that downloads any required data files (e.g., price data), reads card scripts from the Forge card folder, converts them to the structured format used by the model, and produces two datasets: a larger training dataset and a smaller validation dataset. After this step, the user has ready-to-use datasets on disk that can be fed into subsequent pipeline stages.

**Why this priority**: This is the first stage in the pipeline and a prerequisite for all subsequent stages. Without a prepared dataset, no training or prediction is possible. It also validates that the data sources (card scripts and price data) are accessible and correctly processed.

**Independent Test**: Can be fully tested by running the dataset preparation command and verifying that training and validation dataset files are created on disk with the expected structure and record counts.

**Acceptance Scenarios**:

1. **Given** the Forge card scripts directory and a price data source are available, **When** the user runs the dataset preparation command, **Then** the system produces a training dataset and a validation dataset, both saved to disk.
2. **Given** the required price data file is not yet present locally, **When** the user runs dataset preparation, **Then** the system downloads the price data file automatically before proceeding.
3. **Given** the price data file is already present locally, **When** the user runs dataset preparation, **Then** the system uses the existing file without re-downloading.
4. **Given** some card scripts cannot be matched to a price record, **When** dataset preparation runs, **Then** those cards are excluded from the datasets and the system reports how many cards were matched and how many were skipped.
5. **Given** the dataset preparation completes, **When** the user inspects the output, **Then** each record in the datasets contains the structured card data (as defined in feature 006) paired with its price.

---

### User Story 2 - Pre-Training Token Extraction (Priority: P2)

A user wants to prepare the token vocabulary before training the model. They run a CLI command that scans the training and validation datasets, identifies all distinct tokens (words, symbols, and card-specific terms that appear in card text and attributes), and saves this token list to disk. These tokens will be used to train embeddings in the training stage.

**Why this priority**: The token list must exist before training can begin, since embeddings are built from this vocabulary. However, this stage depends on having a prepared dataset (P1) first.

**Independent Test**: Can be tested by running the pre-training command against prepared datasets and verifying that a token list file is produced containing the expected vocabulary items.

**Acceptance Scenarios**:

1. **Given** training and validation datasets have been prepared, **When** the user runs the pre-training command, **Then** the system produces a token list file saved to disk.
2. **Given** the prepared datasets contain cards with diverse oracle text, mana costs, and ability descriptions, **When** pre-training completes, **Then** the token list includes tokens from all relevant text fields in the datasets.
3. **Given** the datasets are not yet prepared, **When** the user runs the pre-training command, **Then** the system reports an error indicating that dataset preparation must be run first.

---

### User Story 3 - Train Model (Priority: P3)

A user wants to train the price prediction model. They run a CLI command that uses the prepared training dataset and the token list to train embeddings and the main prediction model. The trained model is saved as an artifact on disk that can later be used for prediction.

**Why this priority**: Training produces the model artifact that makes predictions possible. It depends on both dataset preparation (P1) and pre-training (P2).

**Independent Test**: Can be tested by running the training command with prepared datasets and token list, then verifying that a model artifact is produced on disk and that training metrics (e.g., loss values) are reported during the process.

**Acceptance Scenarios**:

1. **Given** prepared datasets and a token list exist, **When** the user runs the training command, **Then** the system trains the model and saves a model artifact to disk.
2. **Given** training is in progress, **When** the user observes the CLI output, **Then** the system displays progress information including training metrics (e.g., loss, validation performance) at regular intervals.
3. **Given** the token list or datasets are missing, **When** the user runs the training command, **Then** the system reports an error indicating which prerequisite is missing.
4. **Given** a previous model artifact already exists, **When** the user retrains, **Then** a new timestamped model artifact is saved alongside the previous one, and the "latest" pointer is updated to the new artifact.

---

### User Story 4 - Serve Predictions (Priority: P4)

A user wants to use the trained model to evaluate cards. They run a CLI command that starts the prediction service, which exposes an endpoint where users can submit card descriptions and receive price evaluations. This is the runtime phase that makes the model's predictions accessible to end users.

**Why this priority**: This is the ultimate goal of the pipeline — making predictions available. It depends on having a trained model (P3) and the card evaluation endpoint interface (feature 005).

**Independent Test**: Can be tested by starting the prediction service, submitting a card description to the exposed endpoint, and verifying that a price evaluation is returned.

**Acceptance Scenarios**:

1. **Given** a trained model artifact exists, **When** the user runs the prediction command, **Then** the system starts serving predictions through the endpoint defined in feature 005.
2. **Given** no trained model artifact exists, **When** the user runs the prediction command, **Then** the system reports an error indicating that training must be run first.
3. **Given** the prediction service is running, **When** a user submits a card description to the endpoint, **Then** the service returns a price evaluation using the trained model.

---

### Edge Cases

- What happens when the user runs a stage out of order (e.g., training before dataset preparation)? The system checks for the required prerequisites and reports a clear error message naming the missing prerequisite stage.
- What happens when the Forge card scripts directory path is incorrect or inaccessible? The system reports an error identifying the expected path and what went wrong.
- What happens when the price data download fails (e.g., network issue)? The system reports the download failure with the specific error and does not proceed with incomplete data.
- What happens when the training dataset is very small (e.g., fewer than 100 matched cards)? The system warns the user that accuracy may be poor but proceeds with training unless the dataset is empty.
- What happens when the user interrupts a long-running stage (e.g., training) with Ctrl+C? The system handles the interruption gracefully without corrupting existing artifacts on disk.
- What happens when disk space is insufficient to save datasets or model artifacts? The system reports a clear error about insufficient storage.
- What happens when the user runs dataset preparation again after already having datasets? The system regenerates the datasets from scratch, replacing the previous ones.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a CLI interface with distinct commands for each pipeline stage: dataset preparation, pre-training, training, and prediction. Each stage MUST accept optional CLI flags to override default paths (e.g., `--forge-dir` for the card scripts directory, `--price-url` for the price data download URL). When flags are omitted, the system uses the default paths from the Assumptions section.
- **FR-002**: The dataset preparation stage MUST download required data files (price data) if they are not already present locally.
- **FR-003**: The dataset preparation stage MUST read card scripts from the Forge card folder, convert them to structured card data (per feature 006), match them to price records, and produce a training dataset and a validation dataset.
- **FR-004**: The dataset preparation stage MUST report the number of cards successfully processed, the number of cards skipped (with reasons), and the sizes of the resulting datasets.
- **FR-005**: The pre-training stage MUST scan the training and validation datasets and produce a token list containing all distinct tokens found in the card data fields.
- **FR-006**: The training stage MUST use the prepared training dataset and token list to train embeddings and the main prediction model, producing a timestamped model artifact saved to disk. Each training run preserves previous artifacts and updates a "latest" pointer to the newest artifact.
- **FR-007**: The training stage MUST display progress information during training, including training metrics at regular intervals.
- **FR-008**: The prediction stage MUST start a service that exposes the card evaluation endpoint (per feature 005), using the "latest" model artifact by default for predictions.
- **FR-009**: Each stage MUST validate that its prerequisites exist before executing. If prerequisites are missing, the stage MUST report a clear error naming the missing prerequisite.
- **FR-010**: Each stage MUST handle interruption (e.g., Ctrl+C) gracefully without corrupting previously saved artifacts.
- **FR-011**: The validation dataset MUST be a distinct subset of the overall data, separate from the training dataset, using an 80/20 split (80% training, 20% validation), so it can be used for unbiased model evaluation.
- **FR-012**: All pipeline artifacts (datasets, token list, model) MUST be saved to well-defined locations on disk, so each stage can locate the outputs of previous stages.

### Key Entities

- **Training Dataset**: A collection of structured card data records, each paired with a price value. Used to train the prediction model. Produced by the dataset preparation stage.
- **Validation Dataset**: A smaller collection of structured card data records with prices, held out from training. Used to evaluate model performance during and after training.
- **Token List**: A vocabulary of all distinct tokens extracted from the training and validation datasets. Used to train embeddings in the training stage. Produced by the pre-training stage.
- **Model Artifact**: The trained prediction model saved to disk with a timestamp identifier. Includes trained embeddings and the main model weights. Each training run produces a new timestamped artifact and updates a "latest" pointer. The prediction stage loads "latest" by default. Previous artifacts are preserved for comparison or rollback.
- **Pipeline Stage**: A discrete step in the application lifecycle. Each stage has defined inputs (prerequisites), outputs (artifacts), and a CLI command to invoke it.

## Assumptions

- The Forge card scripts are located at `../forge/forge-gui/res/cardsfolder/` by default and follow the format documented in `resources/CARD_SCRIPTING_REFERENCE.md`. This path is overridable via CLI flag.
- Price data is sourced from MTGJSON's AllPricesToday.json by default (consistent with feature 001). The download URL is overridable via CLI flag.
- The validation dataset is created by splitting the overall matched data, not by using a separate data source. The split ratio is 80/20 (80% training, 20% validation).
- All pipeline artifacts are stored in a single project-level output directory, with consistent naming conventions so stages can discover each other's outputs.
- The prediction stage reuses the endpoint interface defined in feature 005 — this feature specifies how to start and stop that service via the CLI, not the endpoint contract itself.
- Stages are run sequentially by the user. There is no automatic orchestration that chains all stages together (though the user could script this themselves).
- The CLI is intended for a single user on a local workstation. Multi-user or distributed execution is out of scope.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new user can run all four pipeline stages in sequence (dataset preparation through prediction) by following the CLI help text, without needing external documentation.
- **SC-002**: Dataset preparation processes the full Forge card library (20,000+ scripts) and produces datasets within 5 minutes.
- **SC-003**: Each stage clearly reports success or failure upon completion, including a summary of what was produced (e.g., record counts, artifact paths, training metrics).
- **SC-004**: Running a stage with missing prerequisites produces an actionable error message within 1 second (the system does not hang or fail silently).
- **SC-005**: Interrupting any stage with Ctrl+C leaves all previously saved artifacts intact and uncorrupted.
- **SC-006**: The prediction service starts and becomes ready to accept requests within 10 seconds of running the prediction command.
