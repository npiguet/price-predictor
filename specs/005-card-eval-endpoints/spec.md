# Feature Specification: Card Evaluation Endpoints

**Feature Branch**: `005-card-eval-endpoints`
**Created**: 2026-02-27
**Status**: Draft
**Input**: User description: "Users of the application can get an evaluation of a card by passing its description in the CardForge script syntax to both a REST API Endpoint, or to a CLI tool by passing a the path of a file. The CLI tool should just call the REST API endpoint."
**Depends on**: `001-card-price-predictor` (prediction model), `004-cardmarket-eur-pricing` (EUR pricing)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Get Price Evaluation via Network Endpoint (Priority: P1)

A user sends a card description written in Forge card script syntax to a network-accessible endpoint and receives a price evaluation in return. The Forge card script syntax is the same key-value pair format used by MTG Forge card script files (containing fields such as Name, ManaCost, Types, PT, Oracle text, keywords, etc.). The user submits the script content directly in the request body, and the system parses it, extracts the card attributes, runs the prediction model, and returns a price estimate in EUR.

This is the foundational interface. All other access methods (including the CLI tool) depend on this endpoint being available.

**Why this priority**: The network endpoint is the single source of truth for card evaluation. The CLI tool (P2) delegates to it, and any future integrations (e.g., the Forge connector from feature 002) would also use it. Without this endpoint, no evaluation is possible.

**Independent Test**: Can be tested by sending a valid Forge card script as request content to the endpoint and verifying that a numeric EUR price estimate is returned.

**Acceptance Scenarios**:

1. **Given** the prediction service is running with a trained model, **When** a user sends a valid Forge card script (e.g., a script with Name, ManaCost, Types, and Oracle text) to the endpoint, **Then** the system returns a numeric EUR price estimate.
2. **Given** a Forge card script with partial attributes (e.g., only Name, ManaCost, and Types — no Oracle text or PT), **When** the user sends it to the endpoint, **Then** the system returns a price estimate using the available attributes.
3. **Given** a Forge card script describing a completely made-up card, **When** the user sends it to the endpoint, **Then** the system returns a price estimate (hypothetical cards are supported per feature 001).
4. **Given** a malformed or unparseable Forge card script, **When** the user sends it to the endpoint, **Then** the system returns a clear error message indicating what could not be parsed and why.
5. **Given** a valid Forge card script, **When** the user sends it to the endpoint, **Then** the response includes both the price estimate and the model version that produced it.

---

### User Story 2 - Get Price Evaluation via CLI Tool (Priority: P2)

A user has a Forge card script file on their local filesystem and wants to get a price evaluation without manually constructing a network request. They run a command-line tool, passing the path to the card script file as an argument. The CLI tool reads the file, sends its contents to the network endpoint (from P1), and displays the price estimate to the user.

The CLI tool is intentionally a thin client — it does not contain the prediction model or any parsing logic beyond reading the file. All evaluation work happens at the network endpoint.

**Why this priority**: The CLI tool provides a convenient developer/power-user interface but depends entirely on the network endpoint (P1). It adds user convenience without introducing independent prediction logic.

**Independent Test**: Can be tested by creating a Forge card script file, running the CLI tool with the file path, and verifying the tool displays a price estimate (with the network endpoint running).

**Acceptance Scenarios**:

1. **Given** a valid Forge card script file exists at a given path and the prediction service is running, **When** the user runs the CLI tool with that file path, **Then** the tool displays the EUR price estimate.
2. **Given** a file path that does not exist, **When** the user runs the CLI tool with that path, **Then** the tool displays a clear error message indicating the file was not found.
3. **Given** the file exists but contains a malformed Forge script, **When** the user runs the CLI tool, **Then** the tool relays the parsing error from the endpoint to the user.
4. **Given** the prediction service is not running, **When** the user runs the CLI tool, **Then** the tool displays a clear error indicating the prediction service is unreachable.
5. **Given** a valid Forge card script file, **When** the user runs the CLI tool, **Then** the displayed output includes both the price estimate and the model version, matching what the endpoint would return.

---

### User Story 3 - Forge Script Parsing and Validation (Priority: P3)

The system correctly parses the Forge card script syntax and extracts the card attributes needed for price prediction. Forge card scripts are plain-text files using a specific key-value format (e.g., `Name:Lightning Bolt`, `ManaCost:R`, `Types:Instant`, `Oracle:Lightning Bolt deals 3 damage to any target.`). The system must understand this format, extract the relevant fields, and map them to the prediction model's expected input attributes.

When a script contains fields the model does not use, those fields are ignored. When a script is missing expected fields, the system uses the partial attributes available (consistent with feature 001's graceful handling of partial input).

**Why this priority**: Correct parsing underpins both P1 and P2, but the parsing logic is a supporting capability rather than a user journey itself. It can be developed and tested with unit-level checks independent of the full endpoint or CLI.

**Independent Test**: Can be tested by feeding various Forge card scripts (complete, partial, malformed, edge cases) into the parser and verifying the extracted attributes match expected values.

**Acceptance Scenarios**:

1. **Given** a complete Forge card script with Name, ManaCost, Types, SubTypes, PT, Oracle, and Keywords fields, **When** the system parses it, **Then** all fields are correctly extracted and mapped to prediction model input attributes.
2. **Given** a Forge card script with extra fields not relevant to price prediction (e.g., art credits, set info), **When** the system parses it, **Then** the extra fields are ignored and prediction proceeds normally.
3. **Given** a Forge card script with no ManaCost line (e.g., a land card), **When** the system parses it, **Then** the mana cost attribute is treated as absent/empty, and prediction proceeds with the remaining attributes.
4. **Given** a file that is not in Forge card script syntax at all (e.g., a JSON file or random text), **When** the system attempts to parse it, **Then** a clear parsing error is returned indicating the content is not valid Forge script syntax.

---

### Edge Cases

- What happens when the Forge card script contains multi-face card data (e.g., transform, split, or adventure cards with ALTERNATE sections)? The system evaluates only the primary face. Multi-face pricing based on combined attributes is out of scope.
- What happens when the Forge card script uses encoding that differs from the system's expected encoding (e.g., UTF-16 vs. UTF-8)? The system assumes UTF-8; other encodings produce a parsing error with a descriptive message.
- What happens when the CLI tool is given a directory path instead of a file path? The tool reports an error indicating a file path is expected.
- What happens when the Forge card script file is extremely large (e.g., multiple cards concatenated)? The system processes only the first card definition found in the file. Processing multiple cards from a single file is out of scope for this feature.
- What happens when the network endpoint receives an empty request body? The system returns a validation error indicating that card script content is required.
- What happens when the CLI tool successfully connects to the endpoint but the endpoint returns an unexpected error? The CLI tool displays the error message from the endpoint without crashing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a network-accessible endpoint that accepts card descriptions in Forge card script syntax and returns a price evaluation.
- **FR-002**: The endpoint MUST parse the Forge card script syntax, extract card attributes (name, mana cost, card types, subtypes, power/toughness, oracle text, keyword abilities), and use them as input to the prediction model.
- **FR-003**: The endpoint MUST return the price estimate in EUR (consistent with feature 004) along with the model version that produced the estimate.
- **FR-004**: The endpoint MUST return descriptive error messages for malformed, unparseable, or empty input, indicating which part of the script could not be processed.
- **FR-005**: The endpoint MUST handle partial Forge scripts gracefully — missing fields are treated as absent attributes, and prediction proceeds with whatever attributes are available.
- **FR-006**: The endpoint MUST ignore Forge script fields that are not relevant to price prediction (e.g., set information, art references, alternate card faces).
- **FR-007**: The system MUST provide a command-line tool that accepts a file path as its argument, reads the Forge card script from that file, and sends its contents to the network endpoint for evaluation.
- **FR-008**: The CLI tool MUST NOT contain prediction logic. It MUST delegate all evaluation to the network endpoint.
- **FR-009**: The CLI tool MUST display the price estimate and model version returned by the endpoint.
- **FR-010**: The CLI tool MUST display clear error messages when: the file does not exist, the file is unreadable, the endpoint is unreachable, or the endpoint returns an error.
- **FR-011**: The endpoint MUST validate the Forge script content before passing it to the prediction model and MUST reject clearly invalid input (e.g., empty body, binary data) with an appropriate error.

### Key Entities

- **Forge Card Script**: A plain-text description of a card in MTG Forge's key-value script format. Contains fields like Name, ManaCost, Types, SubTypes, PT, Oracle, Keywords, among others. This is the input format for both the endpoint and the CLI tool.
- **Evaluation Request**: A submission of Forge card script content to the network endpoint for price prediction. Contains the raw script text.
- **Evaluation Response**: The result returned by the network endpoint. Contains: predicted price (numeric, EUR), model version (identifier), and optionally the parsed attributes used for prediction.
- **CLI Tool**: A command-line utility that reads a Forge card script from a file and delegates evaluation to the network endpoint. It is a thin client with no prediction logic.

## Assumptions

- The Forge card script syntax is well-documented and stable. The format is the same one used by MTG Forge for its 32,000+ card definitions (one `.txt` file per card, key-value pairs separated by colons or similar delimiters).
- The prediction model (feature 001) accepts structured card attributes. This feature adds a parsing layer that converts Forge script syntax into those structured attributes — it does not modify the model itself.
- The network endpoint from this feature is the same service described in feature 002. This feature specifies the concrete input format (Forge script syntax) and the two user-facing interfaces (endpoint + CLI).
- The CLI tool requires the prediction service to be running. It does not support offline or embedded prediction.
- Only one card per request/file is supported. Batch evaluation of multiple cards from a single file is out of scope (batch support via the endpoint may be added in a separate feature).
- The CLI tool outputs results to standard output in a human-readable format. Machine-readable output (e.g., structured data) may be added later but is not required for this feature.
- Error messages from the endpoint are suitable for display to users — the CLI tool relays them directly without reformatting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user submitting a valid Forge card script to the endpoint receives a price estimate within 3 seconds.
- **SC-002**: The CLI tool completes a full evaluation cycle (read file, send to endpoint, display result) within 5 seconds for a standard single-card script file.
- **SC-003**: 100% of Forge card script fields relevant to price prediction (name, mana cost, types, subtypes, P/T, oracle text, keywords) are correctly parsed and passed to the model, verified against a sample of 20 real Forge card scripts.
- **SC-004**: Malformed input produces a descriptive error message in 100% of cases — no silent failures, no empty responses, no unhandled crashes.
- **SC-005**: The CLI tool correctly reports all error conditions (missing file, unreachable service, endpoint error) with user-friendly messages, verified across all defined error scenarios.
- **SC-006**: The price estimate returned via the CLI tool is identical to the estimate returned by the endpoint directly for the same card script (100% consistency — the CLI is a pure pass-through).
