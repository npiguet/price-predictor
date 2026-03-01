# Feature Specification: MTG Forge Price Prediction Integration

**Feature Branch**: `002-forge-api-integration`
**Created**: 2026-02-27
**Status**: Draft
**Input**: User description: "The interoperability with MTG forge does not mean that the whole project must be able run within the forge process. It is acceptable to have a library on the MTG forge classpath that make a remote call (for example, via a REST API) to a remote server that hosts the application."
**Depends on**: `001-card-price-predictor` (trained prediction model)

## Clarifications

### Session 2026-03-01

- Q: Should the service return USD or EUR? → A: EUR. The model is trained on EUR prices from Cardmarket (updated in feature 001). No currency conversion needed — EUR is the native currency throughout.
- Q: Should the connector send Forge card scripts (text/plain) to `/api/v1/evaluate`, or structured JSON to a separate endpoint? → A: The connector serializes card attributes into Forge card script text and uses the same `POST /api/v1/evaluate` endpoint (text/plain) defined in feature 005. One endpoint for all consumers.
- Q: Should batch prediction (FR-006) remain given spec 005 scopes batch out? → A: No. Remove batch support — connector only supports single-card requests, matching the single-card endpoint.
- Q: How should the prediction service be started? → A: Python CLI command — add a `serve` subcommand to the existing CLI (e.g., `python -m price_predictor serve`), consistent with feature 001's `train` and `evaluate` subcommands.
- Q: Should the service expose a health check endpoint? → A: No. The connector detects availability by attempting a prediction request — no dedicated health endpoint.
- Q: What happens when the service starts but no trained model exists? → A: Fail fast — the service refuses to start and prints an error indicating no trained model was found.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Price Prediction Available as a Network Service (Priority: P1)

The card price prediction capability (trained model from feature 001) is exposed as an independent, network-accessible service. External applications can request price predictions by sending card attributes over the network and receiving price estimates in response, without needing to embed the prediction model or its dependencies in their own process.

This is the foundational capability that enables all integration scenarios. The service accepts the same card attributes defined in feature 001 (mana cost, card types, permanent types, oracle text, power/toughness, keyword abilities) and returns an EUR price estimate.

**Why this priority**: Without a network-accessible prediction service, no external application can integrate with the price predictor. This is the prerequisite for all integration stories.

**Independent Test**: Can be tested by starting the prediction service, sending a card attributes request over the network, and verifying a numeric price estimate is returned.

**Acceptance Scenarios**:

1. **Given** a trained prediction model exists and the service is started, **When** an external client sends a valid set of card attributes over the network, **Then** the service returns a numeric EUR price estimate.
2. **Given** the service is running, **When** a client sends a request with partial card attributes (e.g., only card type and mana cost), **Then** the service returns a price estimate using the available information, consistent with feature 001 behavior.
3. **Given** the service is running, **When** a client sends a request with attributes for a made-up card, **Then** the service returns a price estimate (same behavior as feature 001 — made-up cards are supported).
4. **Given** the service is running, **When** a client sends an invalid or malformed request, **Then** the service returns a clear error message indicating what was wrong with the request.

---

### User Story 2 - MTG Forge Accesses Price Predictions via Lightweight Connector (Priority: P2)

MTG Forge users can access card price predictions from within the Forge application. A lightweight connector module — compatible with Forge's classpath and extension mechanism — handles communication with the prediction service. The connector is a thin layer that sends card attributes to the prediction service and returns results; it does not include the prediction model itself or any of its heavyweight dependencies.

This architecture means Forge's startup time, memory usage, and dependency tree are minimally affected. The connector is designed to be dropped into Forge's classpath without requiring modifications to Forge's core codebase.

**Why this priority**: This delivers the integration value the user described — Forge users get price predictions without the prediction system running inside the Forge process. Depends on P1 (the service) being available.

**Independent Test**: Can be tested by adding the connector to a simulated Forge-like classpath environment, starting the prediction service, and invoking the connector to retrieve a price prediction.

**Acceptance Scenarios**:

1. **Given** the prediction service is running and the connector module is on the application's classpath, **When** the connector is invoked with a card's attributes, **Then** it returns the price estimate from the prediction service.
2. **Given** the connector is on the classpath, **When** a caller requests a prediction for a single card, **Then** the connector sends the request to the prediction service and returns the result without requiring any other dependencies beyond standard runtime libraries.
3. **Given** the connector is on the classpath, **When** a caller invokes the connector with a card's attributes, **Then** the connector serializes the attributes into Forge card script syntax, sends them to the prediction service endpoint, and returns the price estimate without the caller needing to handle HTTP details.

---

### User Story 3 - Graceful Degradation When Service Unavailable (Priority: P3)

When the prediction service is not running, unreachable, or experiencing errors, the connector clearly communicates the unavailability to the calling application. The calling application (e.g., Forge) does not crash, hang, or experience unrecoverable errors. Price prediction is a supplementary feature — its absence should not disrupt the host application's primary functionality.

**Why this priority**: Resilience is essential for a good integration experience, but the core value (P1 + P2) can be delivered and demonstrated without sophisticated error handling. This story ensures production-quality robustness.

**Independent Test**: Can be tested by invoking the connector when the prediction service is deliberately stopped, and verifying the connector returns a clear unavailability indication within a reasonable timeout.

**Acceptance Scenarios**:

1. **Given** the prediction service is not running, **When** the connector attempts to retrieve a price prediction, **Then** it returns a clear "service unavailable" indication within 5 seconds (no hanging or blocking indefinitely).
2. **Given** the prediction service becomes unreachable mid-session, **When** the connector attempts a request, **Then** it detects the failure and returns an error indication without crashing the host application.
3. **Given** the prediction service was previously unavailable, **When** it becomes available again, **Then** subsequent connector requests succeed without requiring a restart of the host application.

---

### Edge Cases

- What happens when no trained model exists? The `serve` command fails fast at startup with a clear error message. The service never enters a running state without a loaded model.
- What happens when the prediction service is running but returns unexpectedly slow responses (e.g., due to system load)?
- How does the connector behave when the prediction service version changes (e.g., a new model version with different capabilities)?
- What happens when the network connection between connector and service is intermittent (e.g., dropping packets)?
- How does the system handle concurrent requests from multiple connectors or applications?
- What happens when card attributes sent via the connector contain characters or formats not expected by the prediction service (e.g., Unicode in oracle text)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose the card price prediction capability as an independent network-accessible service, started via a `serve` subcommand on the existing CLI (e.g., `python -m price_predictor serve`), separate from any consuming application's process.
- **FR-002**: The prediction service MUST expose the REST endpoint `POST /api/v1/evaluate` defined in feature 005, accepting card descriptions in Forge card script syntax as `text/plain` request body and returning a JSON response with a numeric EUR price estimate, consistent with the prediction behavior defined in feature 001.
- **FR-003**: System MUST provide a lightweight connector module that consuming applications can include on their classpath to communicate with the prediction service.
- **FR-004**: The connector MUST NOT require the consuming application to include the prediction model, training data, or heavyweight dependencies (e.g., machine learning libraries). It must depend only on standard runtime libraries.
- **FR-005**: The connector MUST support single-card prediction requests only (one set of card attributes in, one price estimate out). Batch prediction is out of scope, consistent with feature 005's single-card endpoint.
- *(FR-006 removed — batch support out of scope per clarification)*
- **FR-007**: The connector MUST return a clear error indication when the prediction service is unreachable, within a configurable timeout (default: 5 seconds). It MUST NOT block indefinitely.
- **FR-008**: The connector MUST NOT cause the host application to crash or enter an unrecoverable state, regardless of prediction service availability.
- **FR-009**: The prediction service MUST handle concurrent requests from multiple consumers without data corruption or incorrect results.
- **FR-010**: The prediction service MUST validate incoming requests and return descriptive error messages for malformed or incomplete requests.
- **FR-011**: The prediction service MUST produce identical results to the standalone prediction model — wrapping the model as a service MUST NOT alter prediction outcomes.
- **FR-012**: The `serve` command MUST fail fast with a descriptive error message if no trained model file is found at startup. The service MUST NOT start in a degraded state without a loaded model.

### Key Entities

- **Prediction Service**: An independent, network-accessible process started via `python -m price_predictor serve` that hosts the trained card price prediction model and responds to prediction requests at `POST /api/v1/evaluate`. Attributes: service status (running/stopped), loaded model version, service address (default `http://localhost:8000`).
- **Connector**: A lightweight module designed to be added to a consuming application's classpath (e.g., MTG Forge). It serializes card attributes into Forge card script syntax and sends them to the prediction service's `POST /api/v1/evaluate` endpoint. Attributes: service address configuration, timeout settings, connection status.
- **Prediction Request**: An HTTP `POST` to `/api/v1/evaluate` with `Content-Type: text/plain` body containing card attributes serialized as Forge card script text for a single card. Attributes: card attributes (same as feature 001).
- **Prediction Response**: A JSON response returned from the prediction service containing the price estimate for a single card. Attributes: predicted price (numeric, EUR), model version used, any error information.

## Assumptions

- The prediction service and consuming applications (e.g., MTG Forge) run on the same local machine or local network. Internet-scale deployment and associated concerns (load balancing, authentication, rate limiting) are out of scope.
- No authentication is required between the connector and the prediction service. This is a local development/personal-use tool, not a public-facing service.
- MTG Forge is a Java-based application, so the connector module must be compatible with Forge's classpath mechanism. However, the prediction service itself is not required to be written in the same language as Forge.
- The connector's only responsibility is relaying requests and responses. Any card attribute validation happens at the prediction service level, not in the connector.
- The prediction service loads a model trained by feature 001's training workflow. This feature does not modify or extend the training process itself.
- "Lightweight" means the connector module adds minimal size (target: under 1 MB) and minimal startup overhead to the host application.
- The prediction service runs on a single machine. Clustering and horizontal scaling are out of scope.
- The connector exposes a programmatic interface suitable for use by Forge developers or plugin authors. This feature does not include modifications to Forge's user interface — UI integration is a separate concern.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An external application receives a price prediction from the service within 3 seconds of sending a request (including network round-trip for local connections).
- **SC-002**: The connector module adds less than 1 MB of additional size to the consuming application's distribution.
- **SC-003**: When the prediction service is unavailable, the connector returns an error indication within 5 seconds — the host application never hangs or crashes due to prediction service unavailability.
- **SC-004**: The prediction service handles at least 10 concurrent requests without degradation in response time or accuracy.
- **SC-005**: Predictions returned through the service are identical to predictions made directly through the standalone model (100% consistency).
- *(SC-006 removed — batch support out of scope per clarification)*
- **SC-007**: The connector can be added to a consuming application's classpath and used to retrieve a prediction with no more than 5 lines of setup code.
