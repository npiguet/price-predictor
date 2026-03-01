# Tasks: Forge API Integration

**Input**: Design documents from `/specs/002-forge-api-integration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/rest-api.md, contracts/connector-api.md, contracts/cli.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Python service**: `src/price_predictor/`, `tests/` at repository root
- **Java connector**: `forge-connector/src/main/java/com/pricepredictor/connector/`, `forge-connector/src/test/java/com/pricepredictor/connector/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add Python dependencies and create the Java Maven project skeleton

- [ ] T001 Add `fastapi` and `uvicorn[standard]` to dependencies in `pyproject.toml`
- [ ] T002 Create `forge-connector/` Maven project structure: `pom.xml` (groupId `com.pricepredictor`, artifactId `forge-connector`, Java 17, JUnit 5 dependency in test scope), `src/main/java/com/pricepredictor/connector/`, and `src/test/java/com/pricepredictor/connector/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Refactor the Forge parser to support text-based parsing — required by the REST endpoint (US1)

**CRITICAL**: The server endpoint in US1 receives Forge script text over HTTP. The existing `parse_forge_file()` reads from disk. A text-based parser function must exist before the endpoint can be implemented.

### Tests for Foundational Phase (MANDATORY per Constitution)

- [ ] T003 Write tests for `parse_forge_text()` in `tests/unit/infrastructure/test_forge_parser.py` — test parsing a complete Forge card script string (Name, ManaCost, Types, Oracle, K: lines, PT), a partial script (only Types), an empty string, and a malformed string (no Types line). Verify the function returns a Card entity or raises a descriptive error. Reuse existing fixture card scripts from `tests/fixtures/forge_cards/` loaded as strings

### Implementation for Foundational Phase

- [ ] T004 Refactor `src/price_predictor/infrastructure/forge_parser.py` — extract the line-parsing logic from `parse_forge_file()` into a new `parse_forge_text(text: str) -> Card` function. Update `parse_forge_file()` to call `parse_forge_text(path.read_text())`. No behavior change to existing code — this is a mechanical extraction. All existing `test_forge_parser.py` tests must still pass

**Checkpoint**: `parse_forge_text()` available and tested. US1 can now build the server endpoint on top of it.

---

## Phase 3: User Story 1 — Price Prediction Available as a Network Service (Priority: P1) MVP

**Goal**: Expose the trained prediction model as a REST service at `POST /api/v1/evaluate`, started via `python -m price_predictor serve`

**Independent Test**: Start the service, `curl -X POST http://localhost:8000/api/v1/evaluate -H "Content-Type: text/plain" -d "Name:Lightning Bolt\nManaCost:R\nTypes:Instant"`, verify a JSON response with `predicted_price_eur` and `model_version`

### Tests for User Story 1 (MANDATORY per Constitution)

- [ ] T005 [P] [US1] Write unit tests for the evaluate endpoint in `tests/unit/infrastructure/test_server.py` — use FastAPI `TestClient` with a mock model artifact. Test cases: (1) valid complete Forge script returns 200 with `predicted_price_eur` (float) and `model_version` (string), (2) partial script (only Types line) returns 200, (3) empty body returns 400 with error message, (4) unparseable body (no Types line) returns 400 with descriptive error, (5) request with made-up card returns 200. Verify response content-type is `application/json`
- [ ] T006 [P] [US1] Write unit tests for serve command in `tests/unit/infrastructure/test_cli_serve.py` — test cases: (1) `serve` subcommand is registered in argparse, (2) `--model-path`, `--host`, `--port` flags exist with correct defaults (models/latest.joblib, 0.0.0.0, 8000), (3) serve handler exits with code 2 and prints error to stderr when model file does not exist (FR-012), (4) serve handler calls uvicorn.run with correct host/port when model loads successfully (mock uvicorn.run)

### Implementation for User Story 1

- [ ] T007 [US1] Create FastAPI application in `src/price_predictor/infrastructure/server.py` — define `create_app(model_artifact)` factory function that returns a FastAPI app. Implement `POST /api/v1/evaluate` route: read raw body as text/plain, call `parse_forge_text()`, build a Card entity, call `PredictPriceUseCase.execute()`, return JSON `{"predicted_price_eur": float, "model_version": str}`. On parse error return 400 `{"error": str}`. On unexpected error return 500 `{"error": str}`. Model artifact is stored in `app.state` and accessed in the route handler
- [ ] T008 [US1] Add `serve` subcommand to `src/price_predictor/infrastructure/cli.py` — register `serve` subparser with `--model-path` (default: models/latest.joblib), `--host` (default: 0.0.0.0), `--port` (default: 8000). Implement `run_serve(args)` handler: load model via `model_store.load_model()`, if ModelNotFoundError print error to stderr and `sys.exit(2)`, otherwise create app via `create_app(artifact)` and call `uvicorn.run(app, host=args.host, port=args.port)`
- [ ] T009 [US1] Update `src/price_predictor/__main__.py` — add `serve` to the command dispatch. Configure logging to stderr for the serve command (same pattern as train/evaluate)
- [ ] T010 [US1] Write integration test in `tests/integration/test_server_integration.py` — start the FastAPI app with `TestClient` using a real trained model fixture (from conftest.py), send a valid Forge card script, verify response contains a numeric EUR price. Then send the same card attributes through `PredictPriceUseCase.execute()` directly and verify the prices are identical (FR-011: service MUST produce identical results to standalone model). Assert single-request response time is under 3 seconds (SC-001)
- [ ] T011 [US1] Write concurrency test in `tests/integration/test_server_integration.py` — using `TestClient` and `concurrent.futures.ThreadPoolExecutor`, send 10 concurrent requests with different valid Forge card scripts. Verify all 10 return 200 with valid JSON, no errors, and all complete within 3 seconds (FR-009, SC-004). This confirms uvicorn's thread pool handles concurrent predictions correctly

**Checkpoint**: `python -m price_predictor serve` starts the service. `POST /api/v1/evaluate` accepts Forge script text and returns EUR price estimates. Service fails fast without a trained model. Predictions match standalone model output. 10 concurrent requests handled without degradation. All Python tests pass.

---

## Phase 4: User Story 2 — MTG Forge Accesses Predictions via Lightweight Connector (Priority: P2)

**Goal**: Provide a Java 17+ Maven library that Forge can use to get price predictions from the service with 5 lines of code

**Independent Test**: Build the connector JAR, start the Python prediction service, write a small Java program that creates a `PricePredictorClient`, calls `predict()` with card attributes, and prints the EUR price

### Tests for User Story 2 (MANDATORY per Constitution)

- [ ] T012 [P] [US2] Write unit tests for ForgeScriptSerializer in `forge-connector/src/test/java/com/pricepredictor/connector/ForgeScriptSerializerTest.java` — test cases: (1) full card attributes produce correct Forge script text with Name, ManaCost, Types (supertypes + types + subtypes joined), Oracle, K: lines, PT lines, (2) partial attributes (only types) produce minimal valid script, (3) null optional fields are omitted, (4) keywords each get a separate K: line, (5) power/toughness serialize as PT:power/toughness
- [ ] T013 [P] [US2] Write unit tests for CardAttributes builder in `forge-connector/src/test/java/com/pricepredictor/connector/CardAttributesTest.java` — test cases: (1) builder with types set builds successfully, (2) builder without types throws IllegalStateException, (3) all optional fields can be set and retrieved, (4) types varargs and List overloads both work, (5) built object is immutable
- [ ] T014 [P] [US2] Write unit tests for response parsing in `forge-connector/src/test/java/com/pricepredictor/connector/ResponseParsingTest.java` — test cases: (1) parse valid success JSON `{"predicted_price_eur": 2.35, "model_version": "20260301-143000"}` into PriceEstimate record, (2) parse error JSON `{"error": "message"}` and extract error message, (3) handle malformed JSON gracefully (throw InvalidResponseException)

### Implementation for User Story 2

- [ ] T015 [P] [US2] Create PriceEstimate record in `forge-connector/src/main/java/com/pricepredictor/connector/PriceEstimate.java` — Java record with `double predictedPriceEur` and `String modelVersion`
- [ ] T016 [P] [US2] Create exception hierarchy: `PricePredictorException.java` (base, extends Exception, two constructors: message-only and message+cause), `ServiceUnavailableException.java` (extends PricePredictorException — connection refused, timeout), `InvalidResponseException.java` (extends PricePredictorException — non-200 status or unparseable response) in `forge-connector/src/main/java/com/pricepredictor/connector/`
- [ ] T017 [P] [US2] Create CardAttributes with builder in `forge-connector/src/main/java/com/pricepredictor/connector/CardAttributes.java` — immutable class with fields per data-model.md (name, manaCost, types, supertypes, subtypes, oracleText, keywords, power, toughness, loyalty). Static `builder()` method returns Builder inner class. Builder validates types is non-null and non-empty on `build()`. Provide varargs and List overloads for list fields
- [ ] T018 [US2] Create ForgeScriptSerializer in `forge-connector/src/main/java/com/pricepredictor/connector/ForgeScriptSerializer.java` — package-private class with `static String serialize(CardAttributes card)` method. Produces Forge card script text per research.md R4 rules: Name:, ManaCost:, Types: (supertypes + types + subtypes space-joined), Oracle:, K: lines for each keyword, PT:power/toughness, Loyalty:. Omit lines for null/empty fields
- [ ] T019 [US2] Create PricePredictorClient in `forge-connector/src/main/java/com/pricepredictor/connector/PricePredictorClient.java` — three constructors per connector-api.md contract (default, custom URL, custom URL+timeout). `predict(CardAttributes card)` method: serialize card via ForgeScriptSerializer, send HTTP POST to `/api/v1/evaluate` with text/plain body using `java.net.http.HttpClient`, parse JSON response into PriceEstimate. On 200: return PriceEstimate. On 400/500: throw InvalidResponseException with server's error message. On connection error: throw ServiceUnavailableException. On timeout: throw ServiceUnavailableException. Default timeout: 5000ms
- [ ] T020 [US2] Write unit tests for PricePredictorClient in `forge-connector/src/test/java/com/pricepredictor/connector/PricePredictorClientTest.java` — use JDK's `com.sun.net.httpserver.HttpServer` as a test helper (no external deps). Test cases: (1) successful prediction against mock server returning valid JSON → returns PriceEstimate, (2) mock server returning 400 → throws InvalidResponseException with error message, (3) mock server returning 500 → throws InvalidResponseException, (4) verify request sends correct Content-Type text/plain and POST method

**Checkpoint**: `mvn package` produces a JAR. PricePredictorClient can connect to the running Python service and return price estimates. All Java tests pass. JAR has zero external dependencies.

---

## Phase 5: User Story 3 — Graceful Degradation When Service Unavailable (Priority: P3)

**Goal**: The connector handles service unavailability gracefully — no hanging, no crashing, clear error indications within 5 seconds, automatic recovery when service returns

**Independent Test**: Stop the prediction service, call `predict()` from the connector, verify it throws `ServiceUnavailableException` within 5 seconds. Then start the service again and verify the next `predict()` call succeeds without recreating the client

### Tests for User Story 3 (MANDATORY per Constitution)

- [ ] T021 [P] [US3] Write timeout and connection-failure tests in `forge-connector/src/test/java/com/pricepredictor/connector/GracefulDegradationTest.java` — test cases: (1) connection to a closed port throws ServiceUnavailableException within 6 seconds (5s timeout + margin), (2) mock server that sleeps 10 seconds triggers timeout → ServiceUnavailableException, (3) configurable timeout via constructor is respected (e.g., 1 second timeout, mock sleeps 3 seconds), (4) verify exception message contains useful context (URL, timeout duration)
- [ ] T022 [P] [US3] Write recovery tests in `forge-connector/src/test/java/com/pricepredictor/connector/GracefulDegradationTest.java` — test cases: (1) mock server starts, client succeeds, mock server stops, client throws ServiceUnavailableException, mock server restarts, client succeeds again — all without recreating PricePredictorClient, (2) verify PricePredictorClient does not cache connection state that prevents recovery

### Implementation for User Story 3

- [ ] T023 [US3] Verify and refine timeout enforcement in `forge-connector/src/main/java/com/pricepredictor/connector/PricePredictorClient.java` — ensure `HttpClient` connect timeout and `HttpRequest` request timeout are both set to the configured value. Ensure `ConnectException`, `HttpTimeoutException`, and `IOException` are all caught and wrapped in `ServiceUnavailableException` with descriptive messages including the endpoint URL and timeout value
- [ ] T024 [US3] Verify PricePredictorClient creates a fresh HTTP request per `predict()` call — ensure no connection pooling or cached state prevents recovery after service restart. The `HttpClient` instance can be reused (it handles reconnection internally), but verify this works in the recovery test scenario

**Checkpoint**: Connector never hangs or crashes regardless of service availability. Timeout fires within 5 seconds. Service recovery works without client restart. All Java tests pass.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and cross-cutting quality checks

- [ ] T025 [P] Update `README.md` at project root — add section for the `serve` command (usage, flags, examples) per contracts/cli.md. Document relationship between features 001 (model) and 002 (service)
- [ ] T026 [P] Create `forge-connector/README.md` — document Maven coordinates, usage example (5-line code snippet from connector-api.md), build instructions (`mvn package`), error handling contract (exception types and scenarios), and requirements (Java 17+, prediction service running)
- [ ] T027 Verify JAR size is under 1 MB (SC-002) — run `mvn package` and check `forge-connector/target/forge-connector-1.0.0-SNAPSHOT.jar` file size
- [ ] T028 Run `specs/002-forge-api-integration/quickstart.md` validation checklist — execute all checklist items end-to-end: start service, curl endpoint, build JAR, verify connector predict, verify error handling, run all tests. This end-to-end validation satisfies the constitution quality gate "Remote API contract tests MUST pass for both stub and server" by verifying real HTTP communication between the Python service and Java connector

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS US1 (server needs `parse_forge_text`)
- **US1 (Phase 3)**: Depends on Foundational — BLOCKS US2 (connector needs running service for integration testing)
- **US2 (Phase 4)**: Depends on Setup (Maven project from T002). Unit tests (T012–T014) and implementation (T015–T018) can proceed in parallel with US1. Only T020 (client integration tests) needs the Python service from US1
- **US3 (Phase 5)**: Depends on US2 (extends the connector with degradation tests)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Depends on Foundational (Phase 2) only — no dependency on other stories
- **US2 (P2)**: Java types and serialization (T012–T018) can proceed after Setup. Client tests (T020) need US1 service running
- **US3 (P3)**: Extends US2 connector with error-handling tests and refinements

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models/types before services
- Services before client integration
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**: T001 (Python deps) and T002 (Maven project) can run in parallel

**Phase 3 (US1)**: T005 (server tests) and T006 (CLI tests) can run in parallel

**Phase 4 (US2)**: T012, T013, T014 (all test files) can run in parallel. T015, T016, T017 (types with no cross-dependencies) can run in parallel

**Phase 5 (US3)**: T021 and T022 (separate test scenarios) can run in parallel

**Phase 6 (Polish)**: T025 and T026 (different README files) can run in parallel

---

## Parallel Example: User Story 2

```text
# Launch all US2 tests together (different files):
Task T012: "ForgeScriptSerializerTest.java"
Task T013: "CardAttributesTest.java"
Task T014: "ResponseParsingTest.java"

# Launch all US2 type implementations together (no cross-dependencies):
Task T015: "PriceEstimate.java"
Task T016: "Exception hierarchy (3 files)"
Task T017: "CardAttributes.java"

# Then sequentially:
Task T018: "ForgeScriptSerializer.java" (depends on T017 CardAttributes)
Task T019: "PricePredictorClient.java" (depends on T015, T016, T018)
Task T020: "PricePredictorClientTest.java" (depends on T019 + US1 service)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T002)
2. Complete Phase 2: Foundational (T003–T004)
3. Complete Phase 3: User Story 1 (T005–T011)
4. **STOP and VALIDATE**: Start service, curl endpoint, verify predictions
5. Service is usable by any HTTP client — MVP delivered

### Incremental Delivery

1. Setup + Foundational → Parser refactor ready
2. Add US1 → Test independently → Python service operational (MVP!)
3. Add US2 → Test independently → Java connector works with service
4. Add US3 → Test independently → Connector handles failures gracefully
5. Polish → Documentation and validation complete
6. Each story adds integration value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Java connector tests (T012–T014, T020) use JDK's `com.sun.net.httpserver.HttpServer` as test helper — no external mock library needed
- FR-009/SC-004 (10 concurrent requests without degradation) is verified by T011 — concurrency test with ThreadPoolExecutor
- Python server tests (T005, T010) use FastAPI's `TestClient` — no real HTTP server needed
- The `parse_forge_text()` refactor (T004) is the critical enabler: it bridges the existing file-based parser to HTTP text-body input
- FR-011 (identical predictions) is verified by T010 — same card through both standalone model and service
- FR-012 (fail-fast without model) is verified by T006 — serve command exits with error when model not found
