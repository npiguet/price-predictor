# Implementation Plan: Forge API Integration

**Branch**: `002-forge-api-integration` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-forge-api-integration/spec.md`

## Summary

Expose the card price prediction model as a REST service and provide a
lightweight Java connector library for MTG Forge integration. The Python
service adds a `serve` subcommand that starts a FastAPI/uvicorn server
on port 8000, serving the `POST /api/v1/evaluate` endpoint (shared with
feature 005). The Java connector is a zero-dependency Maven library that
serializes card attributes into Forge script text, sends them to the
service, and returns the price estimate. The connector handles timeouts,
connection failures, and error responses gracefully so Forge remains
stable regardless of service availability.

## Technical Context

**Language/Version**: Python 3.14+ (service), Java 17+ (connector)
**Primary Dependencies**: FastAPI, uvicorn (Python); no external deps (Java — uses java.net.http)
**Storage**: Same as feature 001 — joblib model files in `models/`
**Testing**: pytest (Python), JUnit 5 (Java)
**Target Platform**: Windows/Linux workstation
**Project Type**: REST service + Java library
**Performance Goals**: <3s response (SC-001), 10 concurrent requests (SC-004)
**Constraints**: Connector <1 MB (SC-002), 5s timeout default (FR-007), fail-fast without model (FR-012)
**Scale/Scope**: Local/single machine, single-card requests only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | pytest for Python service tests (unit + integration). JUnit 5 for Java connector tests. Both suites run in seconds. |
| II. Simplicity First | PASS | FastAPI is minimal boilerplate. Connector is a thin HTTP client. No unnecessary abstractions. |
| III. Data Integrity | PASS | Service reuses existing prediction pipeline — identical results (FR-011). Input validated at service boundary (FR-010). |
| IV. DDD & Separation | PASS | Server is infrastructure layer (wraps application use case). Domain logic unchanged. Connector is a separate Java module. |
| V. Forge Interop | PASS | This is the feature. Java 17+ Maven library. Standard Java types. Handles all HTTP internally. Graceful error handling. |
| VI. Documentation | PASS | Quickstart covers serve command and connector usage. Contracts document REST API and Java API. |

**Post-Phase 1 re-check**: All gates still pass. Two-project structure
(Python + Java) is required by the constitution (Principle V mandates a
Java stub library while the prediction service uses Python). FastAPI adds
one infrastructure dependency — domain logic remains free of framework
imports. Connector has zero external dependencies.

## Project Structure

### Documentation (this feature)

```text
specs/002-forge-api-integration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── rest-api.md      # REST endpoint contract (shared with feature 005)
│   ├── connector-api.md # Java connector public API
│   └── cli.md           # serve subcommand contract
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/price_predictor/
├── __main__.py                      # Add serve command logging config
├── domain/                          # Unchanged
│   ├── entities.py
│   └── value_objects.py
├── application/
│   └── predict.py                   # Unchanged (reused by server)
└── infrastructure/
    ├── cli.py                       # Add serve subcommand + handler
    ├── server.py                    # NEW: FastAPI app, /api/v1/evaluate route
    ├── forge_parser.py              # Refactor: extract parse_forge_text()
    ├── model_store.py               # Unchanged
    └── mtgjson_loader.py            # Unchanged

tests/
├── unit/
│   └── infrastructure/
│       ├── test_server.py           # NEW: endpoint tests (TestClient)
│       ├── test_cli_serve.py        # NEW: serve command tests
│       └── test_forge_parser.py     # Extend: test parse_forge_text()
├── integration/
│   └── test_server_integration.py   # NEW: full server lifecycle test
└── fixtures/
    └── forge_cards/                 # Existing fixtures reused

forge-connector/                     # NEW: Java Maven project
├── pom.xml
├── src/
│   ├── main/java/com/pricepredictor/connector/
│   │   ├── PricePredictorClient.java
│   │   ├── CardAttributes.java
│   │   ├── PriceEstimate.java
│   │   ├── ForgeScriptSerializer.java
│   │   ├── PricePredictorException.java
│   │   ├── ServiceUnavailableException.java
│   │   └── InvalidResponseException.java
│   └── test/java/com/pricepredictor/connector/
│       ├── PricePredictorClientTest.java
│       ├── CardAttributesTest.java
│       ├── ForgeScriptSerializerTest.java
│       └── ResponseParsingTest.java
└── README.md
```

**Structure Decision**: Two projects — extend the existing Python project
(add server module to infrastructure layer) and create a new Java Maven
project at `forge-connector/`. This is the minimum structure required:
the constitution mandates a Java stub library (Principle V) while the
prediction service is Python (Principle V allows any-tech for the core
application). The Python project follows the existing DDD structure. The
Java project is a flat single-module Maven project with no external
dependencies.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Second project (Java) | Constitution Principle V mandates a Java 17+ Maven library for Forge | Single Python project cannot produce a Java artifact on Forge's classpath |
