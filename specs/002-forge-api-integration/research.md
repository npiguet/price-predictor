# Research: Forge API Integration

**Date**: 2026-03-01
**Feature**: 002-forge-api-integration

## R1: Python Web Framework for REST Service

### Decision: FastAPI + uvicorn

**Rationale**: FastAPI is a lightweight Python web framework that fits
naturally into the existing project. It provides automatic request
validation (helps with FR-010), supports synchronous route handlers
(prediction is CPU-bound, no async benefit), and uvicorn's default
port is 8000 — matching the spec 005 endpoint default. The framework
adds two dependencies (`fastapi`, `uvicorn`) and requires minimal
boilerplate.

**Concurrency model**: uvicorn runs a single-process async event loop
by default. Synchronous route handlers (our case — `joblib.load` and
`model.predict` are blocking) are executed in a thread pool. This
handles SC-004 (10 concurrent requests) without configuration.

**Alternatives considered**:
- Flask + waitress — simpler but no automatic validation. waitress
  handles threading on Windows. Slightly more boilerplate for
  validation. Rejected: FastAPI is equally simple and gives free
  validation.
- stdlib `http.server` — too bare-bones. No routing, no validation,
  no concurrent request handling without manual threading. Rejected.
- Flask dev server (`threaded=True`) — not production-grade. Rejected.

## R2: Java HTTP Client for Connector

### Decision: `java.net.http.HttpClient` (built-in since Java 11)

**Rationale**: FR-004 requires the connector depend only on standard
runtime libraries. `java.net.http.HttpClient` is part of the JDK
since Java 11 and provides synchronous and asynchronous HTTP support,
configurable timeouts, and proper error handling — everything the
connector needs with zero external dependencies.

**Timeout handling**: `HttpClient.newBuilder().connectTimeout()`
for connection timeout, plus `HttpRequest.Builder.timeout()` for
per-request timeout. Both map directly to FR-007 (configurable
timeout, default 5 seconds).

**Alternatives considered**:
- Apache HttpClient — mature, but adds a dependency (~800 KB).
  Violates FR-004 and inflates connector size. Rejected.
- OkHttp — modern, but adds a dependency (~400 KB + Kotlin stdlib).
  Rejected for same reason.
- `java.net.HttpURLConnection` — legacy API, verbose error handling,
  no built-in timeout on the request level. Rejected.

## R3: Java Project Structure

### Decision: Maven project at `forge-connector/` in repo root

**Rationale**: The constitution requires "a Java 17+ stub library
that MTG Forge can consume as a standard Maven/Gradle dependency."
Forge itself is a Maven project. A separate top-level directory
keeps Java and Python code cleanly separated. The connector has
no external dependencies, so the Maven POM is minimal.

**Maven coordinates**:
- Group ID: `com.pricepredictor`
- Artifact ID: `forge-connector`
- Version: `1.0.0-SNAPSHOT`
- Package: `com.pricepredictor.connector`

**Build**: `mvn package` produces a JAR. No shade plugin needed
(zero external dependencies). The JAR can be installed locally
with `mvn install` or copied directly onto Forge's classpath.

**Alternatives considered**:
- Gradle — Forge uses Maven. Matching the build tool simplifies
  integration. Rejected.
- Subdirectory of `src/` — mixing Java and Python source in the
  same tree is confusing. Rejected.
- Separate repository — adds repository management overhead for
  a single small library. Rejected per Simplicity First.

## R4: Forge Script Serialization in Java

### Decision: Simple key-value text builder in the connector

**Rationale**: The clarified spec says the connector serializes card
attributes into Forge card script syntax and sends them to
`POST /api/v1/evaluate` as `text/plain`. The Forge script format is
simple key:value lines — no complex grammar. A dedicated serializer
class converts `CardAttributes` to text.

**Format produced**:
```
Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target.
```

**Rules**:
- Each non-null attribute writes one line: `Key:Value`
- `types`, `supertypes`, `subtypes` join with space into a single
  `Types:` line (supertypes first, then types, then subtypes)
- `keywords` each write a separate `K:` line
- `power`/`toughness` join as `PT:power/toughness`
- Null/empty fields are omitted (server handles partial input)
- No `ALTERNATE` section (single card only)

**Alternatives considered**:
- JSON serialization to a separate endpoint — rejected in
  clarification. One endpoint for all consumers.
- Forge's own serialization classes — would couple the connector
  to Forge's internals. The connector must be independent. Rejected.

## R5: Forge Script Text Parsing (Python Service Side)

### Decision: Extract `parse_forge_text(text: str) -> Card` from existing `forge_parser.py`

**Rationale**: The existing `parse_forge_file(path)` reads a file and
then parses its contents. The REST endpoint receives Forge script text
in the request body (not a file path). Refactoring to extract the
text-parsing logic into a separate function avoids duplication and
lets both the file-based and HTTP-based paths share the same parser.

**Approach**:
1. Extract the line-parsing logic from `parse_forge_file` into a new
   `parse_forge_text(text: str) -> Card` function
2. Have `parse_forge_file` call `parse_forge_text(path.read_text())`
3. The REST endpoint calls `parse_forge_text(request_body)`

This is a mechanical refactor — no behavior change to existing code.

**Alternatives considered**:
- Duplicate parsing logic in the server module — violates DRY and
  risks divergence. Rejected.
- Pass a file-like object (StringIO) to the existing function —
  unnecessary indirection. Rejected.

## R6: Model Loading Strategy

### Decision: Load model once at startup, share across all requests

**Rationale**: The trained model artifact (scikit-learn pipeline +
feature engineering) is read-only after loading. Loading takes ~1
second; prediction takes milliseconds. Loading per-request would add
unacceptable latency. Loading once at startup and sharing across
requests via FastAPI dependency injection is the standard pattern.

**Implementation**: The `serve` command loads the model via
`model_store.load_model()` before starting uvicorn. If loading fails
(model file not found), the command exits immediately with an error
(FR-012). The loaded artifact is stored as application state and
injected into the route handler.

**Thread safety**: scikit-learn's `predict()` is thread-safe for
read-only inference. Multiple concurrent requests can share the same
model instance without locks.

**Alternatives considered**:
- Lazy load on first request — delays the error signal. FR-012
  requires fail-fast at startup. Rejected.
- Load per-request — too slow (~1s per request vs ~10ms). Rejected.
- Global variable — works but harder to test. FastAPI app state
  is slightly cleaner. Acceptable alternative.
