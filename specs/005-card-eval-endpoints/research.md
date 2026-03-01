# Research: Card Evaluation Endpoints

**Feature**: 005-card-eval-endpoints
**Date**: 2026-03-01

## R-001: HTTP Client Library for CLI Tool

**Context**: The CLI `eval` subcommand must send HTTP POST requests to the prediction endpoint. Need to choose an HTTP client that aligns with the "no new dependencies" constraint.

**Decision**: Use `urllib.request` from Python stdlib.

**Rationale**:
- Zero additional dependencies — consistent with the project's existing dependency footprint and the spec header ("No new dependencies")
- The request is trivial: a single POST with `text/plain` body, reading a JSON response
- `urllib.request.Request` + `urllib.request.urlopen` handles this in ~10 lines
- Error handling maps cleanly: `URLError` for connection failures, `HTTPError` for non-200 responses

**Alternatives considered**:
- `httpx`: Modern async-capable HTTP client, but adds a dependency for a single synchronous call. Overkill.
- `requests`: Popular but also an unnecessary dependency. The stdlib is sufficient.
- `http.client`: Lower-level than `urllib.request`, requires more boilerplate for the same result.

## R-002: Structured Logging Format for FR-012

**Context**: FR-012 requires each endpoint request to be logged with: timestamp, HTTP status code, response latency, parsed card attributes, and prediction result. Logs must be structured (e.g., JSON-formatted).

**Decision**: Use Python's `logging` module with `json.dumps()` to emit structured log entries as single-line JSON strings via the existing stderr logger.

**Rationale**:
- The project already uses `logging.basicConfig(stream=sys.stderr, level=logging.INFO)` — structured logs integrate naturally
- Single-line JSON strings are grep-friendly and machine-parseable
- No need for a dedicated JSON logging library (e.g., `python-json-logger`) — a dict → `json.dumps()` call is sufficient
- Latency measured via `time.perf_counter()` (monotonic, high-resolution)

**Log entry format**:
```json
{
  "event": "evaluate_request",
  "timestamp": "2026-03-01T14:30:00.123456",
  "status_code": 200,
  "latency_ms": 42.5,
  "card_name": "Lightning Bolt",
  "card_types": ["Instant"],
  "card_mana_cost": "R",
  "predicted_price_eur": 2.35,
  "model_version": "20260301-143000"
}
```

For error cases:
```json
{
  "event": "evaluate_request",
  "timestamp": "2026-03-01T14:30:01.654321",
  "status_code": 400,
  "latency_ms": 1.2,
  "error": "No Name field found in card script"
}
```

**Alternatives considered**:
- `python-json-logger`: Adds a dependency for formatting that `json.dumps()` handles trivially.
- Middleware-based logging: FastAPI middleware could log all requests, but this endpoint needs domain-specific fields (card name, predicted price) that middleware can't access.

## R-003: CLI Subcommand Naming

**Context**: The existing CLI has subcommands: `train`, `predict`, `evaluate`, `serve`. Need a name for the new "evaluate card from file via endpoint" subcommand.

**Decision**: Use `eval` as the subcommand name.

**Rationale**:
- Short and memorable: `python -m price_predictor eval card.txt`
- Distinct from `evaluate` (which evaluates model accuracy on a test set, not a single card)
- The semantic difference is clear: `eval` = evaluate one card, `evaluate` = evaluate model performance
- Consistent with common CLI conventions (e.g., `terraform plan` vs `terraform apply`)

**Alternatives considered**:
- `eval-card`: More explicit but adds a hyphen for a common operation
- `eval-file`: Describes the input mechanism, not the action — less intuitive
- `check`: Ambiguous (check what?)

## R-004: Binary/Invalid Input Detection

**Context**: FR-011 requires rejecting "clearly invalid input (e.g., empty body, binary data)" with an appropriate error.

**Decision**: Rely on the existing UTF-8 decode + Forge parser validation chain. No additional binary detection needed.

**Rationale**:
- The endpoint already decodes the body as UTF-8: `body = (await request.body()).decode("utf-8")`
- If the body contains binary data, UTF-8 decode produces garbled text
- The Forge parser then fails with a descriptive ValueError ("No Name field found" or "No Types field found")
- This chain produces clear error messages for all invalid inputs:
  - Empty body → `ValueError("Card script text is empty")`
  - Binary data → Parser fails on garbled text → descriptive error
  - Random text → No Name/Types → descriptive error
- Adding explicit binary detection (e.g., checking for null bytes) would be over-engineering for a local/internal endpoint

**Alternatives considered**:
- Explicit null-byte check before parsing: Catches a narrow edge case but adds code for minimal benefit. The parser already handles it.
- Content-Type validation: Could reject non-`text/plain` requests, but the spec says the body is always text/plain and the endpoint is internal-use only.

## R-005: Existing Implementation Gap Analysis

**Context**: Need to understand exactly what code changes are required vs. what already exists.

**Findings**:

| Requirement | Status | Gap |
|-------------|--------|-----|
| FR-001: POST /api/v1/evaluate endpoint | ✅ Exists | None |
| FR-002: Parse Forge script, extract attributes | ✅ Exists | None |
| FR-003: Return EUR price + model version | ✅ Exists | None |
| FR-004: Descriptive error messages | ✅ Exists | None (parser raises descriptive ValueErrors) |
| FR-005: Handle partial scripts | ✅ Exists | None (Card allows None for optional fields) |
| FR-006: Ignore irrelevant fields | ✅ Exists | None (parser only extracts known fields) |
| FR-007: CLI tool with file path + --endpoint | ❌ Missing | New `eval` subcommand needed |
| FR-008: CLI must not contain prediction logic | N/A | By design — CLI sends HTTP request |
| FR-009: CLI displays price + model version | ❌ Missing | Part of eval subcommand |
| FR-010: CLI error messages | ❌ Missing | Part of eval subcommand |
| FR-011: Validate before prediction | ✅ Exists | None (parser validates, empty check exists) |
| FR-012: Structured request logging | ❌ Missing | Add to server.py evaluate handler |

**Summary**: 2 implementation units:
1. **Structured logging** in `server.py` (FR-012)
2. **CLI eval subcommand** in `cli.py` + `__main__.py` (FR-007 through FR-010)
