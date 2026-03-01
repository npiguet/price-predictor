# Tasks: Card Evaluation Endpoints

**Input**: Design documents from `/specs/005-card-eval-endpoints/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/evaluate.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root (relative to `src/` working directory for imports)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No new setup required. Project structure, dependencies, and tooling are already in place from features 001–004. The Forge parser, FastAPI endpoint, and CLI framework all exist.

**No tasks in this phase.**

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No new foundational work. All blocking prerequisites (prediction model, Forge parser, FastAPI app, model store) are fully implemented.

**No tasks in this phase.**

**Checkpoint**: Foundation ready — user story implementation can begin immediately.

---

## Phase 3: User Story 1 — Get Price Evaluation via Network Endpoint (Priority: P1) 🎯 MVP

**Goal**: Add structured request logging (FR-012) to the existing `/api/v1/evaluate` endpoint. The endpoint already handles FR-001 through FR-006 and FR-011 — this story adds the missing observability layer.

**Independent Test**: Send a valid Forge card script to the endpoint, verify a price estimate is returned AND a structured JSON log entry is emitted to stderr with timestamp, status code, latency, card attributes, and prediction result.

### Tests for User Story 1 (MANDATORY per Constitution) ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T001 [P] [US1] Write unit tests for structured request logging in tests/unit/infrastructure/test_server.py — test that a successful evaluate request emits a JSON log with fields: event, timestamp, status_code, latency_ms, card_name, card_types, card_mana_cost, predicted_price_eur, model_version. Test that a failed request (400) emits a JSON log with fields: event, timestamp, status_code, latency_ms, error. Test that latency_ms is a positive number. Use caplog or mock logger to capture log output.

### Implementation for User Story 1

- [ ] T002 [US1] Add structured request logging to the evaluate endpoint handler in src/price_predictor/infrastructure/server.py — import `time` and `json` and `datetime`. Wrap the existing handler logic: capture `start = time.perf_counter()` before processing, compute `latency_ms` after. On success (200), log a JSON dict with event="evaluate_request", ISO timestamp, status_code=200, latency_ms, card_name, card_types, card_mana_cost (from the parsed Card's fields before ManaCost is parsed — use the raw ManaCost string from the Forge text fields), predicted_price_eur, model_version. On parse error (400), log event, timestamp, status_code=400, latency_ms, error message. On prediction error (500), log event, timestamp, status_code=500, latency_ms, error message. Use `logger.info(json.dumps(log_entry))` for single-line structured output.

**Checkpoint**: Endpoint returns correct responses (already verified by existing tests) AND emits structured logs for every request. T001 tests pass.

---

## Phase 4: User Story 2 — Get Price Evaluation via CLI Tool (Priority: P2)

**Goal**: Create a thin CLI `eval` subcommand that reads a Forge card script file from disk and sends its contents to the running prediction endpoint via HTTP, displaying the result to the user.

**Independent Test**: Create a Forge card script file, start the prediction service, run `python -m price_predictor eval <file>`, and verify the price estimate and model version are displayed on stdout.

**Depends on**: US1 (endpoint must be running for CLI to work), but CLI implementation is independent of the logging changes.

### Tests for User Story 2 (MANDATORY per Constitution) ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T003 [P] [US2] Write unit tests for the eval subcommand in tests/unit/infrastructure/test_cli_eval.py — mock `urllib.request.urlopen` to test without a real server. Test cases: (1) successful evaluation displays "Predicted price: €X.XX" and "Model version: ..." on stdout, returns exit code 0; (2) file not found prints "Error: File not found: <path>" to stderr, returns exit code 1; (3) directory path prints error about expecting a file, returns exit code 1; (4) endpoint unreachable (URLError) prints "Error: Could not connect to prediction service at <url>" to stderr, returns exit code 2; (5) endpoint returns 400 error prints "Error: Prediction service returned error (400): <message>" to stderr, returns exit code 2; (6) endpoint returns 500 error prints error to stderr, returns exit code 2; (7) `--endpoint` flag overrides default URL. Use tmp_path for test fixture files, use existing forge_cards fixtures.

### Implementation for User Story 2

- [ ] T004 [US2] Add `eval` subcommand to the argument parser in src/price_predictor/infrastructure/cli.py — add a new subparser "eval" with: positional argument "file" (the path to the Forge card script file), optional `--endpoint` flag (default: "http://localhost:8000/api/v1/evaluate"). Help text should say "Evaluate a card from a Forge script file via the prediction service".

- [ ] T005 [US2] Implement `run_eval()` function in src/price_predictor/infrastructure/cli.py — (1) resolve the file path, check it exists and is a file (not directory), print error to stderr and return 1 if not; (2) read file contents as UTF-8; (3) send POST request to endpoint using `urllib.request.Request` with Content-Type text/plain; (4) on success, parse JSON response and print "Predicted price: €{price}" and "Model version:   {version}" to stdout, return 0; (5) on `urllib.error.URLError` (connection refused etc.), print connection error to stderr, return 2; (6) on `urllib.error.HTTPError`, read error body, parse JSON, print "Error: Prediction service returned error ({status}): {message}" to stderr, return 2.

- [ ] T006 [US2] Add eval command dispatch to src/price_predictor/__main__.py — import `run_eval` from cli module, add `elif args.command == "eval":` branch that calls `run_eval(args)` and returns the result. No logging configuration needed for eval (it's a thin client).

- [ ] T007 [US2] Write integration test for eval CLI → endpoint in tests/integration/test_eval_integration.py — use the existing `trained_model_dir` fixture pattern. Start a TestClient, mock `urllib.request.urlopen` to route to the TestClient instead of real HTTP. Test: (1) eval with a valid forge card fixture file produces stdout matching endpoint response; (2) eval with a malformed file relays the endpoint's 400 error; (3) SC-006 verification: price from CLI matches price from direct endpoint call for the same card script.

**Checkpoint**: CLI eval subcommand reads a file, sends to endpoint, displays result. All error paths handled. T003 and T007 tests pass.

---

## Phase 5: User Story 3 — Forge Script Parsing and Validation (Priority: P3)

**Goal**: Verify that the existing Forge parser (already implemented in features 001/002) satisfies all US3 acceptance scenarios and edge cases defined in the spec.

**Independent Test**: Feed various Forge card scripts (complete, partial, malformed, edge cases) into `parse_forge_text()` and verify extracted attributes match expected values.

**Note**: The parser at `src/price_predictor/infrastructure/forge_parser.py` is already fully implemented. This phase focuses on test coverage verification against spec acceptance scenarios.

### Tests for User Story 3 (MANDATORY per Constitution) ✅

- [ ] T008 [P] [US3] Verify and extend parser acceptance scenario tests in tests/unit/infrastructure/test_forge_parser.py — ensure dedicated test cases exist for each US3 scenario: (1) complete script with Name, ManaCost, Types, SubTypes, PT, Oracle, Keywords → all fields extracted correctly; (2) script with extra irrelevant fields (e.g., "SetInfo:M21", "ArtCredit:John Doe") → extra fields ignored, card parsed normally; (3) land card with no ManaCost → mana_cost is None, card parses successfully; (4) non-Forge content (random text, JSON) → raises ValueError with descriptive message. Also verify edge cases from spec: (5) multi-face card with ALTERNATE section → only front face parsed; (6) empty string input → raises ValueError("Card script text is empty").

**Checkpoint**: All parser behavior verified against spec. No implementation changes needed.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation updates and cross-cutting validation.

- [ ] T009 Update README.md with eval subcommand documentation — add `eval` to the CLI commands section, document usage syntax (`python -m price_predictor eval <file> [--endpoint URL]`), describe structured request logging behavior and log format for the serve command.
- [ ] T010 Run quickstart.md validation — verify all commands in specs/005-card-eval-endpoints/quickstart.md work as documented: serve starts, curl returns response, eval subcommand works, error cases produce expected messages.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Skipped — no setup needed
- **Foundational (Phase 2)**: Skipped — no new foundational work
- **US1 (Phase 3)**: Can start immediately — builds on existing endpoint
- **US2 (Phase 4)**: Can start immediately (independent of US1 logging changes). Integration test (T007) requires endpoint to work (already does).
- **US3 (Phase 5)**: Can start immediately — parser already exists, this is verification only
- **Polish (Phase 6)**: Depends on US1 + US2 completion

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories — adds logging to existing endpoint
- **User Story 2 (P2)**: Runtime dependency on endpoint (P1) being available, but implementation is independent. Can be developed in parallel with US1.
- **User Story 3 (P3)**: No dependencies — parser already exists, this is test verification only

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation tasks are sequential within a story
- Story complete before moving to next priority (or parallel if capacity allows)

### Parallel Opportunities

- **T001** (US1 logging tests) and **T003** (US2 CLI tests) and **T008** (US3 parser tests): All in different files, can run in parallel
- **US1** and **US2** implementation: Different files (`server.py` vs `cli.py`), can proceed in parallel after their respective tests
- **US3**: Fully independent, can run any time

---

## Parallel Example: All Test Tasks

```text
# All three test tasks can be launched simultaneously (different files):
T001: "Write structured logging tests in tests/unit/infrastructure/test_server.py"
T003: "Write eval CLI tests in tests/unit/infrastructure/test_cli_eval.py"
T008: "Verify parser tests in tests/unit/infrastructure/test_forge_parser.py"
```

## Parallel Example: US1 + US2 Implementation

```text
# After their respective tests pass, US1 and US2 implementation can proceed in parallel:
Stream A (US1): T002 (server.py logging)
Stream B (US2): T004 → T005 → T006 (cli.py + __main__.py) → T007 (integration test)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete T001 (logging tests) → T002 (logging implementation)
2. **STOP and VALIDATE**: Endpoint returns prices AND logs structured entries
3. Existing endpoint tests + new logging tests all pass

### Incremental Delivery

1. US1: Structured logging → Endpoint fully observable (MVP)
2. US2: CLI eval → Users can evaluate cards from files
3. US3: Parser verification → Full test coverage confirmed
4. Polish: README + quickstart validation → Documentation complete

### Parallel Strategy

With capacity for parallel work:
1. Launch T001, T003, T008 simultaneously (all tests)
2. T002 (US1 impl) and T004→T005→T006 (US2 impl) in parallel
3. T007 (US2 integration) after US2 impl
4. T009, T010 (polish) after US1 + US2

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- FR-001 through FR-006, FR-011 are already implemented — no tasks needed
- FR-012 → US1 (T001, T002)
- FR-007 through FR-010 → US2 (T003–T007)
- US3 is verification-only — parser already exists from features 001/002
- No new dependencies added — uses stdlib `urllib.request` per research.md R-001
- Structured logs use `json.dumps()` per research.md R-002
- CLI subcommand named `eval` per research.md R-003
