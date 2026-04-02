# Implementation Plan: Validate Card Embeddings

**Branch**: `014-validate-embeddings` | **Date**: 2026-04-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/014-validate-embeddings/spec.md`

## Summary

Add a `validate-embeddings` CLI subcommand that trains lightweight linear probes on top of frozen card embeddings to verify they encode the features Stage 2 training depends on (land detection, card color, pip counts, mana value, mana production). The command reports per-probe pass/fail scores and exits with code 0 (all pass), 1 (any fail), or 2 (input error). Ground truth extraction reuses existing `mana_scorer` parsing functions and `EmbeddingAdapter.is_land()` logic.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: scikit-learn 1.8.0 (LogisticRegression, LinearRegression, cross_val_score), numpy, existing `sealed.domain.mana_scorer` functions
**Storage**: Reads `.npz` embedding files and `.txt` card text files from `--cards-path` directory (no writes)
**Testing**: pytest (unit tests with synthetic fixtures, no real embeddings needed)
**Target Platform**: Windows / Linux CLI
**Project Type**: CLI subcommand addition to existing `python -m sealed` tool
**Performance Goals**: N/A — one-time validation step, runs in seconds on ~30k cards
**Constraints**: Must reuse existing parsing logic (FR-003). Probes must use cross-validation (FR-004).
**Scale/Scope**: ~30k card files in production `output/cardsfolder/`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit tests for ground truth extraction, probe logic, CLI wiring. All use synthetic data — no real embeddings or sklearn training in fast suite. Integration test (actual sklearn fit) kept separate. |
| II. Simplicity First | PASS | Linear probes are the simplest possible validation. sklearn's built-in cross_val_score handles evaluation. No custom ML framework needed. |
| III. Data Integrity | PASS | Ground truth reuses validated `mana_scorer` parsers. Cross-validation prevents train-on-test leakage. Minimum card count enforced (50 cards). |
| IV. Domain-Driven Design | PASS | Probe logic (ground truth extraction, probe specs) in domain layer. Orchestration in application layer. CLI parsing in infrastructure. Domain depends only on numpy/sklearn (scientific libs, not infrastructure). |
| V. MTG Forge Interoperability | N/A | Validation CLI tool — not exposed via remote API or Java stub. |
| VI. Documentation | PASS | New CLI command will be documented. Quickstart covers usage. |

**Gate result: PASS** — No violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/014-validate-embeddings/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/sealed/
├── domain/
│   └── embedding_probe.py       # NEW: ground truth extraction, probe definitions, runner
├── application/
│   └── validate_embeddings.py   # NEW: use case orchestration (load cards, run probes, return results)
└── infrastructure/
    └── cli.py                   # MODIFIED: add validate-embeddings subcommand + handler

tests/
├── unit/sealed/
│   ├── domain/
│   │   └── test_embedding_probe.py     # NEW: ground truth extraction, mana value, probe specs
│   └── infrastructure/
│       └── test_cli_sealed_validate.py # NEW: CLI argument parsing and handler wiring
└── integration/sealed/
    └── test_validate_embeddings_integration.py  # NEW: end-to-end with real sklearn fit on synthetic data
```

**Structure Decision**: Follows existing sealed module layering (domain/application/infrastructure). One new domain file for probe logic, one new application file for orchestration, CLI extension in existing cli.py. Mirrors the pattern used by encode-cards, train, and sample commands.

## Complexity Tracking

No constitution violations — this section is empty.
