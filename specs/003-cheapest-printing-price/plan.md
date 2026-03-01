# Implementation Plan: Cheapest Printing Price for Training

**Branch**: `003-cheapest-printing-price` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-cheapest-printing-price/spec.md`

## Summary

When a card has multiple printings with different prices, the system must select the cheapest price as the training label. The existing codebase already implements the `min(all_prices)` selection logic in `build_price_map()`. This feature adds three concrete refinements:

1. **Price floor at €0.01** — prices below €0.01 (including €0.00) are clamped to €0.01 before use as a training label, preventing log-domain errors during model training.
2. **Allow zero-priced printings into comparison** — change the existing `price > 0` filter to `price is not None and price >= 0` so that zero-priced printings participate in the cheapest-price selection (and are then clamped by the floor).
3. **Multi-printing transparency logging** — log per-card price selection details and a summary of how many cards required price selection, via INFO-level stderr messages.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: scikit-learn, pandas, numpy, joblib (no new dependencies)
**Storage**: Local JSON files (AllPrintings.json, AllPricesToday.json), joblib model files
**Testing**: pytest + ruff
**Target Platform**: Local CLI (Windows/Linux/macOS)
**Project Type**: CLI / ML training pipeline
**Performance Goals**: SC-004: cheapest-price selection adds no more than 30 seconds to training data preparation
**Constraints**: Must not break existing training pipeline behavior for cards with single printings
**Scale/Scope**: ~30k card names in MTGJSON, ~15k with valid prices; ~13k matched to Forge card scripts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Justification |
|-----------|--------|---------------|
| I. Fast Automated Tests | PASS | All new behavior covered by unit tests. No slow integration tests added. Existing test suite runs in seconds. |
| II. Simplicity First | PASS | Minimal changes to 1 existing file (`mtgjson_loader.py`). No new abstractions, no new modules. Logging added inline where data is already available. |
| III. Data Integrity | PASS | Price floor at €0.01 is deterministic and prevents log(0). Zero-price handling is explicitly defined. Selection is reproducible given same input. |
| IV. Domain-Driven Design | PASS | Price floor is a domain rule applied in the infrastructure layer at the data loading boundary (appropriate — it's a data normalization step applied at ingestion, not a domain entity rule). No new cross-layer dependencies. |
| V. MTG Forge Interoperability | N/A | This feature modifies the training pipeline only. No changes to the prediction API, Java stub, or any external interface. |
| VI. Documentation | PASS | quickstart.md describes the new behavior. No new CLI commands or output formats. |

**Quality Gates**: All pass. No violations.

### Post-Design Re-Check (after Phase 1)

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit tests for price floor and logging. No new integration tests. |
| II. Simplicity First | PASS | Changes touch only `mtgjson_loader.py` (primary) and test files. No new abstractions, modules, or dependencies introduced. |
| III. Data Integrity | PASS | Price floor is deterministic. Logging is informational only (does not affect data flow). |
| IV. Domain-Driven Design | PASS | All changes remain in the infrastructure layer. Domain entities unchanged. |
| V. MTG Forge Interoperability | N/A | No API or stub changes. |
| VI. Documentation | PASS | quickstart.md created. No new CLI commands. |

**Post-design gate**: PASS — no violations found.

## Project Structure

### Documentation (this feature)

```text
specs/003-cheapest-printing-price/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/price_predictor/
├── domain/
│   └── entities.py              # TrainingExample (no change — 0.01 > 0 satisfies existing validation)
├── application/
│   └── train.py                 # No changes needed (consumes price map transparently)
└── infrastructure/
    └── mtgjson_loader.py        # PRIMARY CHANGE: price floor + transparency logging

tests/
├── unit/
│   └── infrastructure/
│       └── test_mtgjson_loader.py   # New tests: price floor, zero-price, multi-printing logging
├── fixtures/
│   └── allprices_sample.json        # Add edge-case entries (zero-price, sub-cent price)
└── integration/
    └── test_end_to_end.py           # Existing — verify no regression
```

**Structure Decision**: Single project (existing). All changes are within the existing `src/price_predictor/infrastructure/` and `tests/` directories. No new modules or packages needed.

## Complexity Tracking

> No violations — table not applicable.
