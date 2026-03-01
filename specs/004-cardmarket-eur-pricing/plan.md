# Implementation Plan: CardMarket EUR Pricing

**Branch**: `004-cardmarket-eur-pricing` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-cardmarket-eur-pricing/spec.md`

## Summary

This feature formalizes the existing CardMarket EUR pricing behavior as explicit requirements and adds one new capability: reporting the count of cards excluded due to missing CardMarket prices (FR-008).

The codebase already implements FR-001–FR-006, FR-009, and FR-010:
- `build_price_map()` and `get_cheapest_price()` in `mtgjson_loader.py` navigate `paper → cardmarket → retail → {normal, foil}` and use the most recent date snapshot
- No TCGPlayer, USD, or other vendor references exist in the Python source
- All domain entities (`TrainingExample`, prediction output) already use EUR field names (`actual_price_eur`, `predicted_price_eur`)

The single new code change is an INFO-level log line in `build_price_map()` reporting how many cards from `name_to_uuids` had no CardMarket price and were therefore excluded from the price map.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: No new dependencies (existing: scikit-learn, pandas, numpy, joblib)
**Storage**: Local JSON files (AllPrintings.json, AllPricesToday.json), joblib model files
**Testing**: pytest + ruff
**Target Platform**: Local CLI (Windows/Linux/macOS)
**Project Type**: CLI / ML training pipeline
**Performance Goals**: N/A — single log line addition has negligible performance impact
**Constraints**: Must not break existing training pipeline behavior
**Scale/Scope**: ~30k card names in MTGJSON; exclusion count = len(name_to_uuids) - len(result)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Justification |
|-----------|--------|---------------|
| I. Fast Automated Tests | PASS | New log line covered by unit test. Existing tests verify CardMarket EUR extraction. No slow tests added. |
| II. Simplicity First | PASS | One log line added to one existing function. No new abstractions, modules, or dependencies. |
| III. Data Integrity | PASS | Exclusion count is derived deterministically from existing data (len difference). No data transformation changes. |
| IV. Domain-Driven Design | PASS | Change is in infrastructure layer (logging). No domain entity changes. |
| V. MTG Forge Interoperability | N/A | No changes to prediction API, Java stub, or external interfaces. |
| VI. Documentation | PASS | quickstart.md documents the new log output. Feature 001 spec already uses EUR/CardMarket terminology. |

**Quality Gates**: All pass. No violations.

### Post-Design Re-Check (after Phase 1)

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit test for exclusion count log. Existing tests validate CardMarket EUR behavior. |
| II. Simplicity First | PASS | Single log line in `build_price_map()`. No other code changes. |
| III. Data Integrity | PASS | Exclusion count is a simple subtraction, deterministic and reproducible. |
| IV. Domain-Driven Design | PASS | Infrastructure layer only. Domain entities unchanged. |
| V. MTG Forge Interoperability | N/A | No API changes. |
| VI. Documentation | PASS | quickstart.md created. |

**Post-design gate**: PASS — no violations found.

## Project Structure

### Documentation (this feature)

```text
specs/004-cardmarket-eur-pricing/
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
│   └── entities.py              # No change — already uses EUR field names
├── application/
│   └── train.py                 # No change — consumes price map transparently
└── infrastructure/
    └── mtgjson_loader.py        # FR-008: add exclusion count log line in build_price_map()

tests/
├── unit/
│   └── infrastructure/
│       └── test_mtgjson_loader.py   # New test: exclusion count logging
└── integration/
    └── test_end_to_end.py           # Existing — verify no regression
```

**Structure Decision**: Single project (existing). The only code change is one log line in `build_price_map()` and one new test.

## Complexity Tracking

> No violations — table not applicable.
