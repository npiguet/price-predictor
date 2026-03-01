# Implementation Plan: Card Price Predictor

**Branch**: `001-card-price-predictor` | **Date**: 2026-03-01 (revised) | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-card-price-predictor/spec.md`

## Summary

Build an ML-based card price predictor using Python + scikit-learn that
reads card attributes from MTG Forge card scripts, pairs them with EUR
Cardmarket prices from MTGJSON data files, trains a gradient boosted
regression model, and predicts EUR prices for any card (real or
made-up). The system exposes three CLI subcommands: `train`, `predict`,
and `evaluate`. Long-running operations (train, evaluate) display
human-readable progress messages on stderr so the user can see the
system is working.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: scikit-learn, pandas, numpy, joblib, ijson
**Storage**: Local JSON files (resources/), joblib model files (models/)
**Testing**: pytest, ruff
**Target Platform**: Windows/Linux workstation (CLI)
**Project Type**: CLI tool
**Performance Goals**: Training 10k+ cards < 10 minutes; prediction < 2 seconds
**Constraints**: Reproducible predictions (fixed random seed); stderr-only progress messages
**Scale/Scope**: ~32,000 Forge card scripts, ~52 MB price data, ~512 MB printings data

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | pytest unit tests run in <10s. Integration tests separated. |
| II. Simplicity First | PASS | scikit-learn GBR, Python logging stdlib. No unnecessary abstractions. |
| III. Data Integrity | PASS | Input validation at boundaries. Deterministic transforms. Fixed seed reproducibility. |
| IV. DDD & Separation | PASS | Domain (entities, value objects) → Application (use cases, feature engineering) → Infrastructure (CLI, parser, loader, model store). |
| V. Forge Interop | N/A | Java stub library is feature 002. This feature uses any-tech allowance for Python. |
| VI. Documentation | PASS | README exists. Progress logging improves operational visibility. |

**Post-Phase 1 re-check**: All gates still pass. Progress logging uses
stdlib `logging` (Simplicity First), routes to stderr (does not corrupt
stdout JSON output per Data Integrity), and is configured in the
infrastructure layer (DDD compliance).

## Project Structure

### Documentation (this feature)

```text
specs/001-card-price-predictor/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── cli.md           # CLI contract
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/price_predictor/
├── __main__.py                      # Entry point, logging configuration
├── domain/
│   ├── entities.py                  # Card, PriceEstimate, TrainedModel, etc.
│   └── value_objects.py             # ManaCost
├── application/
│   ├── feature_engineering.py       # Card → numeric vector
│   ├── train.py                     # TrainModelUseCase (with progress logging)
│   ├── evaluate.py                  # EvaluateModelUseCase (with progress logging)
│   └── predict.py                   # PredictPriceUseCase
└── infrastructure/
    ├── cli.py                       # Argument parsing, command dispatch
    ├── forge_parser.py              # Parse Forge card scripts (with progress logging)
    ├── mtgjson_loader.py            # Load AllPrintings/AllPricesToday (with progress logging)
    └── model_store.py               # Save/load model artifacts

tests/
├── unit/
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── integration/
└── fixtures/
    └── forge_cards/                 # Sample card scripts for testing
```

**Structure Decision**: Single Python project with domain/application/infrastructure
layers following DDD. All existing — the structure is already in place. Progress
logging is added to existing modules, not new files.
