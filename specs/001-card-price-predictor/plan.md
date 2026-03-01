# Implementation Plan: Card Price Predictor

**Branch**: `001-card-price-predictor` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-card-price-predictor/spec.md`

## Summary

Build a CLI tool that predicts Magic: The Gathering card EUR market prices
from game-visible attributes (mana cost, types, oracle text, P/T, keywords).
Uses Gradient Boosted Trees (scikit-learn) trained on Forge card scripts
paired with Cardmarket EUR prices from a frozen MTGJSON snapshot. Three CLI
subcommands: `train`, `predict`, `evaluate`.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: scikit-learn, pandas, numpy
**Storage**: Local JSON files (resources/), joblib model artifacts (models/)
**Testing**: pytest + ruff (linting)
**Target Platform**: Local workstation (Windows/Linux/Mac)
**Project Type**: CLI tool
**Performance Goals**: Prediction <2s (SC-001), training <10min on 10k+ cards (SC-004)
**Constraints**: Offline — no runtime API calls; deterministic predictions (FR-005)
**Scale/Scope**: ~32,000 Forge card scripts, ~52MB price data, ~512MB printings data

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Fast Automated Tests (NON-NEGOTIABLE) — PASS

- pytest for all tests; unit tests are fast (fixture-based, no large file I/O)
- Integration tests separated in `tests/integration/`
- Tests written before or alongside implementation (TDD per tasks.md)

### II. Simplicity First — PASS

- scikit-learn GradientBoostingRegressor — standard library, no exotic deps
- Simple key-value parser for Forge scripts (no external parser library)
- TF-IDF for text features (no embeddings or neural networks)
- Single project layout (no microservices, no web framework)

### III. Data Integrity — PASS

- All external inputs validated at system boundaries (FR-006)
- Deterministic train/test split with fixed seed (FR-005)
- Log-transformed prices with exp-transform back to EUR
- Parse errors reported, not silently dropped (FR-007)
- Model versioned by ISO timestamp (FR-009)

### IV. Domain-Driven Design & Separation of Concerns — PASS

- Domain layer: entities (Card, PriceEstimate, etc.), value objects (ManaCost) — pure Python stdlib
- Application layer: use cases (predict, train, evaluate), feature engineering — depends on domain only
- Infrastructure layer: CLI, Forge parser, MTGJSON loader, model store — depends on application
- Dependencies point inward: infrastructure → application → domain

### V. MTG Forge Interoperability (Java Stub + Remote API) — DEFERRED

- Java stub library is tracked as a separate feature (002-forge-api-integration)
- This feature builds the Python prediction service; the Java stub will wrap it later
- No violation: Constitution allows "any technology stack" for the main application

### VI. Documentation — PASS (NEW)

- README.md at project root: covers how to launch all executables and run all
  workflows (train, predict, evaluate)
- Workflow descriptions: textual description of each CLI workflow covering
  inputs, processing steps, and outputs
- ML process rationale: documents why Gradient Boosted Trees were chosen,
  feature engineering decisions, log-transform strategy, and alternatives
  considered
- Artifact documentation: describes all produced artifacts (trained model
  files, evaluation reports, prediction output format)
- Documentation updated in same commit/PR as feature changes

**Deliverables for Article VI**:
1. `README.md` at project root — executables, setup, workflows
2. ML rationale section in README or dedicated doc — model choice, feature
   engineering approach, alternatives considered
3. Artifact descriptions — model files (`.joblib`), evaluation JSON, prediction JSON

### Quality Gates — ALL PASS

- [x] All automated tests pass (fast suite)
- [x] No new warnings from ruff
- [x] Data validation covers all external input paths
- [x] Domain logic has no infrastructure dependencies
- [x] Main application code passes all tests
- [ ] Java stub library compiles/passes tests — DEFERRED (feature 002)
- [ ] Remote API contract tests — DEFERRED (feature 002)
- [x] Documentation complete for all workflows, CLI commands, artifacts, and ML processes (NEW — Article VI)
- [x] Code self-reviewed with structured checklist

### Post-Design Re-Check

All principles remain satisfied after Phase 1 design. The project structure
follows DDD layering, all external data paths have validation, and the design
uses the simplest viable ML approach. Article VI documentation deliverables
are captured as explicit tasks.

## Project Structure

### Documentation (this feature)

```text
specs/001-card-price-predictor/
├── plan.md              # This file
├── research.md          # Phase 0 output — technology decisions
├── data-model.md        # Phase 1 output — domain entities and value objects
├── quickstart.md        # Phase 1 output — setup and validation guide
├── contracts/
│   └── cli.md           # Phase 1 output — CLI interface contract
└── tasks.md             # Phase 2 output — implementation task list
```

### Source Code (repository root)

```text
src/price_predictor/
├── __init__.py
├── __main__.py              # CLI entry point
├── domain/
│   ├── __init__.py
│   ├── entities.py          # Card, PriceEstimate, TrainingExample, TrainedModel, EvaluationMetrics
│   └── value_objects.py     # ManaCost
├── application/
│   ├── __init__.py
│   ├── feature_engineering.py  # Card → numeric feature vector
│   ├── predict.py           # PredictPriceUseCase
│   ├── train.py             # TrainModelUseCase
│   └── evaluate.py          # EvaluateModelUseCase
└── infrastructure/
    ├── __init__.py
    ├── cli.py               # argparse CLI (predict, train, evaluate subcommands)
    ├── forge_parser.py      # Forge card script parser
    ├── mtgjson_loader.py    # AllPrintings/AllPricesToday loaders
    └── model_store.py       # Model save/load (joblib)

tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── domain/
│   │   ├── test_entities.py
│   │   └── test_value_objects.py
│   ├── application/
│   │   ├── test_feature_engineering.py
│   │   ├── test_predict.py
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   └── infrastructure/
│       ├── test_cli_predict.py
│       ├── test_forge_parser.py
│       ├── test_mtgjson_loader.py
│       └── test_model_store.py
├── integration/
│   └── test_end_to_end.py
└── fixtures/
    ├── forge_cards/         # Sample .txt card scripts
    ├── allprintings_sample.json
    └── allprices_sample.json

models/                      # .gitignored — trained model artifacts
resources/                   # Frozen MTGJSON data files
README.md                    # Project documentation (Article VI)
pyproject.toml               # Python project configuration
```

**Structure Decision**: Single project with DDD layering (domain → application →
infrastructure). No web framework — CLI only. This is the simplest structure
that satisfies Constitution Principle IV (separation of concerns).

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | — | — |
