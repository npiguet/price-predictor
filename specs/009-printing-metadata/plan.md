# Implementation Plan: Printing Data Fields in Training & Prediction

**Branch**: `009-printing-metadata` | **Date**: 2026-03-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/009-printing-metadata/spec.md`

## Summary

Enrich card text with five printing data fields (reserved, rarity, printings count, set code, legalities) derived from AllPrintings.json and AllPricesToday.json. The fields are appended as inline text lines using the existing `key: value` card text format. Both sklearn and transformer training pipelines are updated to enrich card text during data loading. The prediction API auto-fills fields for known cards and applies defaults for unknown cards, remaining fully backward-compatible.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: scikit-learn, pandas, numpy, joblib, PyTorch, transformers (BERT tokenizer), FastAPI, uvicorn, ijson
**Storage**: Local JSON files (AllPrintings.json, AllPricesToday.json), joblib model files (models/sklearn/), .pt model files (models/transformer/)
**Testing**: pytest (unit + integration), ruff (linting)
**Target Platform**: Windows / Linux (local development)
**Project Type**: CLI + web-service (ML training pipeline + REST prediction API)
**Performance Goals**: N/A (metadata enrichment is a data-loading concern; no latency-sensitive hot path)
**Constraints**: No new dependencies. Enrichment must work with both sklearn (Card entity features) and transformer (raw tokenized text) pipelines identically.
**Scale/Scope**: ~30k cards in AllPrintings, ~20k with prices

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit tests for metadata extraction, enrichment, parsing, feature engineering. Integration tests for enriched training pipeline. |
| II. Simplicity First | PASS | Extends existing patterns (key:value text lines, Card entity fields, FeatureEngineering dense features). No new abstractions beyond a `PrintingData` value object. |
| III. Data Integrity | PASS | Metadata validated at extraction (AllPrintings boundary). Defaults are explicit. Enriched text format is deterministic and tested with known cards. |
| IV. Domain-Driven Design | PASS | `PrintingData` is a domain value object. Enrichment logic lives in application layer. AllPrintings loading stays in infrastructure. |
| V. MTG Forge Interoperability | PASS | No Java stub changes. API is backward-compatible (existing requests work unchanged). |
| VI. Documentation | PASS | Updated card text format documented. New CLI options documented. |

## Project Structure

### Documentation (this feature)

```text
specs/009-printing-metadata/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── rest-api.md      # Updated predict endpoint
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/price_predictor/
├── domain/
│   ├── entities.py            # MODIFY: Add optional printing data fields to Card
│   └── value_objects.py       # MODIFY: Add PrintingData value object
├── application/
│   ├── card_enrichment.py     # NEW: Metadata enrichment logic (enrich text, apply defaults, parse)
│   ├── train.py               # MODIFY: Enrich cards with metadata during training
│   ├── train_transformer.py   # MODIFY: Enrich text with metadata during training
│   ├── evaluate.py            # MODIFY: Enrich cards with metadata during evaluation
│   ├── evaluate_transformer.py # MODIFY: Enrich text with metadata during evaluation
│   ├── feature_engineering.py # MODIFY: Add printing data features to sklearn feature vector
│   ├── predict.py             # MODIFY: Support enriched Card entities
│   └── predict_transformer.py # MODIFY: Support enriched text
├── infrastructure/
│   ├── mtgjson_loader.py      # MODIFY: New function to build metadata map (card→PrintingData)
│   ├── converted_card_parser.py # MODIFY: Recognize 5 new key:value fields → Card entity
│   ├── server.py              # MODIFY: Load metadata at startup, auto-fill/default at prediction time
│   └── cli.py                 # MODIFY: Pass printings/prices paths to serve command

tests/
├── unit/
│   ├── domain/
│   │   └── test_printing_data.py       # NEW: PrintingData value object tests
│   ├── application/
│   │   ├── test_card_enrichment.py     # NEW: Enrichment logic tests
│   │   ├── test_feature_engineering.py # MODIFY: Test new printing data features
│   │   ├── test_train.py              # MODIFY: Test enriched training pipeline
│   │   └── test_train_transformer.py  # MODIFY: Test enriched transformer training
│   └── infrastructure/
│       ├── test_converted_card_parser.py # MODIFY: Test parsing enriched card text
│       └── test_mtgjson_loader.py       # MODIFY: Test metadata map building
├── integration/
│   ├── test_end_to_end.py              # MODIFY: Enriched training pipeline
│   └── test_server_integration.py      # MODIFY: Auto-fill & defaults in prediction
└── fixtures/
    ├── allprintings_sample.json        # MODIFY: Add isReserved, rarity, legalities data
    └── allprices_sample.json           # (no change needed — already has UUID→price)
```

**Structure Decision**: Single-project layout matching existing architecture. No new top-level directories. One new application module (`card_enrichment.py`) and one new domain value object (`PrintingData`).

## Complexity Tracking

No constitution violations to justify — all gates pass cleanly.
