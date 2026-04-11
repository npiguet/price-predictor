# Implementation Plan: Sealed Deck Scorer

**Branch**: `013-sealed-deck-scorer` | **Date**: 2026-04-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/013-sealed-deck-scorer/spec.md`
**Parent Spec**: [`specs/sealed-deck-picker.md`](../sealed-deck-picker.md)

## Summary

Train a Set Transformer-based deck scorer on pairwise match outcomes using Bradley-Terry loss. The scorer assigns a scalar quality score to any sealed deck (spells + non-basic lands only), enabling downstream search-based deck building. This feature extends the existing `encode-cards` command to produce 544-dimensional card vectors (512 text embedding + 32 deterministic game features), implements the scorer model and training loop, adds validation metrics, provides a Forge baseline evaluation pipeline, and supports embedding fine-tuning.

## Technical Context

**Language/Version**: Python 3.14+ (scorer, training, evaluation CLI), Java 17+ (evaluation workers)
**Primary Dependencies**: PyTorch (model architecture, training), numpy (embeddings, data loading); existing: scikit-learn, pandas, joblib, FastAPI, uvicorn; Java: forge-game 2.0.10-SNAPSHOT (already in pom.xml)
**Storage**: `.npz` embedding files (544-dim card vectors), `.pt` model checkpoints (scorer), `match-outcomes.txt` flat text (training data from feature 012)
**Testing**: pytest (Python unit + integration), JUnit 5 (Java)
**Target Platform**: Windows 11 (primary dev), Linux compatible
**Project Type**: CLI + ML training pipeline
**Performance Goals**: Greedy search ~1 second per deck (1500 candidate swaps, each a forward pass); evaluation time dominated by Forge match count (N² matches for N pools)
**Constraints**: Variable-length deck input (20-29 non-land cards), permutation invariance required, shared scorer weights for both decks in a training pair
**Scale/Scope**: 10,000+ match outcomes for training, 544-dim feature vectors, ~5-15M model parameters

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit tests for deterministic feature parsing, model forward pass, loss computation, normalization, data loading. All use small synthetic data — no Forge dependency. Integration tests for end-to-end training on tiny datasets. |
| II. Simplicity First | PASS | Set Transformer is the simplest attention architecture for unordered sets. Bradley-Terry is standard BCE on score differences. No speculative features. |
| III. Data Integrity | PASS | Raw features in .npz (incremental encoding preserved). Normalization stats in checkpoint (can't desync). Validation split by match (no leakage). Missing card → clear error. |
| IV. Domain-Driven Design | PASS | Domain: scorer model, deterministic features, training objective. Application: train, evaluate, encode use cases. Infrastructure: CLI, file I/O, model serialization, Java connector. |
| V. MTG Forge Interoperability | PASS | Evaluation workers reuse forge-connector JAR. New Java entry point (ValidationWorkerMain) for evaluation matches. Communication via flat text files. No changes to stub library. |
| VI. Documentation | PASS | CLI contracts documented. Quickstart with full workflow. ML rationale covered in research.md (Set Transformer choice, Bradley-Terry loss, normalization). |

### Post-Design Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Test plan: 8+ unit test files (features, model, loss, data loading, checkpointing, CLI args), 1 integration test (small training run). All synthetic data, fast execution. |
| II. Simplicity First | PASS | No ISAB (standard SAB sufficient for 20-29 card decks). No custom loss (standard BCE). No learning rate scheduling (manual Phase A/B). Single forward pass architecture. |
| III. Data Integrity | PASS | Feature parsing fully tested with known card examples. Checkpoint round-trip tested. Normalization statistics validated against manual computation. |
| IV. Domain-Driven Design | PASS | Clear layer separation — see Project Structure below. Model architecture (domain) has no file I/O concerns. CLI (infrastructure) delegates to use cases (application). |
| V. MTG Forge Interoperability | PASS | New ValidationWorkerMain reads two pre-built decks from flat file, plays via GamePlayer, writes outcomes. Forge's SealedDeckBuilder used in a separate deck-building step. Same forge-connector module. |
| VI. Documentation | PASS | CLI contract, file format contract, quickstart, and research.md all produced. |

## Project Structure

### Documentation (this feature)

```text
specs/013-sealed-deck-scorer/
├── plan.md              # This file
├── research.md          # Phase 0 output — architecture decisions
├── data-model.md        # Phase 1 output — entities and relationships
├── quickstart.md        # Phase 1 output — getting started guide
├── contracts/
│   ├── cli.md           # CLI interface contract
│   └── file-formats.md  # File format contracts (validation matches, outcomes, checkpoints)
└── tasks.md             # Phase 2 output (generated by /speckit.tasks)
```

### Source Code (repository root)

```text
src/sealed/
├── domain/
│   ├── card_encoder.py              # EXISTING — extend to produce 544-dim vectors
│   ├── deterministic_features.py    # NEW — parse 32 deterministic features from card text
│   └── scorer_model.py              # NEW — Set Transformer scorer (nn.Module)
├── application/
│   ├── encode_cards.py              # EXISTING — update skip logic for dimension check
│   ├── train_scorer.py              # NEW — training use case (data loading, training loop, validation)
│   └── evaluate_scorer.py           # NEW — evaluation use case (greedy search, worker coordination)
└── infrastructure/
    ├── cli.py                       # EXISTING — add train-scorer and evaluate-scorer subcommands
    ├── embedding_store.py           # EXISTING — no changes needed
    ├── match_data_loader.py         # NEW — parse match-outcomes.txt, build PyTorch datasets
    ├── scorer_store.py              # NEW — save/load model checkpoints (.pt files)
    └── evaluation_connector.py      # NEW — launch/manage Java ValidationWorkerMain processes

forge-connector/src/main/java/com/pricepredictor/connector/
├── ValidationWorkerMain.java        # NEW — entry point for evaluation workers
└── ValidationMatchPlayer.java       # NEW — read matches file, play games, write outcomes

tests/unit/sealed/
├── domain/
│   ├── test_card_encoder.py               # EXISTING — extend with 544-dim tests
│   ├── test_deterministic_features.py     # NEW — unit tests for feature parsing
│   └── test_scorer_model.py               # NEW — forward pass, permutation invariance, masking
├── application/
│   ├── test_encode_cards.py               # EXISTING — add 544-dim / skip-logic tests
│   ├── test_train_scorer.py               # NEW — training loop, loss computation, validation
│   └── test_evaluate_scorer.py            # NEW — greedy search, result aggregation
└── infrastructure/
    ├── test_match_data_loader.py          # NEW — parsing, dataset construction, batching
    ├── test_scorer_store.py               # NEW — checkpoint save/load round-trip
    └── test_evaluation_connector.py       # NEW — worker command construction

forge-connector/src/test/java/com/pricepredictor/connector/
└── ValidationMatchPlayerTest.java         # NEW — match file parsing, outcome writing, crash recovery
```

**Structure Decision**: Extends the existing `src/sealed/` package following the established domain/application/infrastructure layering. No new top-level packages. The forge-connector module gains two new Java classes in the existing package. Test structure mirrors source exactly.

## Complexity Tracking

No constitution violations to justify. All design decisions align with existing patterns.
