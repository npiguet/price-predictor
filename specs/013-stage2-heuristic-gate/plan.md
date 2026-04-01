# Implementation Plan: Stage 2 Training — Heuristic Gate

**Branch**: `013-stage2-heuristic-gate` | **Date**: 2026-03-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/013-stage2-heuristic-gate/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Stage 2 continues training the pool transformer (encoder frozen) using a heuristic mana-score reward. The
model learns to build decks whose land base matches the mana requirements of its spells. A new domain module
computes pip counts, ideal mana distributions, actual mana sources, and a score in [0, 1]. Completed episodes
receive the mana-score reward uniformly across all 40 steps; duplicate-pick episodes fall back to Stage 1
per-step rewards. Training halts when all 32 episodes in a batch score > 0.90.

The implementation reuses the existing EpisodeRunner, PPOTrainer, and checkpoint infrastructure unchanged.
New code is limited to: (1) a domain mana_scorer module, (2) application use cases for Stage 2 train/sample,
(3) CLI routing for `--stage 2` and `--init-from`, and (4) extraction of the shared embedding adapter.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: PyTorch, numpy
**Storage**: File-based (.pt checkpoints, .npz embeddings, .txt card files)
**Testing**: pytest, ruff
**Target Platform**: Local machine (CPU or CUDA)
**Project Type**: CLI (ML training tool)
**Performance Goals**: N/A (offline training, no latency targets)
**Constraints**: Reuse existing Stage 1 infrastructure; only pool transformer trains (encoder frozen)
**Scale/Scope**: 32 episodes/batch, 40 picks/episode, 6 mana colors (W/U/B/R/G/C)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Mana score is a pure function — extensive unit tests with hand-calculated values. Use cases testable with mocked dependencies. |
| II. Simplicity First | PASS | Reuses existing episode runner, PPO trainer, checkpoint store. New code limited to mana scorer + use case orchestration. Adapter extraction justified by 4 consumers. |
| III. Data Integrity | PASS | Mana score computation is deterministic given the same deck. Card file parsing follows spec 006 format. |
| IV. Domain-Driven Design | PASS | mana_scorer.py is pure domain (no I/O). Card text access via port/adapter. Use cases orchestrate domain + infrastructure. |
| V. MTG Forge Interop | N/A | Training feature — no API/stub changes. |
| VI. Documentation | PASS | Will document `--stage 2` workflow, `--init-from` flag, and sample output format. |

**Post-Phase 1 re-check**: All gates still pass. The adapter extraction adds one new infrastructure file but
removes a cross-module private import. No new abstractions beyond the justified adapter extraction.

## Project Structure

### Documentation (this feature)

```text
specs/013-stage2-heuristic-gate/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cli.md           # Phase 1 output — CLI contract changes
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/sealed/
├── domain/
│   ├── card_embedding_port.py   # MODIFIED — add get_card_text()
│   ├── mana_scorer.py           # NEW — pure functions: pip counting, source counting, ideal dist, score
│   ├── card_encoder.py          # unchanged
│   ├── episode_runner.py        # unchanged
│   ├── pool_transformer.py      # unchanged
│   ├── ppo_trainer.py           # unchanged
│   └── replay_buffer.py         # unchanged
├── application/
│   ├── train_stage2.py          # NEW — TrainStage2UseCase
│   ├── sample_stage2.py         # NEW — SampleStage2UseCase
│   ├── train_stage1.py          # MODIFIED — remove _EmbeddingAdapter (extracted)
│   ├── sample_stage1.py         # MODIFIED — import adapter from new location
│   ├── encode_cards.py          # unchanged
│   └── generate_pools.py        # unchanged
└── infrastructure/
    ├── cli.py                   # MODIFIED — add --init-from, route --stage 2
    ├── embedding_adapter.py     # NEW — extracted from train_stage1._EmbeddingAdapter + get_card_text()
    ├── embedding_store.py       # unchanged
    ├── pool_connector.py        # unchanged
    ├── pool_loader.py           # unchanged
    └── pool_model_store.py      # unchanged

tests/
├── unit/sealed/
│   ├── domain/
│   │   └── test_mana_scorer.py       # NEW — extensive unit tests (one per acceptance scenario)
│   ├── application/
│   │   ├── test_train_stage2.py      # NEW
│   │   └── test_sample_stage2.py     # NEW
│   └── infrastructure/
│       ├── test_cli_sealed_train.py  # MODIFIED — test --stage 2, --init-from
│       ├── test_cli_sealed_sample.py # MODIFIED — test --stage 2
│       └── test_embedding_adapter.py # NEW — test extracted adapter + get_card_text()
└── integration/sealed/
    └── test_train_stage2_integration.py  # NEW
```

**Structure Decision**: Single-project layout following existing DDD layering. No new packages or
structural changes — Stage 2 files are added alongside their Stage 1 counterparts.
