# Implementation Plan: Recoverability-Based Per-Step Stage 2 Loss

**Branch**: `016-recoverability-loss-function` | **Date**: 2026-04-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/016-recoverability-loss-function/spec.md`

## Summary

Replace Stage 2's uniform end-of-episode mana score reward with a per-step reward that combines the existing Stage 1 budget signal (+1/-1) with a recoverability-based shaping signal bounded to (-1, 1) via `tanh`. The shaping signal measures how each pick changes the deck's mana recoverability — the ratio of mana imbalance to remaining picks raised to a configurable exponent. This gives the model step-by-step credit assignment: each pick is immediately rewarded or penalized based on whether it moved the deck closer to its ideal mana distribution, with urgency that naturally increases as fewer picks remain.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: PyTorch, numpy, math (stdlib tanh)
**Storage**: N/A (rewards computed in-memory during training; no new persistence)
**Testing**: pytest (fast unit tests in `tests/unit/`, slower integration tests in `tests/integration/`)
**Target Platform**: Local workstation (Windows 11, CUDA optional)
**Project Type**: CLI / ML training pipeline
**Performance Goals**: Reward computation must not significantly slow the training loop. Currently the bottleneck is `collect` (episode rollout, ~seconds) and `update` (PPO gradient step). The new per-step reward computation reuses existing mana analysis functions and adds O(40) arithmetic per episode — negligible.
**Constraints**: Total reward must stay in (-2, 2). Stage 1 budget signal must be preserved at full strength. No new external dependencies.
**Scale/Scope**: 40 picks per episode, 32 episodes per batch. Reward computed per-step inline during episode collection.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Unit tests for recoverability ratio, shaping signal, and reward composition. All pure arithmetic — runs in milliseconds. |
| II. Simplicity First | PASS | No new abstractions. Adds ~1 function to `mana_scorer.py` and modifies the reward override in `train_stage2.py`. Two configurable hyperparameters driven by real need (spec FR-008). |
| III. Data Integrity | PASS | Deterministic computation from existing pip/source data. Same input → same reward. No new external inputs. |
| IV. Domain-Driven Design | PASS | Recoverability ratio and shaping signal are domain computations living in `mana_scorer.py` (domain layer). Training orchestration stays in `train_stage2.py` (application layer). No infrastructure dependencies in domain code. |
| V. MTG Forge Interoperability | N/A | Internal training reward — no API or stub changes. |
| VI. Documentation | PASS | FR-011 adds mana cost display to sample output. Batch log line updated per FR-012. No new CLI commands or workflows requiring README updates. |

**Quality Gates**:
- All automated tests pass — new unit tests + existing regression suite
- No new linting warnings (ruff check)
- Domain logic (recoverability computation) has no infrastructure imports
- No new CLI commands or workflows → no documentation update needed beyond the spec

## Project Structure

### Documentation (this feature)

```text
specs/016-recoverability-loss-function/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/sealed/
├── domain/
│   └── mana_scorer.py          # ADD: compute_recoverability_ratio(), compute_per_step_rewards()
├── application/
│   ├── train_stage2.py          # MODIFY: per-step reward override, new hyperparams, batch logging
│   └── sample_stage2.py         # MODIFY: mana cost prefix on non-land picks (FR-011)
└── infrastructure/
    └── cli.py                   # MODIFY: add --urgency-exponent, --temperature CLI args

tests/
├── unit/sealed/domain/
│   └── test_mana_scorer.py      # ADD: tests for recoverability ratio and shaping signal
├── unit/sealed/application/
│   ├── test_train_stage2.py     # MODIFY: tests for per-step reward, logging format
│   └── test_sample_stage2.py    # MODIFY: tests for mana cost prefix display
└── unit/sealed/infrastructure/
    └── test_cli_sealed_train_stage2.py  # MODIFY: tests for new CLI args
```

**Structure Decision**: Single project layout. All changes touch existing files in `src/sealed/` and `tests/`. No new modules or packages needed.

## Complexity Tracking

No constitution violations to justify.
