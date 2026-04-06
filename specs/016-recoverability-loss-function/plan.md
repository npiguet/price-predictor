# Implementation Plan: Recoverability-Based Per-Step Stage 2 Loss

**Branch**: `016-recoverability-loss-function` | **Date**: 2026-04-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/016-recoverability-loss-function/spec.md`

## Summary

Replace Stage 2's uniform end-of-episode mana score reward with a per-step reward that combines the existing Stage 1 budget signal (+1/-1) with a discrete imbalance shaping signal in {-1, -0.5, 0, +0.5, +1}. The shaping signal measures whether each pick improved or worsened the deck's mana imbalance, with stronger signals (±1) when imbalance is >= 3 and weaker signals (±0.5) when < 3. Shaping is 0 until both pip demand and mana supply exist. This gives the model step-by-step credit assignment with clear directional feedback.

> **Revision 2026-04-06**: Originally used continuous PBRS (`tanh(delta_ratio / temperature)`) with `--urgency-exponent` and `--temperature` CLI args. Replaced with discrete shaping after the continuous signal proved too weak (~0.001 magnitude for early picks, constant `shaping=-0.07` during training).

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: PyTorch, numpy
**Storage**: N/A (rewards computed in-memory during training; no new persistence)
**Testing**: pytest (fast unit tests in `tests/unit/`, slower integration tests in `tests/integration/`)
**Target Platform**: Local workstation (Windows 11, CUDA optional)
**Project Type**: CLI / ML training pipeline
**Performance Goals**: Reward computation must not significantly slow the training loop. Currently the bottleneck is `collect` (episode rollout, ~seconds) and `update` (PPO gradient step). The new per-step reward computation reuses existing mana analysis functions and adds O(40) arithmetic per episode — negligible.
**Constraints**: Total reward must stay in [-2, 2]. Stage 1 budget signal must be preserved at full strength. No new external dependencies.
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
│   └── mana_scorer.py          # ADD: compute_per_step_rewards() with discrete shaping
├── application/
│   ├── train_stage2.py          # MODIFY: per-step reward override, batch logging
│   └── sample_stage2.py         # MODIFY: mana cost prefix on non-land picks (FR-011)
└── infrastructure/
    └── cli.py                   # No new CLI args (discrete shaping has no hyperparameters)

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
