# Implementation Plan: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Date**: 2026-05-20 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/017-one-shot-deck-picker/spec.md`

## Summary

Add a one-shot sealed deck picker — a policy transformer over a sealed pool
that emits 23 spell picks (plus any nonbasic lands ranked above the 23rd
spell) in a single forward pass, replacing the iterative simulated-annealing
search used by `build-decks`. Training is REINFORCE from random init against
a frozen scorer, with a per-pool empirical-mean baseline, an entropy bonus
on a val-reward-driven decay schedule, and an auxiliary pool-quality head
trained against the per-pool mean reward. Inference is deterministic: argsort
logits, walk the order taking spells until the 23-spell quota fills,
encountering nonbasic lands along the way; the existing `compute_basic_lands`
fills to 40 cards. Two new CLI subcommands (`train-picker`, `pick-decks`)
ship in the existing `sealed` package; the resulting `generated-decks.txt`
is drop-in compatible with the existing `match-outcomes` self-play
infrastructure.

## Technical Context

**Language/Version**: Python 3.14+ (sealed and price_predictor packages); no Java changes (forge-connector is untouched).
**Primary Dependencies**: `torch` (CUDA 12.6 wheels), `numpy`, `scipy.stats` (for `spearmanr` in the cross-scorer audit — already available as a sklearn transitive dependency; explicit import is added). No new top-level dependencies.
**Storage**: PyTorch `.pt` checkpoints under `models/sealed/picker/` (`latest.pt`, `best_{timestamp}.pt` — see `contracts/checkpoint-format.md`); flat-text input (`output/sealed/pools/*/pools.txt`); flat-text output in the existing `LABEL;SET_CODE;Card1|...|Card40` schema (`output/sealed/generated-decks.txt`).
**Testing**: `pytest` for Python. New unit tests under `tests/unit/sealed/domain/test_picker_model.py`, `tests/unit/sealed/application/test_train_picker.py`, `tests/unit/sealed/application/test_pick_decks.py`, `tests/unit/sealed/infrastructure/test_picker_store.py`. No new integration tests in this feature: end-to-end Forge validation is documented as a manual procedure (spec § "End-of-training Forge validation").
**Target Platform**: Local development (Windows 11) and any OS with Python 3.14 + a CUDA-capable GPU. Training is GPU-required at the spec's scale (16 pools × 64 sampled decks per step + frozen-scorer forward); CPU is sufficient only for unit-test fixture sizes.
**Project Type**: CLI tool. Two new subcommands on `python -m sealed`: `train-picker` and `pick-decks`. No new top-level packages or modules outside `sealed/`.
**Performance Goals**: SC-001: wall-clock per deck on a single GPU at least an order of magnitude lower than the existing search-based builder (one picker forward vs. tens of thousands of scorer forwards). SC-002: from-scratch training on a 100k-pool corpus completes within a few hours on a single GPU.
**Constraints**: `d_model` must be divisible by `n_heads` — fail fast at startup (FR-033). `.npz` cache width must match the picker's `embedding_dim` (FR-002, FR-034, FR-035). `kl_coef != 0` requires `--picker-checkpoint` (FR-025). `--resume` and `--picker-checkpoint` are mutually exclusive (FR-024). Architecture flags forbidden on either (FR-022, FR-023). Random seed hardcoded to 42 (Clarify Q5, FR-018).
**Scale/Scope**: ~100k pools per training run (spec assumption), ~60–90 cards per pool, 16 pools × 64 sampled decks per gradient step = 1024 sampled decks scored per step. Picker checkpoint payload is small (~tens of MB even at `d_model = 512`). No background workers, no streaming I/O, no async.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Fast Automated Tests | **PASS** | All planned unit tests are pure-CPU and run in milliseconds: forward-shape checks, sampler termination on a hand-crafted tiny pool, baseline / advantage / log-prob arithmetic on hand-checked inputs, resume-precedence resolution, error-path coverage for width mismatches and mutually-exclusive flags. Integration tests are deliberately out of scope per the spec's "manual procedure" stance on end-of-training Forge validation. |
| II. Simplicity First | **PASS** | No new packages, no new domain abstractions beyond the picker model itself, no shared base class extracted from the three trainers. SAB primitive imported from `sealed.domain.scorer_model` (research §D1); `compute_basic_lands` / `is_land_embedding` / `ConvertedCardLocator` / `parse_pools` / `ScorerStore` / `torch_checkpoint` all reused unchanged; resume-precedence pattern lifted verbatim from `train-scorer`. The sampler, the Plackett-Luce log-prob, the distributional summaries, and the cross-scorer correlation are each inline ~20–40 line helpers in `train_picker.py` — no abstraction layer over them. |
| III. Data Integrity | **PASS** | Pool file parsed via the existing `parse_pools` (rejects malformed lines, tolerates blanks). `.npz` cache width validated against the scorer's `d_model` and the picker checkpoint's `embedding_dim` at startup (FR-002, FR-034, FR-035) with clear fail-fast messages. Random seed hardcoded to 42 so a fresh run is bit-for-bit reproducible. Frozen scorer is loaded once, `.eval()`-ed, never modified — verified via a unit test that asserts no scorer gradient flow after a training step. |
| IV. Domain-Driven Design | **PASS** | Layering preserved: `domain/picker_model.py` is framework-free except for `torch.nn` (the project convention for ML domain entities); `application/train_picker.py` and `application/pick_decks.py` orchestrate use cases; `infrastructure/picker_store.py` handles `.pt` persistence and `infrastructure/cli.py` wires argparse. Dependency direction is inward: infra → application → domain. No domain class reaches into infrastructure. The cross-package dependency direction (sealed depends on price_predictor for the shared tokenizer / checkpoint helper, never the reverse) is preserved. |
| V. MTG Forge Interoperability | **PASS** | No Forge-facing changes. `forge-connector` is untouched. The new `generated-decks.txt` files use the existing format that `match-outcomes` already consumes; no API or remote-protocol change. |
| VI. Documentation | **PASS** | `contracts/cli.md` documents both new subcommands and every flag, exit code, and error condition. `contracts/checkpoint-format.md` documents the `.pt` payload schema and resume-precedence rule. `quickstart.md` is an end-to-end walkthrough from setup to self-play participation. CLAUDE.md will be updated with `train-picker` and `pick-decks` entries during implementation (one of the survey follow-up tasks). |
| VII. Codebase-Aware Planning | **PASS** | Survey complete; outcome below. |

### Codebase Survey (Principle VII — required)

Full survey: [research.md#codebase-survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 11 concepts surveyed; 9 reused as-is or mirrored at the pattern level (`SAB`, `ScorerStore`, `ConvertedCardLocator`, `parse_pools`, `compute_basic_lands`, `is_land_embedding`, `EmbeddingTable`, `torch_checkpoint`, `_parse_label`); 2 retained as parallel concepts because the picker is a different *model* than the scorer / greedy-builder (`SetTransformerScorer`, `GreedyDeckBuilder`). 0 parallel concepts introduced silently.
- **Adjacent prior art**: 11 reuse decisions, 3 new inline helpers (Plackett-Luce sampler, rank-correlation diagnostic, distributional-summary aggregations) — none have prior art in the codebase and each is too small / single-use to warrant a dedicated module.
- **Convention alignment**: The picker mirrors `train-scorer`'s conventions one-for-one (folder layout, dataclass-backed CLI config, `--resume` semantics, architecture-flag rejection, fail-fast width checks, dual best/latest checkpoint persistence, hardcoded `random_seed = 42`). Zero deviations.
- **Third-instance check**: Best-checkpoint and training-loop scaffolding patterns now exist in three trainers (`train-encoder`, `train-scorer`, `train-picker`). Extraction is **deferred** — the three diverge on loss type, metric direction, and dataset shape; a unifying abstraction would be a generic training loop (premature). A TODO comment in `train_picker.py` will note the three sites for a future fourth-instance trigger.

**Follow-up tasks from survey** (carry into `tasks.md`):

- *Add* `train-picker` and `pick-decks` subcommands to `src/sealed/infrastructure/cli.py` via two new `_build_*_parser` functions and `run_*` dispatchers; mirror `_TRAIN_SCORER_ARCHITECTURE_FLAGS` and resume-precedence handling.
- *Add* `PickerModel` and `PickerConfig` to `src/sealed/domain/picker_model.py`; import `SAB` from `sealed.domain.scorer_model` rather than re-implementing.
- *Add* `PickerStore` and `LoadedPickerCheckpoint` to `src/sealed/infrastructure/picker_store.py` as a thin mirror of `ScorerStore`.
- *Add* `TrainPickerConfig`, `TrainPickerUseCase`, and the sampler / log-prob / distrib helpers to `src/sealed/application/train_picker.py`.
- *Add* `PickDecksConfig`, `PickDecksUseCase` to `src/sealed/application/pick_decks.py`; reuse `_count_complete_lines_and_truncate_partial` from `build_decks.py`.
- *Update* CLAUDE.md to document the two new subcommands.
- *Add* TODO comment naming the three trainers (`train-encoder`, `train-scorer`, `train-picker`) as candidate extraction sites for a future fourth REINFORCE-style trainer.
- *Verify* during implementation: the `POWER > 0 or TOUGHNESS > 0` heuristic for "is creature" against `deterministic_features.py`; if unsound, fall back to reading the type line via `ConvertedCardLocator`.

## Project Structure

### Documentation (this feature)

```text
specs/017-one-shot-deck-picker/
├── plan.md                          # This file
├── research.md                      # Codebase survey + Phase 0 design decisions
├── data-model.md                    # Entities, configs, runtime tensors, artifacts
├── quickstart.md                    # End-to-end walkthrough
├── contracts/
│   ├── cli.md                       # train-picker + pick-decks CLI surface
│   └── checkpoint-format.md         # Picker .pt payload schema + resume rules
├── spec.md                          # (existing, from /speckit.specify)
└── tasks.md                         # (generated by /speckit.tasks, NOT created here)
```

### Source Code (repository root)

```text
src/sealed/
├── application/
│   ├── train_picker.py              # NEW:
│   │                                #   - TrainPickerConfig (dataclass, all CLI knobs)
│   │                                #   - TrainPickerUseCase.execute() — orchestrates
│   │                                #     the full training run (load scorer + auditor,
│   │                                #     load pool file, build/resume picker, training
│   │                                #     loop, persist checkpoints, early stop)
│   │                                #   - _build_optimizer, _check_picker_width,
│   │                                #     _resume_or_build_picker, _persist_checkpoint
│   │                                #     — mirror train_scorer.py shapes
│   │                                #   - _sample_decks(): sequential without-replacement
│   │                                #     sampler vectorized across batch × n_samples
│   │                                #     (research §D2)
│   │                                #   - _plackett_luce_log_prob(): differentiable
│   │                                #     log-prob of the sampled deck under the
│   │                                #     current logits (FR-014, spec § 3.5)
│   │                                #   - _entropy_schedule(): val-reward-driven
│   │                                #     decay (FR-016)
│   │                                #   - _kl_penalty(): KL against bootstrap picker
│   │                                #     when --kl-coef != 0 (FR-025)
│   │                                #   - _validate(): deterministic walk + scorer
│   │                                #     forward over the val slice (FR-019)
│   │                                #   - _audit_correlation(): scipy.stats.spearmanr
│   │                                #     on val decks under both scorers (FR-030)
│   │                                #   - _distrib_summaries(): color/CMC/creature/
│   │                                #     type stats over val decks (FR-032)
│   ├── pick_decks.py                # NEW:
│   │                                #   - PickDecksConfig (dataclass)
│   │                                #   - PickDecksUseCase.execute() — one pool, one
│   │                                #     forward, deterministic walk, manabase fill,
│   │                                #     write line. Mirrors build_decks.py main loop.
│   │                                #   - reuses _count_complete_lines_and_truncate_partial
│   │                                #     from build_decks.py for --resume semantics
│   └── (no changes to existing app files)
├── domain/
│   ├── picker_model.py              # NEW:
│   │                                #   - PickerConfig (dataclass, mirror ScorerConfig)
│   │                                #     with __post_init__ divisibility check (FR-033)
│   │                                #   - PickerModel(nn.Module):
│   │                                #       input_projection (Linear or Identity)
│   │                                #       sab_layers (ModuleList[SAB])
│   │                                #       per_card_head (Linear → 1)
│   │                                #       aux_head (Linear → 1 over masked mean-pool)
│   │                                #     forward(pool_cards, pool_mask)
│   │                                #         -> (logits, aux_pred)
│   │                                #   - imports SAB from sealed.domain.scorer_model
│   └── (no changes to existing domain files;
│        SAB / is_land_embedding / compute_basic_lands reused unchanged)
└── infrastructure/
    ├── cli.py                       # MODIFIED:
    │                                #   - _build_train_picker_parser() — new helper
    │                                #     registering --pools-path, --scorer-checkpoint,
    │                                #     --auditor-scorer-checkpoint, --resume, etc.
    │                                #   - _build_pick_decks_parser() — new helper
    │                                #     registering --pools-path, --picker-checkpoint,
    │                                #     --label, --output, --resume.
    │                                #   - _TRAIN_PICKER_ARCHITECTURE_FLAGS,
    │                                #     _RESUMABLE_PICKER_FLAG_NAMES — mirrors of the
    │                                #     scorer tuples for argument validation
    │                                #   - run_train_picker(), run_pick_decks() —
    │                                #     dispatchers mirroring run_train_scorer /
    │                                #     run_build_decks shape
    │                                #   - build_parser() — two new subparser calls
    └── picker_store.py              # NEW:
                                     #   - LoadedPickerCheckpoint (frozen dataclass,
                                     #     mirror LoadedScorerCheckpoint)
                                     #   - PickerStore.save_checkpoint(): writes via
                                     #     torch_checkpoint.save_checkpoint; payload
                                     #     per contracts/checkpoint-format.md
                                     #   - PickerStore.load_checkpoint(): inverse,
                                     #     reconstructs PickerConfig from stored dict

tests/unit/sealed/
├── application/
│   ├── test_train_picker.py         # NEW:
│   │                                #   - sampler exits at 23-spell quota
│   │                                #   - sampler respects is_land bucketing
│   │                                #   - sampler does not pick the same card twice
│   │                                #   - per-pool baseline = rewards[i].mean()
│   │                                #   - advantage = rewards - baseline (and is detached)
│   │                                #   - aux loss target is detached
│   │                                #   - Plackett-Luce log-prob = sum of step-wise terms
│   │                                #     on a hand-checked tiny example
│   │                                #   - entropy schedule: held constant for K epochs,
│   │                                #     then decays only on plateaus
│   │                                #   - KL coef enforcement: kl_coef != 0 requires
│   │                                #     --picker-checkpoint (FR-025)
│   │                                #   - resume precedence: CLI > train_config > default
│   │                                #   - architecture-flag rejection on --resume /
│   │                                #     --picker-checkpoint
│   │                                #   - width-mismatch error at startup (cache vs.
│   │                                #     scorer, cache vs. picker checkpoint)
│   │                                #   - missing scorer at default path → fail fast
│   │                                #     with directing message (FR-036)
│   └── test_pick_decks.py           # NEW:
│                                    #   - deterministic walk matches § 1.1 pseudocode
│                                    #     on a hand-crafted pool
│                                    #   - pool with 0 picked nonbasic lands → 23 + 17 = 40
│                                    #   - pool with k > 0 picked nonbasic lands → 23 +
│                                    #     k + (17 − k) basics = 40
│                                    #   - --resume: existing lines counted, partial
│                                    #     trailing line truncated, append continues
│                                    #   - --label written verbatim as first column
│                                    #   - width-mismatch error at picker load (FR-035)
├── domain/
│   └── test_picker_model.py         # NEW:
│                                    #   - forward output shapes: (B, N) logits + (B,) aux
│                                    #   - mask=False positions contribute 0 to aux mean-pool
│                                    #   - input_projection inserted iff d_model != embedding_dim
│                                    #   - d_model % n_heads != 0 raises in __post_init__ (FR-033)
│                                    #   - state_dict contains aux_head and per_card_head keys
└── infrastructure/
    └── test_picker_store.py         # NEW:
                                     #   - round-trip save/load preserves weights, config,
                                     #     optimizer state, epoch, best_val_reward, train_config
                                     #   - load_checkpoint reconstructs PickerConfig from
                                     #     stored dict (not dataclass instance)
                                     #   - best_{timestamp}.pt and latest.pt both use
                                     #     the same payload schema

# No existing tests modified (the new code lives in new files; the few
# touch-points in `cli.py` and CLAUDE.md are configuration / docs).
```

**Structure Decision**: All Python changes live inside the existing `sealed`
package (one new file per layer: `domain/picker_model.py`,
`application/train_picker.py`, `application/pick_decks.py`,
`infrastructure/picker_store.py`) plus two helper functions and two
dispatchers added to the existing `infrastructure/cli.py`. No changes to
the `price_predictor` package and no changes to the Java `forge-connector`
module — the picker is downstream of both.

## Complexity Tracking

No constitution violations. No complexity justifications needed.
