---
description: "Task list for One-Shot Sealed Deck Picker implementation"
---

# Tasks: One-Shot Sealed Deck Picker

**Input**: Design documents from `/specs/017-one-shot-deck-picker/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, contracts/checkpoint-format.md

**Tests**: MANDATORY per the project constitution (Principle I: Fast Automated Tests). Every new module ships with unit tests; no integration tests (the end-to-end Forge validation is a documented manual procedure per spec § "End-of-training Forge validation").

**Organization**: Tasks are grouped by user story. US1 (train-picker) is the load-bearing MVP; US2 (pick-decks) depends on the foundational model + store; US3 (monitoring) layers audits onto US1's training loop.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1 / US2 / US3 (Setup, Foundational, and Polish carry no story label)
- All paths are repository-relative.

## Path Conventions

Single project, hexagonal layout under `src/sealed/` (`domain` → `application` → `infrastructure`); tests under `tests/unit/sealed/`. Matches the existing scorer/encoder modules.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project-level prerequisites for the picker feature.

- [X] T001 [P] Add `scipy` to `[project].dependencies` in `pyproject.toml` (used explicitly for `scipy.stats.spearmanr` in US3's cross-scorer audit; currently only present transitively via `scikit-learn`). Verify `python -c "import scipy.stats"` succeeds in the venv.

**Checkpoint**: Dependency available; existing `tests/unit/sealed/{domain,application,infrastructure}/` directories already exist (no scaffolding needed).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The picker model, the deterministic pick-decomposition walk, and checkpoint persistence — all shared by US1 (training + validation) and US2 (inference). No user story can proceed until these exist.

**⚠️ CRITICAL**: US1 and US2 both consume every component in this phase.

### Tests (write first, expect failure)

- [X] T002 [P] Unit tests for `PickerModel` and the deterministic walk in `tests/unit/sealed/domain/test_picker_model.py`: forward output shapes `(B, N)` logits + `(B,)` aux; input projection inserted iff `d_model != embedding_dim` (FR-002); aux head's masked mean-pool ignores padding positions; `d_model % n_heads != 0` raises in `PickerConfig.__post_init__` (FR-033); `state_dict` contains both `per_card_head` and `aux_head` keys; deterministic walk reproduces the § 1.1 pseudocode on a hand-crafted pool (spell quota stops at 23, nonbasic lands encountered above the 23rd spell are taken, no card picked twice) (FR-006, FR-008).
- [X] T003 [P] Unit tests for `PickerStore` round-trip in `tests/unit/sealed/infrastructure/test_picker_store.py`: save → load preserves weights, `PickerConfig`, optimizer state, `epoch`, `best_val_reward`, `train_config`; `config` is reconstructed from the stored dict (not a dataclass instance); `latest.pt` and `best_{timestamp}.pt` are written to the right paths and share the same payload schema (contracts/checkpoint-format.md, FR-037, FR-038).

### Implementation

- [X] T004 [P] Create `PickerConfig` dataclass (`embedding_dim`, `d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`) with a `__post_init__` divisibility check (`d_model % n_heads == 0`, FR-033) and `d_ff` default of `4 * d_model` in `src/sealed/domain/picker_model.py`. Prior art: mirror `ScorerConfig` in `src/sealed/domain/scorer_model.py:14` (new sibling config justified — picker is a distinct model per plan.md Codebase Survey).
- [X] T005 Implement `PickerModel(nn.Module)` in `src/sealed/domain/picker_model.py`: optional `Linear(embedding_dim, d_model)` projection (else `nn.Identity`) (FR-002); `nn.ModuleList[SAB]` of `n_layers` blocks importing `SAB` from `sealed.domain.scorer_model` (FR-003, no positional encoding); shared `per_card_head = Linear(d_model, 1)` over each token output (FR-004); `aux_head = Linear(d_model, 1)` over the padding-masked mean-pool of token outputs (FR-005); `forward(pool_cards, pool_mask) -> (logits, aux_pred)`. Depends on T004.
- [X] T006 Implement the deterministic pick-decomposition walk in `src/sealed/domain/picker_model.py` — a function taking `(logits, pool_embeddings, pool_names)` that argsorts logits descending, walks taking spells until the 23-spell quota fills and taking nonbasic lands encountered along the way, using `is_land_embedding` (`src/sealed/domain/card_embedding_layout.py:65`) for the partition (FR-006, FR-008). Returns the chosen card-name list. Depends on T005.
- [X] T007 [P] Implement `LoadedPickerCheckpoint` (frozen dataclass) and `PickerStore` in `src/sealed/infrastructure/picker_store.py`: `save_checkpoint(model, optimizer, epoch, best_val_reward, config, path, *, train_config)` writing via `torch_checkpoint.save_checkpoint` (`src/price_predictor/infrastructure/torch_checkpoint.py:14`); `load_checkpoint(path)` reconstructing `PickerConfig`; payload per contracts/checkpoint-format.md (FR-037, FR-038). Prior art: mirror `ScorerStore` in `src/sealed/infrastructure/scorer_store.py:39`.

**Checkpoint**: `PickerModel` builds + does a forward pass; the deterministic walk produces a valid chosen list; checkpoints round-trip. T002/T003 pass.

---

## Phase 3: User Story 1 - Train a one-shot picker against a frozen scorer (Priority: P1) 🎯 MVP

**Goal**: A `train-picker` subcommand that trains a picker from random init via REINFORCE against a frozen scorer, with per-pool baseline, entropy schedule, aux head, validation-reward best-checkpoint selection, early stopping, resume, and prior-picker bootstrap.

**Independent Test**: Run `python -m sealed train-picker --pools-path <small-pools> --scorer-checkpoint <scorer>` for a few epochs; verify `latest.pt` + `best_{timestamp}.pt` appear in `models/sealed/picker/`, validation reward improves from epoch 0 to the final epoch, and each epoch logs the policy/entropy/aux decomposition plus validation reward.

### Tests for User Story 1 (write first, expect failure)

- [X] T008 [P] [US1] Sampler + log-prob tests in `tests/unit/sealed/application/test_train_picker.py`: sequential without-replacement sampler exits at the 23-spell quota (FR-012); it never picks the same card twice; nonbasic-land picks are bucketed via `is_land_embedding`; the GPU-batched sampler operates across the full `(B*S, N)` tensor in vectorized steps (FR-013); Plackett-Luce log-prob equals the summed `logit_picked − logsumexp(remaining_logits)` on a hand-checked tiny example and is differentiable in the logits (FR-014).
- [X] T009 [P] [US1] Loss + schedule tests in `tests/unit/sealed/application/test_train_picker.py`: per-pool baseline `= rewards[i].mean()` (FR-011, § 3.3); `advantage = rewards − baseline` and is `.detach()`-ed (FR-014); aux-loss target `rewards.mean(dim=1)` is `.detach()`-ed (FR-015); entropy coefficient is held constant until `entropy_decay_after` consecutive monotonically-improving val-reward epochs, then decays only on plateaus (FR-016); KL penalty is zero when `kl_coef == 0` and active against the bootstrap reference otherwise (FR-025).
- [X] T010 [P] [US1] CLI validation + width tests in `tests/unit/sealed/application/test_train_picker.py`: `--resume` and `--picker-checkpoint` mutually exclusive (FR-024); architecture flags rejected alongside either (FR-022, FR-023); resume precedence (CLI > checkpoint `train_config` > dataclass default); `kl_coef != 0` without `--picker-checkpoint` fails fast (FR-025); `.npz` cache width vs. resumed/bootstrap checkpoint width mismatch fails fast (FR-034); missing scorer at the default path with no `--scorer-checkpoint` fails fast with a directing message (FR-036).
- [X] T010a [P] [US1] Training-loop behavioral tests in `tests/unit/sealed/application/test_train_picker.py`: the validation slice is exactly the front `--val-fraction` of the pools file and is excluded from the per-epoch shuffle, and the hardcoded `random_seed = 42` yields an identical train-order and split across two constructions (FR-018, Principle III); `_validate` returns the mean reward over the deterministically-built chosen-card decks and the best-checkpoint selector updates only on a strict improvement (FR-019); the early-stop counter fires after exactly `--patience` epochs without val-reward improvement and resets on a new best (FR-020); the degenerate "all sampled decks in a pool score equally → advantage all-zero → zero policy-gradient contribution" case (spec Edge Cases). Relies on the pure helpers factored in T017 and T021.

### Implementation for User Story 1

- [X] T011 [US1] Create `TrainPickerConfig` dataclass (all fields + defaults per data-model.md §1.3; hardcoded `random_seed = 42` constant, FR-018) and a `_build_train_config` flattener (Path→str) for checkpoint persistence in `src/sealed/application/train_picker.py`. Prior art: `TrainScorerConfig` (`src/sealed/application/train_scorer.py:40`) and `_build_train_config` (`train_scorer.py:955`).
- [X] T012 [US1] Implement the GPU-batched sequential without-replacement sampler `_sample_decks(logits, is_land_mask, pool_mask, n_samples, temperature)` in `src/sealed/application/train_picker.py` (research §D2, FR-012, FR-013): vectorized `torch.multinomial` loop over `(B*S, N)`, per-row spell-quota stop, returns pick indices + picked mask under `torch.no_grad()`. Depends on T011.
- [X] T013 [US1] Implement `_plackett_luce_log_prob(logits, pick_indices, picked_mask)` (differentiable, FR-014, § 3.5) and `_policy_entropy(logits, pool_mask)` in `src/sealed/application/train_picker.py`. Depends on T012.
- [X] T014 [US1] Implement the frozen-scorer reward path in `src/sealed/application/train_picker.py`: load training scorer via `ScorerStore`, `.eval()`, GPU; normalize the pool array once per batch via `scorer.normalize_features`; score each sampled deck as the chosen pool cards only — spells + nonbasic lands, **no basic lands** (FR-012) — by indexing the normalized pool array and calling `scorer.forward_prenormalized` under `torch.no_grad()` + fp16 autocast; return `rewards (B, S)`. This is bit-for-bit the input `GreedyDeckBuilder` scores (`deck_spells + deck_lands`, no basics). Prior art: `GreedyDeckBuilder._score_batch` (`src/sealed/domain/greedy_deck_builder.py:516`). Depends on T011.
- [X] T015 [US1] Implement the loss assembly in `src/sealed/application/train_picker.py`: per-pool baseline + detached advantage (FR-011, § 3.3); `policy_loss = -(advantage.detach() * log_prob).mean()`; `entropy_loss = -entropy_coef * entropy.mean()`; `aux_loss = mse(aux_pred, rewards.mean(dim=1).detach())` (FR-015); optional `kl_coef * kl(picker || bootstrap)` (FR-025); `total = policy + entropy + aux_weight*aux + kl`. Depends on T013, T014.
- [X] T016 [US1] Implement the entropy schedule `_entropy_schedule` (held constant until `entropy_decay_after` consecutive monotonic val-reward improvements, then ×0.9 on each subsequent plateau epoch, FR-016) in `src/sealed/application/train_picker.py`. Depends on T011.
- [X] T017 [US1] Implement pool loading + split in `src/sealed/application/train_picker.py`: `parse_pools` (`src/sealed/infrastructure/pool_file_reader.py:32`) loads the whole file; the front `--val-fraction` is the fixed validation slice; the remainder is shuffled per epoch with `random.Random(42)` (FR-010, FR-018, research §D3). Factor the front-slice split and the per-epoch shuffle into pure helpers (e.g. `_split_pools(pools, val_fraction)`, `_shuffle_train(train, rng)`) so FR-018 is unit-testable without a training run (T010a). Depends on T011.
- [X] T018 [US1] Implement `_validate` in `src/sealed/application/train_picker.py`: run the deterministic walk (T006) over the entire validation slice, score the resulting chosen-card decks (spells + nonbasic lands, **no basic lands** — same scoring input as T014) with the frozen training scorer, return mean reward (FR-019). Depends on T006, T014, T017.
- [X] T019 [US1] Implement `_build_optimizer` (single "picker" AdamW group) + per-group gradient-norm clipping at `--max-grad-norm` (FR-017) in `src/sealed/application/train_picker.py`. Prior art: `_build_optimizer` / `_clip_per_group` (`train_scorer.py:591, 817`). Depends on T011.
- [X] T020 [US1] Implement `_resume_or_build_picker` + width checks in `src/sealed/application/train_picker.py`: fresh build sized to the `.npz` cache width; `--resume` restores weights/optimizer/epoch/`best_val_reward`; `--picker-checkpoint` loads weights only; `_check_picker_width` fails fast on cache/checkpoint width mismatch (FR-034); missing-scorer guard (FR-036). Prior art: `_resume_or_build_model` + `_check_scorer_width` (`train_scorer.py:465, 430`). Depends on T007, T011.
- [X] T021 [US1] Implement `TrainPickerUseCase.execute` orchestration in `src/sealed/application/train_picker.py`: setup (load scorer, cache, pools, build/resume picker, optimizer), epoch loop (per-step forward → sample → score → loss → backward → clip → step), per-epoch `_validate`, per-epoch log line with the policy/entropy/aux decomposition + validation reward (FR-029), checkpoint persistence (`latest.pt` every epoch, `best_{timestamp}.pt` on new best, FR-037), early stopping after `--patience` epochs without val-reward improvement (FR-020) — factor the early-stop decision into a pure helper (e.g. `_should_stop(epochs_since_best, patience)`) so FR-020 is unit-testable (T010a). Add a TODO comment naming the three trainers (`train-encoder`, `train-scorer`, `train-picker`) as candidate shared-abstraction extraction sites for a future fourth REINFORCE-style trainer (plan.md survey follow-up). Depends on T012–T020.
- [X] T022 [US1] Add `_build_train_picker_parser`, `run_train_picker`, `_TRAIN_PICKER_ARCHITECTURE_FLAGS`, and `_RESUMABLE_PICKER_FLAG_NAMES` to `src/sealed/infrastructure/cli.py`, and register the subparser in `build_parser()`. Wire every flag from contracts/cli.md (FR-021); enforce `--resume`/`--picker-checkpoint` mutual exclusivity (FR-024), architecture-flag rejection (FR-022, FR-023), `kl_coef != 0` requires `--picker-checkpoint` (FR-025), and resume-precedence resolution. Prior art: `_build_train_scorer_parser` / `run_train_scorer` (`cli.py:389, 920`). Depends on T021.

**Checkpoint**: `python -m sealed train-picker` runs end-to-end on a small pools file, persists both checkpoints, logs per-epoch decomposition + val reward, resumes correctly. US1 independently testable. **MVP COMPLETE.**

---

## Phase 4: User Story 2 - Use a trained picker to build decks from pools (Priority: P2)

**Goal**: A `pick-decks` subcommand that runs deterministic picker inference once per pool, fills basic lands, and writes a `generated-decks.txt` drop-in for `match-outcomes`.

**Independent Test**: Given a trained picker checkpoint and a pools file, run `python -m sealed pick-decks --pools-path <pools> --label <tag>`; verify one output line per pool, each `LABEL;SET_CODE;Card1|...|Card40` with exactly 40 cards (23 spells + nonbasic lands + basics).

### Tests for User Story 2 (write first, expect failure)

- [X] T023 [P] [US2] Unit tests for `PickDecksUseCase` in `tests/unit/sealed/application/test_pick_decks.py`: deterministic walk + manabase fill yields exactly 40 cards for a pool with 0 picked nonbasic lands (23 + 17 basics) and for a pool with `k > 0` picked nonbasic lands (23 + k + (17−k) basics) (FR-006, FR-007); `--label` written verbatim as the first column (FR-027); `--resume` counts complete lines, truncates a partial trailing line, and appends remaining decks (FR-028); picker-checkpoint vs. `.npz` cache width mismatch fails fast (FR-035); pools with fewer than 23 embeddable cards are skipped.

### Implementation for User Story 2

- [X] T024 [US2] Create `PickDecksConfig` dataclass (fields per data-model.md §1.4) in `src/sealed/application/pick_decks.py`. Prior art: `BuildDecksConfig` (`src/sealed/application/build_decks.py:50`).
- [X] T025 [US2] Implement `PickDecksUseCase.execute` in `src/sealed/application/pick_decks.py`: load picker via `PickerStore` (+ width check, FR-035); load `.npz` cache via `ConvertedCardLocator`; iterate pools from `parse_pools`; per pool run one `PickerModel` forward + the deterministic walk (T006) + `compute_basic_lands` fill to 40 (FR-006, FR-007); write `LABEL;SET_CODE;Card1|...|Card40` lines (FR-026, FR-027); `--resume` append-and-skip reusing `_count_complete_lines_and_truncate_partial` from `src/sealed/application/build_decks.py:26` (FR-028). Prior art: `BuildDecksUseCase.execute` (`build_decks.py:90`) — parallel use case justified (forward pass vs. SA search) per plan.md survey. Depends on T006, T007, T024.
- [X] T026 [US2] Add `_build_pick_decks_parser` + `run_pick_decks` to `src/sealed/infrastructure/cli.py` and register in `build_parser()`; wire flags per contracts/cli.md, reusing `_parse_label` (`cli.py:35`) for `--label` validation (FR-027). Prior art: `_build_build_decks_parser` / `run_build_decks` (`cli.py:287, 850`). Depends on T025.

**Checkpoint**: `python -m sealed pick-decks` produces a valid `generated-decks.txt`; the file is accepted unmodified by `match-outcomes --side-a-decks` (SC-005, verified manually). US1 + US2 both work.

---

## Phase 5: User Story 3 - Monitor training quality and detect reward hacking (Priority: P3)

**Goal**: Per-epoch reward-hacking audits layered onto US1's training loop — cross-scorer rank correlation (when an auditor scorer is configured) and distributional summaries of the validation decks.

**Independent Test**: Run `train-picker` with both `--scorer-checkpoint` and `--auditor-scorer-checkpoint`; verify each per-epoch log line includes the cross-scorer rank correlation and the distributional summaries (color count, condensed CMC histogram, creature count, type balance) alongside validation reward.

### Tests for User Story 3 (write first, expect failure)

- [X] T027 [P] [US3] Audit + summary tests in `tests/unit/sealed/application/test_train_picker.py`: cross-scorer rank correlation computed via `scipy.stats.spearmanr` over validation decks when an auditor is set, and skipped (no auditor forward, no correlation field) when omitted (FR-030); distributional summaries compute mean color count, the 5-bin condensed CMC histogram (CMC≤2, 3, 4, 5, 6+), mean creature count, and creature/noncreature type-balance ratios from the validation-deck embeddings (FR-032); auditor-scorer width mismatch vs. the `.npz` cache fails fast at startup (FR-035 analogue).

### Implementation for User Story 3

- [X] T028 [US3] Add `--auditor-scorer-checkpoint` to `_build_train_picker_parser` and `TrainPickerConfig`; load the auditor scorer (frozen, `.eval()`, startup width check) in `TrainPickerUseCase` setup (FR-021, FR-030). Files: `src/sealed/infrastructure/cli.py`, `src/sealed/application/train_picker.py`. Depends on T022.
- [X] T029 [US3] Implement `_audit_correlation` (`scipy.stats.spearmanr` between training-scorer and auditor scores on the validation decks) in `src/sealed/application/train_picker.py`, invoked in `_validate` only when the auditor is configured (FR-030). Depends on T018, T028.
- [X] T030 [US3] Implement `_distrib_summaries` over the validation decks in `src/sealed/application/train_picker.py`: mean color count (via `COLOR_FLAGS`), 5-bin condensed CMC histogram (via `MANA_VALUE`), mean creature count + type-balance ratios (via the `POWER`/`TOUGHNESS` slots; research §D7) (FR-032). **Verify** the `POWER > 0 or TOUGHNESS > 0` creature heuristic against `src/sealed/domain/deterministic_features.py`; if unsound for some card types (e.g., vehicles), fall back to reading the type line via `ConvertedCardLocator.load_text(...)`. Depends on T018.
- [X] T031 [US3] Extend the per-epoch log line in `TrainPickerUseCase.execute` to append the audit correlation (when configured) and the distributional summaries to the existing decomposition + val-reward line (FR-029, FR-030, FR-032) in `src/sealed/application/train_picker.py`. Depends on T029, T030.

**Checkpoint**: Training runs with an auditor configured emit per-epoch correlation + distribution lines; runs without an auditor omit the correlation cleanly. All three stories functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final validation across stories.

- [X] T032 [P] Update `CLAUDE.md` to document the `train-picker` and `pick-decks` subcommands under the `sealed` section (plan.md survey follow-up), including the `models/sealed/picker/` artifact layout (`latest.pt` + `best_{timestamp}.pt`).
- [X] T037 [P] Add `train-picker` and `pick-decks` sections to `README.md`, mirroring the existing per-subcommand sections (`build-decks` ~L622, `train-scorer` ~L576): prerequisites, example invocations, full flag tables (from contracts/cli.md), exit codes, and output/artifact format (`models/sealed/picker/{latest.pt,best_{timestamp}.pt}` and the `LABEL;SET_CODE;...|Card40` deck output). Constitution VI requirement — the new training/inference workflows MUST be documented in the root README in the same PR.
- [X] T033 [P] Add `picker_model.py`, `train_picker.py`, `pick_decks.py`, `picker_store.py` to the "Key modules inside `sealed`" listing in `CLAUDE.md`.
- [X] T034 Run `ruff check src/ tests/` and fix any new lint findings introduced by the picker modules.
- [X] T035 Run the full `pytest` unit suite (`pytest tests/unit/sealed/`) and confirm all picker tests pass; confirm the fast suite stays fast (Principle I).
- [X] T036 Walk through `specs/017-one-shot-deck-picker/quickstart.md` Steps 2–4 against a small real pools file to confirm the documented commands and outputs match the implementation.

> **Note**: FR-031 (baseline cross-scorer correlation over the match-outcomes corpus) and the cold-start sanity check (spec § 3.6) are documented manual procedures, not code — no implementation task. They are described in `quickstart.md` Step 1 and the spec.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS US1 and US2.
- **US1 (Phase 3)**: Depends on Foundational. The MVP.
- **US2 (Phase 4)**: Depends on Foundational (model + walk + store). Independent of US1 at the code level, but a trained checkpoint from US1 is needed for a real end-to-end run.
- **US3 (Phase 5)**: Depends on US1 (it extends the `train-picker` loop and CLI parser). Cannot precede US1.
- **Polish (Phase 6)**: Depends on all implemented stories.

### User Story Dependencies

- **US1 (P1)**: Foundational only. The load-bearing artifact producer.
- **US2 (P2)**: Foundational only at the code level (reuses `PickerModel`, the walk, `PickerStore`). Functionally consumes US1's checkpoint output.
- **US3 (P3)**: Builds directly on US1's `train_picker.py` + CLI parser; not independently deliverable without US1.

### Within Each Story

- Tests are written first and expected to fail before implementation (constitution Principle I).
- Foundational: config → model → walk → store.
- US1: config → sampler → log-prob → reward path → loss → schedule/pool/validate/optimizer/resume → orchestration → CLI.
- US2: config → use case → CLI.
- US3: flag+load → correlation → summaries → log wiring.

### Parallel Opportunities

- T001 (setup) stands alone.
- Foundational: T002/T003 (tests, different files) in parallel; T004 and T007 (config + store, different files) in parallel; T005/T006 are sequential after T004 (same file `picker_model.py`).
- US1 tests T008/T009/T010/T010a are all in `test_train_picker.py` — write together but they share a file, so not strictly `[P]` against each other; they are `[P]` against any non-`train_picker` work.
- Across stories: once Foundational is done, US2's T024 (new file `pick_decks.py`) can proceed in parallel with US1 implementation by a second developer, since US2 only depends on foundational components.
- Polish T032/T033 (both `CLAUDE.md`) are sequential with each other; T034/T035/T036 are sequential (build on the finished code).

---

## Parallel Example: Foundational Phase

```bash
# Tests (different files) in parallel:
Task: "Unit tests for PickerModel + walk in tests/unit/sealed/domain/test_picker_model.py"
Task: "Unit tests for PickerStore in tests/unit/sealed/infrastructure/test_picker_store.py"

# Implementation (different files) in parallel:
Task: "PickerConfig in src/sealed/domain/picker_model.py"
Task: "PickerStore + LoadedPickerCheckpoint in src/sealed/infrastructure/picker_store.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1: Setup (T001).
2. Phase 2: Foundational (T002–T007) — model, walk, store. CRITICAL, blocks everything.
3. Phase 3: US1 (T008–T022) — the full training pipeline.
4. **STOP and VALIDATE**: train a picker on a small pools file, confirm checkpoints + improving val reward.
5. Run the cold-start sanity check (manual, spec § 3.6) before committing to a full 100k-pool run.

### Incremental Delivery

1. Setup + Foundational → model/store/walk ready.
2. US1 → train-picker works → **MVP** (produces the load-bearing artifact).
3. US2 → pick-decks works → picker decks feed self-play.
4. US3 → in-training audits → reward-hacking early warning.
5. Polish → docs + full-suite validation.

### Parallel Team Strategy

After Foundational completes: one developer drives US1 (the critical path); a second can start US2's new files (`pick_decks.py`, its parser) in parallel since US2's code only depends on foundational components. US3 must wait for US1's `train_picker.py` + parser to land.

---

## Notes

- `[P]` = different files, no incomplete dependencies.
- Every task that creates a new entity cites prior art per Constitution Principle VII (Codebase-Aware Planning): `PickerConfig`↔`ScorerConfig`, `PickerModel`↔`SetTransformerScorer`/`SAB`, `PickerStore`↔`ScorerStore`, `TrainPickerConfig`/`UseCase`↔`TrainScorer*`, `PickDecksConfig`/`UseCase`↔`BuildDecks*`. The two parallel concepts (`PickerModel` vs. `GreedyDeckBuilder`, `PickDecksUseCase` vs. `BuildDecksUseCase`) are justified in plan.md — the picker is a different model, not a reimplementation.
- No integration tests: end-of-training Forge validation is a documented manual procedure (spec § "End-of-training Forge validation"); building it as a pytest target would contradict that decision.
- Commit after each task or logical group; reference the task ID.
- Verify tests fail before implementing.
