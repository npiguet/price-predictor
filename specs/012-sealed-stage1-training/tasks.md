# Tasks: Stage 1 Training — Legal Pick Gate

**Input**: Design documents from `/specs/012-sealed-stage1-training/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/cli-train-sample.md ✅, quickstart.md ✅

**Tests**: Specified explicitly in plan.md (unit tests per test table + integration test). All test tasks are included per project constitution.

**Organization**: Tasks grouped by user story. US1 (training loop) → US2 (resume/checkpoints) → US3 (sample command).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1 = Launch Stage 1 Training, US2 = Resume from Checkpoint, US3 = Inspect Model Picks
- Miniaturized test params: `n_slots=4`, `d_model=8`, `n_layers=1`, `n_heads=2`, `card_embed_dim=8`

---

## Phase 1: Setup

**Purpose**: Create new test package directories for the sealed unit test tree introduced by this feature.

- [ ] T001 Create `__init__.py` stubs for new test packages: `tests/unit/sealed/domain/`, `tests/unit/sealed/application/`, `tests/unit/sealed/infrastructure/` (one empty `__init__.py` per directory)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure files required by all three user stories. No user story work can begin until this phase is complete.

**⚠️ CRITICAL**: US1, US2, and US3 all depend on these three components.

- [ ] T002 [P] Implement `PoolTransformerConfig` dataclass and `PoolTransformerModel(nn.Module)` in `src/sealed/domain/pool_transformer.py` — config fields: `n_layers=8`, `n_heads=8`, `d_model=516`, `ff_dim=2048`, `n_slots=96`, `card_embed_dim=512`, `dropout=0.1`; model: `nn.TransformerEncoder` with `batch_first=True` (no positional encoding), output head `nn.Linear(516, 96)`; `forward(slot_features: Tensor[batch, n_slots, 516]) → logits: Tensor[batch, n_slots]`
- [ ] T003 [P] Implement `PoolLoader` in `src/sealed/infrastructure/pool_loader.py` — `load_pools(pools_path: Path) → list[str]` raises `ValueError` if file missing or empty; `assemble_pool_tensor(pool_names: str, cards_path: Path, embedding_store: EmbeddingStore, slot_state: SlotState) → Tensor[96, 516]` loads `.npz` card embeddings, appends 6 basic land slot embeddings (Plains/Island/Swamp/Mountain/Forest/Wastes from `cards_path`), zero-pads to 96 slots, concatenates 4 flags (`picked_flag`, `available_flag`, `is_land`, `basic_land_count`); raises `FileNotFoundError` with card name if any `.npz` is missing
- [ ] T004 [P] Implement `PoolModelStore` in `src/sealed/infrastructure/pool_model_store.py` — `save(path: Path, model, optimizer, training_state, replay_buffer) → None` using atomic temp-file-then-rename pattern; `load(path: Path) → CheckpointData` returns model state dicts, optimizer state, training state, replay buffer; `save_timestamped(base_path: Path, ...) → Path` writes to `base_path.parent/checkpoints/{ISO8601}.pt`; `CheckpointData` dataclass with fields: `pool_transformer_state_dict`, `optimizer_state_dict`, `training_state`, `replay_buffer`

**Checkpoint**: Foundational layer complete — user story implementation can begin.

---

## Phase 3: User Story 1 — Launch Stage 1 Training (Priority: P1) 🎯 MVP

**Goal**: A working end-to-end PPO training loop that collects episodes from the pool dataset, updates the model, saves `latest.pt`, and halts when Stage 1 is complete.

**Independent Test**: Run `python -m sealed train --stage 1` against a small fixture dataset. Verify episodes execute, rewards are logged, and `latest.pt` is written after the first batch.

### Tests for User Story 1

- [ ] T005 [P] [US1] Write unit tests for `PoolTransformerModel` in `tests/unit/sealed/domain/test_pool_transformer.py` — use miniaturized config (`n_slots=4`, `d_model=8`, `n_layers=1`, `n_heads=2`, `card_embed_dim=8`); test: forward pass output shape is `(batch, 4)`, logits cover all 4 slots (no masking), log-probs over full distribution sum to 0
- [ ] T006 [P] [US1] Write unit tests for `PoolLoader` in `tests/unit/sealed/infrastructure/test_pool_loader.py` — test: empty `pools.txt` raises `ValueError`, missing card `.npz` raises `FileNotFoundError` containing the card name, correct 516-dim tensor shape with all flag fields populated
- [ ] T007 [P] [US1] Write unit tests for `Episode` dataclass and `ReplayBuffer` in `tests/unit/sealed/domain/test_replay_buffer.py` — test: FIFO eviction when `len == max_size`, `append`/`sample` behaviour, `sample(n > len)` returns all, `to_list`/`from_list` serialization round-trip preserves all fields
- [ ] T008 [P] [US1] Write unit tests for `EpisodeRunner` in `tests/unit/sealed/domain/test_episode_runner.py` — use miniaturized model (`n_slots=4`); test: legal episode with no repeats completes with `len(actions) == 4`; illegal pick (same pool index twice, even from different shuffled input positions) terminates early; reward formula `(current_run / best_run) × 2 - 1` is correct; two different pool indices occupying the same shuffled input position at different steps are correctly treated as legal
- [ ] T009 [P] [US1] Write unit tests for `PPOTrainer` in `tests/unit/sealed/domain/test_ppo_trainer.py` — use miniaturized model; test: KL divergence warning fires to stdout when per-episode KL exceeds 1.5 nats; PPO loss backward pass produces non-zero gradients on model parameters; `reward_baseline` EMA updates after each processed episode; `TrainBatchResult` fields are populated
- [ ] T010 [P] [US1] Write unit tests for `TrainStage1UseCase` startup validation in `tests/unit/sealed/application/test_train_stage1.py` — test: empty `pools.txt` raises `ValueError` before any training; missing card `.npz` raises `FileNotFoundError`; model-path directory created automatically when absent; fresh start (no checkpoint) initializes model from scratch
- [ ] T011 [P] [US1] Write unit tests for train CLI in `tests/unit/sealed/infrastructure/test_cli_sealed_train.py` — test: `--stage 1` dispatches to `TrainStage1UseCase`; `--batch-size` value is passed through; unknown `--stage` value exits with code 1
- [ ] T012 [US1] Write integration test in `tests/integration/sealed/test_train_stage1_integration.py` — create a tiny fixture: 4 fake `.npz` files with 8-dim embeddings (matching miniaturized config), a `pools.txt` with one 4-card pool; run `TrainStage1UseCase.execute(batch_size=2)` for 2 batches; assert `latest.pt` is written, `best_run >= 1`, no exception raised

### Implementation for User Story 1

- [ ] T013 [P] [US1] Implement `Episode` dataclass and `ReplayBuffer` in `src/sealed/domain/replay_buffer.py` — `Episode` fields: `pool_names: str`, `shuffle_seeds: np.ndarray[40, int32]`, `actions: np.ndarray[n, int32]` (pool indices), `log_probs: np.ndarray[n, float32]`, `reward: float`; `ReplayBuffer`: `max_size=1000`, `append` with FIFO eviction, `sample(n)` without replacement, `__len__`, `to_list`, `from_list`
- [ ] T014 [P] [US1] Implement `EpisodeRunner` in `src/sealed/domain/episode_runner.py` — `run(pool_names, embedding_store, model, cards_path, rng_seed) → Episode`; at each step: applies the step's shuffle seed to permute non-basic-land slots; builds shuffled input tensor with current slot flags; forward pass → logits[n_slots] (no masking); samples a shuffled input position; translates to pool index via inverse permutation; if pool index already in `picked_set` (non-basic-land): terminates; records pool index in `actions` and log-prob of the sampled input position in `log_probs`; reward = `(current_run / best_run) × 2 - 1`
- [ ] T015 [US1] Implement `PPOTrainer` and `TrainBatchResult` in `src/sealed/domain/ppo_trainer.py` — `__init__(model, optimizer, clip_eps=0.2, kl_warn_threshold=1.5)`; `update(episodes, pool_loader, best_run) → TrainBatchResult`: for each episode reconstructs shuffled input tensors from stored seeds, computes new log-probs for stored pool-index actions, computes per-episode KL as `mean(old_log_p - new_log_p)`, prints `[warn] KL divergence {kl:.2f} nats for episode at buffer index {i} — policy has drifted` when threshold exceeded, computes per-step importance ratios, clipped PPO surrogate loss (advantage = `reward - baseline`), backward + optimizer step; `reward_baseline` EMA decay 0.99 updated per episode; `TrainBatchResult`: `mean_reward: float`, `episode_runs: list[int]`, `kl_warnings: int`
- [ ] T016 [US1] Implement `TrainStage1UseCase` in `src/sealed/application/train_stage1.py` — `execute(pools_path, cards_path, model_path, batch_size, set_code) → None`; startup: validate `pools.txt` non-empty (raise `ValueError`), validate all card embeddings present (raise `FileNotFoundError` with card name), create `model_path` directory tree if absent, initialize model + optimizer from scratch; main loop: collect `batch_size` episodes sequentially from pool dataset (loop back at end), append to replay buffer, sample batch, call `PPOTrainer.update`, print `[ep {episode_count}] batch runs: {r0},{r1},...  best_run={best_run}  mean_reward={mean:.3f}`, save `latest.pt` atomically via `PoolModelStore.save`; update `best_run`, `consecutive_successes`; halt and print `Stage 1 complete: 100 consecutive episodes with 40 legal picks. Model saved to {model_path}.` when `consecutive_successes >= 100`
- [ ] T017 [US1] Add `train` subparser and `run_train(args) → int` to `src/sealed/infrastructure/cli.py` — args: `--stage INT` (required), `--set STR` (default `RVR`), `--pools-path PATH` (default `output/sealed/pools/{set}/`), `--cards-path PATH` (default `output/cardsfolder/`), `--model-path PATH` (default `models/sealed/stage1/latest.pt`), `--batch-size INT` (default 32); `run_train` dispatches to `TrainStage1UseCase.execute`; unknown `--stage` exits with code 1; extend `main()` dispatch table

**Checkpoint**: User Story 1 complete — `python -m sealed train --stage 1` runs the full training loop end-to-end.

---

## Phase 4: User Story 2 — Resume Training from Checkpoint (Priority: P2)

**Goal**: An interrupted training run resumes from `latest.pt`, restoring `best_run`, `episode_count`, `consecutive_successes`, `reward_baseline`, and the full replay buffer. Timestamped checkpoints are saved every 1000 episodes.

**Independent Test**: Run training for one batch, kill the process, re-launch with the same `--model-path`, and verify that `episode_count` and `best_run` are restored from the checkpoint rather than reset to initial values.

### Tests for User Story 2

- [ ] T018 [P] [US2] Write unit tests for `PoolModelStore` in `tests/unit/sealed/infrastructure/test_pool_model_store.py` — test: save + load round-trip preserves all fields (`pool_transformer_state_dict`, `optimizer_state_dict`, `training_state`, `replay_buffer`); atomic write (temp file is absent after save completes); `save_timestamped` produces a filename matching ISO 8601 format under `checkpoints/`
- [ ] T019 [US2] Extend `tests/unit/sealed/application/test_train_stage1.py` with checkpoint resume scenarios — test: when a checkpoint exists at `model_path`, `execute()` loads model weights, optimizer state, `best_run`, `episode_count`, `consecutive_successes`, `reward_baseline`, and replay buffer from it instead of initializing from scratch; pool iteration index is NOT restored (always restarts from pool 0)

### Implementation for User Story 2

- [ ] T020 [US2] Extend `TrainStage1UseCase.execute` to resume from checkpoint in `src/sealed/application/train_stage1.py` — on startup: if `model_path` exists, call `PoolModelStore.load(model_path)` and restore model weights, optimizer state, and all `TrainingState` fields; if not, initialize from scratch (already implemented in T016)
- [ ] T021 [US2] Add timestamped checkpoint trigger to `TrainStage1UseCase` in `src/sealed/application/train_stage1.py` — after each episode increments `episode_count`, check `episode_count % 1000 == 0`; if true, call `PoolModelStore.save_timestamped` to write `{model_path.parent}/checkpoints/{ISO8601}.pt`

**Checkpoint**: User Stories 1 and 2 complete — training survives interruption and produces rollback checkpoints.

---

## Phase 5: User Story 3 — Inspect Current Model Picks (Priority: P3)

**Goal**: `python -m sealed sample` loads the current checkpoint and prints N human-readable pick sequences showing card names in pick order plus a SUCCESS or ILLEGAL PICK result line.

**Independent Test**: Run `python -m sealed sample` after any `latest.pt` checkpoint exists. Verify it prints the expected block format for each sample and exits with code 0.

### Tests for User Story 3

- [ ] T022 [P] [US3] Write unit tests for `SampleStage1UseCase` in `tests/unit/sealed/application/test_sample_stage1.py` — test: SUCCESS case prints `Result: SUCCESS (40/40 legal picks)` and all 40 pick lines; ILLEGAL PICK case prints `Result: ILLEGAL PICK at step {n} ({n-1}/40 legal picks)` with correct step number; `n_samples` controls the number of output blocks; missing checkpoint raises `FileNotFoundError`
- [ ] T023 [P] [US3] Write unit tests for sample CLI in `tests/unit/sealed/infrastructure/test_cli_sealed_sample.py` — test: `--n-samples` value is passed through to `SampleStage1UseCase`; missing checkpoint at `model-path` exits with code 2 and message to stderr

### Implementation for User Story 3

- [ ] T024 [US3] Implement `SampleStage1UseCase` in `src/sealed/application/sample_stage1.py` — `execute(pools_path, cards_path, model_path, n_samples) → None`; raises `FileNotFoundError` if checkpoint absent; loads checkpoint via `PoolModelStore.load`; sets `model.eval()` and wraps in `torch.no_grad()`; picks N pools at random from the loaded pool list; for each pool runs one episode via `EpisodeRunner`; prints formatted block: `Sample {n}:`, one `  Pick {i:2d}: {card_name}` line per pick, then `  Result: SUCCESS (40/40 legal picks)` or `  Result: ILLEGAL PICK at step {n} ({run}/40 legal picks)`, followed by a blank line
- [ ] T025 [US3] Add `sample` subparser and `run_sample(args) → int` to `src/sealed/infrastructure/cli.py` — args: `--set STR` (default `RVR`), `--pools-path PATH` (default `output/sealed/pools/{set}/`), `--cards-path PATH` (default `output/cardsfolder/`), `--model-path PATH` (default `models/sealed/stage1/latest.pt`), `--n-samples INT` (default 10); `run_sample` delegates to `SampleStage1UseCase.execute`; missing checkpoint exits with code 2 and message to stderr; extend `main()` dispatch table

**Checkpoint**: All three user stories complete — training, resuming, and sampling all work end-to-end.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T026 [P] Update `CLAUDE.md` with the 012-sealed-stage1-training entry (already partially present — verify tech stack and storage lines are accurate)
- [ ] T027 Validate quickstart.md scenarios against the implemented code: run `python -m sealed train --stage 1` with a small test dataset, then `python -m sealed sample`, verify console output matches the formats shown in `specs/012-sealed-stage1-training/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 — no dependency on US2/US3
- **US2 (Phase 4)**: Depends on Phase 3 (extends `TrainStage1UseCase`) — no dependency on US3
- **US3 (Phase 5)**: Depends on Phase 2 (needs `PoolLoader`, `PoolModelStore`, `PoolTransformerModel`) — no dependency on US2
- **Polish (Phase 6)**: Depends on all user stories

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2 — no dependency on US2 or US3
- **US2 (P2)**: Extends `TrainStage1UseCase` from US1 — depends on US1 (T016 must complete)
- **US3 (P3)**: Depends only on Phase 2 foundational components — can be developed in parallel with US1 and US2 after Phase 2 completes

### Within Each User Story

- Tests written and failing before implementation
- Domain models before services
- Services (use cases) before CLI wiring
- Core implementation before integration

### Parallel Opportunities

- **Phase 2**: T002, T003, T004 — all parallel (different files)
- **Phase 3 tests**: T005–T011 — all parallel; T012 depends on T005–T011 completing
- **Phase 3 domain**: T013, T014 — parallel; T015 depends on T013+T014; T016 depends on T015; T017 depends on T016
- **Phase 4**: T018 parallel with T019; T020 depends on T018+T019; T021 depends on T020
- **Phase 5**: T022, T023 parallel; T024 depends on T022; T025 depends on T023+T024
- **US3 vs US1**: Once Phase 2 is done, US3 tests (T022, T023) can start in parallel with US1 tests

---

## Parallel Example: User Story 1 Tests

```bash
# All US1 unit test files can be written simultaneously (different files):
Task T005: tests/unit/sealed/domain/test_pool_transformer.py
Task T006: tests/unit/sealed/infrastructure/test_pool_loader.py
Task T007: tests/unit/sealed/domain/test_replay_buffer.py
Task T008: tests/unit/sealed/domain/test_episode_runner.py
Task T009: tests/unit/sealed/domain/test_ppo_trainer.py
Task T010: tests/unit/sealed/application/test_train_stage1.py
Task T011: tests/unit/sealed/infrastructure/test_cli_sealed_train.py
```

## Parallel Example: Phase 2 Foundational

```bash
# All three foundational components can be built simultaneously:
Task T002: src/sealed/domain/pool_transformer.py
Task T003: src/sealed/infrastructure/pool_loader.py
Task T004: src/sealed/infrastructure/pool_model_store.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T004)
3. Complete Phase 3: User Story 1 (T005–T017)
4. **STOP and VALIDATE**: Run `pytest tests/unit/sealed/ tests/integration/sealed/` — all tests pass
5. Run the quickstart train command against a real dataset, confirm console output

### Incremental Delivery

1. Phase 1 + Phase 2 → Foundational infrastructure in place
2. Phase 3 (US1) → Full training loop working → **first usable milestone**
3. Phase 4 (US2) → Training survives restarts → **production-ready training**
4. Phase 5 (US3) → Qualitative inspection tool → **complete feature**
5. Phase 6 → Polish and validate

---

## Notes

- `[P]` tasks operate on different files with no outstanding dependencies — safe to run concurrently
- `[Story]` label traces each task back to the user story it fulfils
- Miniaturized test config: `n_slots=4`, `d_model=8`, `n_layers=1`, `n_heads=2`, `card_embed_dim=8` (keep consistent across all unit tests)
- The integration test (T012) must use real `.npz` file I/O — do not mock the file system there (research.md Decision 7)
- `EpisodeRunner` and `PPOTrainer` must not import from `infrastructure/` — PyTorch models are injected via constructor (domain purity constraint from plan.md)
- `pool_dataset_index` is NOT persisted in checkpoints — always restart from pool 0 on resume (clarification in spec.md)
