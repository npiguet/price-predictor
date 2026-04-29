# Tasks: Encoder Fine-Tuning (Phase B) for Sealed Scorer

**Input**: Design documents from `/specs/015-encoder-fine-tuning/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/ (`train-scorer-cli.md`, `encode-cards-cli.md`, `checkpoint-format.md`), quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Single Python project with hexagonal layout under `src/`:
- `src/sealed/{domain,application,infrastructure}/`
- `src/price_predictor/{domain,application,infrastructure}/` (untouched by this feature)
- `tests/unit/sealed/{application,infrastructure}/`, `tests/integration/sealed/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No new packages or modules introduced. Ensure local environment is current before touching code.

- [X] T001 Verify dev environment: activate `.venv`, run `pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126`, then `pytest tests/unit/sealed/ -q` to confirm the unit suite is green pre-feature. No code changes in this task.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Cross-cutting changes that affect both Phase A and Phase B and MUST be in place before any user story work can land. Removes legacy APIs, switches optimizers, adds new persistence keys, and unifies clipping/validation cadence.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. Existing fast tests are updated here so the suite stays green.

### Domain & infrastructure shape changes

- [X] T002 [P] Remove `EmbeddingTable.freeze()`, `EmbeddingTable.unfreeze()`, `EmbeddingTable.is_frozen()` and add `EmbeddingTable.set_text_vectors(indices: LongTensor, text_vectors: FloatTensor) -> None` (writes the leading `2 * d_model` columns of the rows at `indices` from `text_vectors` through a non-leaf assignment so autograd flows back) in `src/sealed/infrastructure/match_data_loader.py`. Prior art: existing `EmbeddingTable.__init__`/`forward` in the same file (research.md `EmbeddingTable` row).

- [X] T003 [P] Extend `LoadedScorerCheckpoint` in `src/sealed/infrastructure/scorer_store.py` with optional fields `encoder_state_dict: dict[str, Any] | None = None`, `encoder_config: dict[str, Any] | None = None`, and `train_config: dict[str, Any] | None = None`. Prior art: existing `LoadedScorerCheckpoint` dataclass in the same file (research.md, data-model.md §LoadedScorerCheckpoint).

- [X] T004 Update `ScorerStore.save_checkpoint` and `ScorerStore.load_checkpoint` in `src/sealed/infrastructure/scorer_store.py` to (a) persist `encoder_state_dict` when supplied (omit the key entirely when `None` so Phase A checkpoints don't carry the field), (b) persist `encoder_config` (`asdict(TransformerConfig)`) when supplied (same omit-when-None rule, MUST be present whenever `encoder_state_dict` is), and (c) always persist `train_config: dict[str, Any]`. Read all three keys back on load. Depends on T003. Prior art: shared `save_checkpoint`/`load_checkpoint` in `src/price_predictor/infrastructure/torch_checkpoint.py` (research.md "Checkpoint persistence").

### `TrainScorerConfig` & training flag surface

- [X] T005 Update `TrainScorerConfig` in `src/sealed/application/train_scorer.py`: add `scorer_checkpoint: Path | None = None`, `encoder_checkpoint: Path = Path("models/price-predictor/transformer/latest.pt")`, `patience: int = 5`; flip `embedding_lr` default to `0.0`; flip `lr` default from `1e-3` to `1e-5` (per `specs/encoder-fine-tuning.md` original brief); remove `unfreeze_embeddings` and `val_interval` fields; add a `phase` property returning `"A"` if `embedding_lr == 0` else `"B"`. Prior art: existing `TrainScorerConfig` dataclass in the same file (research.md, data-model.md §TrainScorerConfig).

- [X] T006 Update `_build_train_scorer_parser` in `src/sealed/infrastructure/cli.py` to (a) remove `--unfreeze-embeddings` and `--val-interval` flag registrations, (b) register **every resumable training flag** (`--epochs`, `--batch-size`, `--lr`, `--embedding-lr`, `--patience`, `--val-fraction`, `--random-seed`, plus the architecture flags `--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`, `--mlp-hidden`, `--dropout`, plus `--outcomes-path`, `--cards-path`, `--checkpoint-dir`) with `default=None` (sentinel — late-resolved in T030 per the resume-precedence rule), (c) register `--encoder-checkpoint` and `--scorer-checkpoint` with `default=None` (Decision §3 carve-out), (d) keep `--resume` as a non-sentinel `default=None` flag. Help text for every flag MUST state purpose, **the effective default after late-resolution** (i.e. the dataclass default for non-resume runs), and any mutual-exclusivity / phase semantics (FR-016). Depends on T005.

### Optimizer, clipping, validation cadence

- [X] T007 Switch `_build_optimizer` in `src/sealed/application/train_scorer.py` from `torch.optim.Adam` to `torch.optim.AdamW` for both branches (single group in Phase A; two groups when an encoder is present — encoder param-group population lands in T024). FR-005a. Prior art: existing `_build_optimizer` in the same file (research.md "Optimizer with multiple parameter groups").

- [X] T008 In `_train_one_epoch` (`src/sealed/application/train_scorer.py`), apply `torch.nn.utils.clip_grad_norm_(group_params, max_norm=1.0)` once per `optimizer.param_groups` entry between `loss.backward()` and `optimizer.step()`. Replace the existing scorer-only clipping. Phase A has one group, Phase B has two — both are clipped independently (FR-008). Update `EpochStats.grad_norms` to be a `dict[str, float]` keyed `"scorer"` and (Phase B only) `"encoder"`, where each value is the **pre-clip** combined L2 norm returned by `clip_grad_norm_` (the value the function returns *is* the pre-clip total — post-clip would be bounded by 1.0 and carry no diagnostic signal per FR-012). Prior art: existing per-component `_gradient_norms()` in the same file (research.md `EpochStats` row).

- [X] T009 In `TrainScorerUseCase.execute` (`src/sealed/application/train_scorer.py`), remove the `val_interval` gate so validation runs once at end of every epoch unconditionally, and add `--patience`-driven early stopping: stop training after `patience` consecutive epochs without a new peak `val_acc`; preserve the best checkpoint as `best_*.pt` (FR-011, Decision §6). Default `patience` is `5` (set on `TrainScorerConfig` in T005).

### Self-describing checkpoint plumbing

- [X] T010 In the `_save_checkpoint` callsite inside `src/sealed/application/train_scorer.py`, build the `train_config` dict from the active `TrainScorerConfig` (`asdict(...)` with `Path`s coerced to strings, `None`s preserved) and pass it through to `ScorerStore.save_checkpoint`. The dict MUST include every flag in `data-model.md §train_config Schema`. Depends on T004, T005.

### Existing test suite alignment

- [X] T011 [P] Update `tests/unit/sealed/application/test_train_scorer.py`: drop fixtures and assertions referencing `--unfreeze-embeddings` and `--val-interval`; rename uses to `--embedding-lr 0` (Phase A) where the prior `unfreeze_embeddings=False` semantics apply. Existing Phase A behavior tests stay green; new Phase B tests land in Phase 3.

- [X] T012 [P] Update `tests/unit/sealed/infrastructure/test_match_data_loader.py`: drop tests covering `EmbeddingTable.freeze/unfreeze/is_frozen`. Add a focused unit test for `EmbeddingTable.set_text_vectors` asserting (a) the leading `2 * d_model` columns of the rows at `indices` are overwritten, (b) the trailing deterministic-feature slice is untouched, (c) gradient flows from a downstream loss back through `text_vectors` to its source tensor. Depends on T002.

- [X] T013 [P] Update `tests/unit/sealed/infrastructure/test_cli.py`: drop `--unfreeze-embeddings` and `--val-interval` references in `train-scorer` test fixtures. (The `--encoder-path` → `--encoder-checkpoint` rename for `encode-cards` test fixtures lands in Phase 4 alongside the actual flag rename — see T049.)

- [X] T014 Update `tests/unit/sealed/infrastructure/test_scorer_store.py`: existing Phase A round-trip continues to pass with the new `train_config` field populated and `encoder_state_dict`, `encoder_config` absent. Depends on T003, T004.

- [X] T047 [P] Add deterministic train/val split test in `tests/unit/sealed/application/test_train_scorer.py` (FR-011a, research.md Decision §10): given the same `match-outcomes.txt` content and `random_seed=42`, two invocations of `_load_dataset` produce identical train/val index sets. Asserts existing behavior — no implementation change — locking it down so future refactors don't accidentally non-determinize the split.

**Checkpoint**: Foundation ready — both phases now use AdamW with per-group max-norm 1.0 clipping, validation runs once per epoch, `--patience` drives early stopping, `EmbeddingTable.set_text_vectors` exists, and every checkpoint persists `train_config`. User story implementation can now begin in parallel.

---

## Phase 3: User Story 1 — Train Phase B with encoder fine-tuning (Priority: P1) 🎯 MVP

**Goal**: A single `train-scorer` invocation with `--scorer-checkpoint <phaseA>.pt --encoder-checkpoint <pp>.pt --embedding-lr <nonzero>` runs Phase B end-to-end: scorer + encoder are jointly trained, encoder gradient norms and `embedding_drift` are logged each epoch, and the saved best checkpoint contains both `scorer.state_dict` and `encoder.state_dict`.

**Independent Test**: Run Phase A to early-stop, then `train-scorer --scorer-checkpoint <best_phaseA>.pt --encoder-checkpoint <pp>.pt --embedding-lr 1e-7`. Verify: (a) the resulting checkpoint contains both `model_state_dict` and `encoder_state_dict`, (b) training completes within the `--patience` window, (c) per-epoch encoder gradient norms and `embedding_drift` values are non-zero. Cross-phase resume, bare `--embedding-lr` without bootstrap, and architecture flags alongside `--scorer-checkpoint` are all rejected with clear errors.

### Tests for User Story 1 (write FIRST, ensure they FAIL before implementation) ✅

- [X] T015 [P] [US1] Add CLI rejection tests in `tests/unit/sealed/infrastructure/test_cli.py`: (a) `--resume <phaseA>` + `--embedding-lr 1e-7` rejected (cross-phase, FR-004), (b) `--resume <phaseB>` + `--embedding-lr 0` rejected (cross-phase, FR-004), (c) `--resume <phaseB>` + explicit `--encoder-checkpoint` rejected (FR-004 carve-out), (d) `--resume` + `--scorer-checkpoint` rejected (mutually exclusive, FR-003a), (e) `--scorer-checkpoint` + any architecture flag rejected with the offending flag name in the error (FR-003a), (f) bare `--embedding-lr 1e-7` (no `--scorer-checkpoint`, no `--resume`) rejected (FR-004a).

- [X] T016 [P] [US1] Add architecture-inheritance test in `tests/unit/sealed/application/test_train_scorer.py`: when `--scorer-checkpoint <phaseA>.pt` bootstraps a fresh Phase B run, the constructed scorer's architecture matches the checkpoint's stored `config` and the CLI architecture-flag defaults are ignored (FR-003a).

- [X] T017 [P] [US1] Add AdamW two-group dispatch test in `tests/unit/sealed/application/test_train_scorer.py`: in Phase A `_build_optimizer` returns one param group; in Phase B it returns two groups (scorer at `lr`, encoder at `embedding_lr`) with the encoder group containing `CardPriceTransformerModel` parameters (FR-005, FR-005a).

- [X] T018 [P] [US1] Add within-batch encoder cache test in `tests/unit/sealed/application/test_train_scorer.py`: with a mock encoder counting forward calls, a batch containing card X three times triggers exactly one encoder forward for card X, but autograd accumulates gradients from all three references into the encoder parameters (FR-007).

- [X] T019 [P] [US1] Add reference-batch drift test in `tests/unit/sealed/application/test_train_scorer.py`: run two Phase B steps on a tiny synthetic corpus + tiny encoder; assert the captured `ReferenceBatch.step0_text_vectors` has shape `(num_unique, 2 * d_model)`, that step-0 drift is `0.0`, and that drift after one optimizer step is `> 0.0` (FR-012, Decision §5).

- [X] T020 [P] [US1] Add `--patience` early-stopping test in `tests/unit/sealed/application/test_train_scorer.py`: synthetic loop where validation accuracy peaks at epoch 1 and never improves; with `--patience 3` training stops after epoch 4 (1 + 3) and `best_*.pt` reflects epoch 1 (FR-011).

- [X] T021 [P] [US1] Add Phase B checkpoint round-trip test in `tests/unit/sealed/infrastructure/test_scorer_store.py`: save a payload that includes `encoder_state_dict` and `train_config`; reload and assert both are present and structurally equal to what was saved. Phase A round-trip (no encoder key) continues to pass (FR-009). Depends on T004.

- [X] T022 [P] [US1] Add `train_config`-in-checkpoint test in `tests/unit/sealed/application/test_train_scorer.py`: a Phase B run produces a `best_*.pt` whose `train_config` dict contains `embedding_lr`, `lr`, `patience`, every architecture flag, `scorer_checkpoint`, `encoder_checkpoint`, and `resume` keys (FR-009, Decision §1).

- [X] T048 [P] [US1] Add resume-precedence test in `tests/unit/sealed/application/test_train_scorer.py` (FR-010, contract `checkpoint-format.md §Resume Precedence`): given a Phase B checkpoint whose stored `train_config` has `lr=2e-5` and `patience=10`, assert that (a) `--resume <ckpt>` with no other flags resumes with `lr=2e-5` and `patience=10` (the resumed config wins over the dataclass defaults), (b) `--resume <ckpt> --lr 7e-6` resumes with `lr=7e-6` and `patience=10` (explicit CLI wins for `lr` only; `patience` falls through to the resumed value), and (c) the new run's saved `train_config` reflects the resolved values (`lr=7e-6`, `patience=10`).

### Implementation for User Story 1

- [X] T023 [P] [US1] Add `CardEncoder.encode_batch_text(input_ids: LongTensor, attention_mask: LongTensor, *, with_grad: bool) -> FloatTensor` in `src/sealed/domain/card_encoder.py`. Returns a `(B, 2 * d_model)` text-vector slice (no deterministic-feature concat). When `with_grad=True`, runs without `torch.no_grad()` and uses `CardPriceTransformerModel._encode_and_pool` directly so gradients flow into the encoder parameter group. The existing single-card `encode()` method stays as-is (used by `encode-cards`). Prior art: existing `CardEncoder.encode` and `CardPriceTransformerModel.encode` (research.md `CardEncoder` and `CardPriceTransformerModel.encode` rows).

- [X] T024 [US1] Extend `_TrainingContext` and `ResumeState` in `src/sealed/application/train_scorer.py`. `_TrainingContext` gains `encoder: CardPriceTransformerModel | None`, `tokenizer: MtgTokenizer | None`, `card_token_cache: dict[int, tuple[Tensor, Tensor]] | None`, `reference_batch: ReferenceBatch | None`, `train_config: dict[str, Any]`. `ResumeState` gains `encoder_state_dict: dict | None` and `phase: Literal["A", "B"]`. Add a private `ReferenceBatch` dataclass in the same file (fields per `data-model.md §ReferenceBatch`). Depends on T005.

- [X] T025 [US1] Implement `_resume_or_build_model` in `src/sealed/application/train_scorer.py` to support all bootstrap paths: (a) `--resume <phaseA>` with `embedding_lr == 0` — Phase A continuation (existing behavior); (b) `--resume <phaseB>` with `embedding_lr != 0` — Phase B continuation, encoder weights loaded from the resumed checkpoint's `encoder_state_dict`; (c) `--scorer-checkpoint <phaseA>` with `embedding_lr != 0` — Phase B kickoff, scorer weights from `--scorer-checkpoint`, encoder weights from `--encoder-checkpoint`, optimizer state / `epoch` / `best_val_accuracy` reset; (d) no resume / no scorer-checkpoint — Phase A from scratch (existing behavior). Cross-phase resume MUST raise with the exact message in `contracts/train-scorer-cli.md` (FR-004). Architecture is read from the loaded checkpoint's `config` in cases (a)-(c). Depends on T024.

- [X] T026 [US1] Update `_build_optimizer` in `src/sealed/application/train_scorer.py` so that when an encoder is present the second AdamW param group is `encoder.parameters()` at `embedding_lr` (replacing the prior `EmbeddingTable.parameters()` second group). When no encoder is present (Phase A), the existing single-group dispatch stands. Depends on T007, T024.

- [X] T027 [US1] Implement encoder forward + within-batch cache in `_train_one_epoch` (`src/sealed/application/train_scorer.py`). For each batch in Phase B: (a) collect the unique embedding-table row indices from `winner_indices ∪ loser_indices`; (b) populate `card_token_cache[row]` for any rows seen for the first time using the tokenizer + `ConvertedCardLocator.load_text`; (c) stack the unique cards' `(input_ids, attention_mask)` and call `encoder.encode_batch_text(..., with_grad=True)` exactly once per batch; (d) write the resulting text vectors into `EmbeddingTable.set_text_vectors(unique_indices, text_vectors)`; (e) proceed with the existing scorer forward + Bradley-Terry loss + backward path. The cache `dict` is local to the per-step scope so it goes out of scope after `optimizer.step()` (FR-007, Decision §4). Depends on T002, T023, T024, T026.

- [X] T028 [US1] Implement reference-batch capture at step 0 of Phase B in `_train_one_epoch` (`src/sealed/application/train_scorer.py`): on the first Phase B batch, after the encoder forward but before `optimizer.step()`, store `card_indices`, `input_ids`, `attention_mask`, and `step0_text_vectors = text_vectors.detach().clone()` on the training device into `ctx.reference_batch`. Log a single line recording the reference batch size (number of unique cards). Depends on T024, T027.

- [X] T029 [US1] Implement the new drift metric helper in `src/sealed/application/train_scorer.py`: a private `_embedding_drift(ctx: _TrainingContext) -> float` that, when `ctx.reference_batch is not None`, runs the encoder under `model.eval()` over the cached `(input_ids, attention_mask)` and returns `(current - step0).norm(dim=-1).mean().item()`. Replaces the prior lookup-table snapshot logic. Append the value to `TrainingMetrics.embedding_drifts` once per epoch, and pipe it (plus the encoder grad-norm computed in T008) into `_print_epoch_report`. In Phase A, `embedding_drifts` stays empty and the encoder line is omitted (FR-012). Depends on T024, T028.

- [X] T030 [US1] Implement `run_train_scorer` validations and resume-precedence resolution in `src/sealed/infrastructure/cli.py`. Order: (1) reject `--resume` + `--scorer-checkpoint` (FR-003a); (2) when `args.scorer_checkpoint` or `args.resume` is set, reject any architecture flag whose namespace value is not `None` — i.e. explicitly passed by the user (FR-003a + checkpoint-format.md §Resume Precedence: architecture is loaded from the resumed/bootstrap checkpoint's `config` and `encoder_config`, never overridden); (3) load the resumed/scorer checkpoint metadata if either is set; (4) **resolve every other training flag by precedence (FR-010, checkpoint-format.md §Resume Precedence)**: for each resumable field, if `args.<field> is not None` use the CLI value; else if `--resume` is set and the field is in the resumed `train_config` use that value; else fall back to the dataclass default from `TrainScorerConfig`. Architecture fields are always taken from the resumed/bootstrap checkpoint's `config` (and `encoder_config` for the encoder); (5) **after** resume-precedence resolution, evaluate the run's effective `embedding_lr`: when non-zero, require `args.scorer_checkpoint` xor `args.resume` (FR-004a); (6) reject cross-phase resume by comparing the resumed checkpoint's phase (presence of `encoder_state_dict`) against the resolved `embedding_lr`'s phase (FR-004); (7) when `args.encoder_checkpoint is not None and args.resume is not None and resumed-checkpoint-is-Phase-B`, reject (FR-004 carve-out); (8) late-resolve `args.encoder_checkpoint` to the literal default `models/price-predictor/transformer/latest.pt` only when None and not in resumed `train_config` (Decision §3). Construct `TrainScorerConfig` only after all checks and resolutions pass. Depends on T005, T006.

- [X] T031 [US1] Wire encoder + tokenizer construction into the bootstrap path in `src/sealed/application/train_scorer.py`. When `phase == "B"`: (a) for the `--scorer-checkpoint` path, load `TransformerConfig` + `CardPriceTransformerModel` via `transformer_store.load_model(encoder_checkpoint)` (the price-predictor file, default `models/price-predictor/transformer/latest.pt`); (b) for the `--resume` path, construct `CardPriceTransformerModel` from `TransformerConfig(**resumed_checkpoint.encoder_config)` and populate weights via `model.load_state_dict(resumed_checkpoint.encoder_state_dict)` — **the resumed checkpoint is self-contained**, no dependency on the price-predictor `latest.pt` being unchanged. `.to(device)` in both cases. Load `MtgTokenizer` via `load_tokenizer(vocab_path)` (use the same default path as `encode-cards`). Depends on T003, T024, T025.

- [X] T032 [US1] Persist Phase B checkpoints in `src/sealed/application/train_scorer.py`: at every `_save_checkpoint` callsite, when `ctx.encoder is not None`, pass both `encoder.state_dict()` and `asdict(encoder.config)` (the `TransformerConfig`) through to `ScorerStore.save_checkpoint`; otherwise pass `None` for both so the keys are omitted (Phase A). Always pass `train_config` (built in T010). The `train_config` saved during Phase B must reflect the resolved (post-precedence) flag values for this run, including any CLI overrides on top of the resumed `train_config` (FR-010). Depends on T004, T010, T024.

**Checkpoint**: User Story 1 is fully functional and testable independently — Phase B training runs end-to-end, produces a Phase B checkpoint, and rejects every invalid invocation shape with the contract's exact error messages.

---

## Phase 4: User Story 2 — Refresh cached card embeddings from a Phase B scorer (Priority: P1)

**Goal**: After Phase B finishes, `encode-cards --scorer-checkpoint <phaseB>.pt --clean` rewrites every `.npz` under `output/cardsfolder/` using encoder weights extracted from the scorer checkpoint's `encoder_state_dict`. The default `--encoder-checkpoint` flow is unchanged; Phase A scorer checkpoints are rejected with a clear error.

**Independent Test**: With a Phase B `best_*.pt` (or a hand-crafted checkpoint payload containing an `encoder_state_dict`), run `encode-cards --scorer-checkpoint <ckpt> --clean` against a tiny `output/cardsfolder/` fixture. Verify every `.txt` produced a sibling `.npz` of shape `(2 * d_model + FEATURE_COUNT,)`, and that the new vectors differ from the pre-Phase-B baseline. `encode-cards --scorer-checkpoint <phaseA>.pt` rejects with the contract's error message; passing both `--encoder-checkpoint` and `--scorer-checkpoint` explicitly rejects with the contract's mutual-exclusivity message.

### Tests for User Story 2 (write FIRST, ensure they FAIL before implementation) ✅

- [X] T049 [P] [US2] Rename every `--encoder-path` reference in `encode-cards` test fixtures within `tests/unit/sealed/infrastructure/test_cli.py` (and any related fixture files) to `--encoder-checkpoint`, paired with the actual flag rename in T036. Land both in the same commit/PR so the test suite stays green at every checkpoint. Originally bundled into T013 in Phase 2; moved here per the resolution of speckit.analyze finding I2.

- [X] T033 [P] [US2] Add `encode-cards` mutual-exclusivity test in `tests/unit/sealed/infrastructure/test_cli.py`: explicit `--encoder-checkpoint` + explicit `--scorer-checkpoint` rejected; `--scorer-checkpoint` alone (default `--encoder-checkpoint` not explicitly passed) accepted (FR-013 carve-out).

- [X] T034 [P] [US2] Add Phase A rejection test in `tests/unit/sealed/infrastructure/test_cli.py`: `encode-cards --scorer-checkpoint <phaseA>.pt` (no `encoder_state_dict` in the payload) rejected with the contract error message pointing at `--encoder-checkpoint` (FR-014).

- [X] T035 [P] [US2] Add `encode-cards --scorer-checkpoint` happy-path test in `tests/unit/sealed/application/test_encode_cards.py`: with a hand-crafted Phase B checkpoint payload (containing `encoder_state_dict`, `encoder_config`, `config`, etc.), `run_encode_cards` constructs the encoder from `encoder_config` and loads weights from `encoder_state_dict`; every `.txt` under a tiny fixture produces a `.npz` (FR-014, FR-015). The fixture MUST include at least one card whose name is **not** referenced anywhere in the synthetic Phase B "training" payload (i.e. a never-seen card per US2 acceptance scenario 3); assert that card's `.npz` is produced with the expected `(2 * d_model + FEATURE_COUNT,)` shape.

### Implementation for User Story 2

- [X] T036 [US2] Rename `encode-cards` flag `--encoder-path` to `--encoder-checkpoint` in `_build_encode_cards_parser` (`src/sealed/infrastructure/cli.py`). Register with `default=None` (late-resolved to `models/price-predictor/transformer/latest.pt` after the conflict check). Update help text per FR-016 (Decision §7).

- [X] T037 [US2] Add `--scorer-checkpoint` flag to `_build_encode_cards_parser` in `src/sealed/infrastructure/cli.py`. `default=None`, no value when omitted. Help text mentions mutual exclusivity with `--encoder-checkpoint` per the contract (FR-013, FR-016).

- [X] T038 [US2] Implement `run_encode_cards` validations in `src/sealed/infrastructure/cli.py`. Order: (1) when both `args.encoder_checkpoint is not None and args.scorer_checkpoint is not None`, reject with the contract's mutual-exclusivity message (FR-013); (2) when `args.scorer_checkpoint` is set, load it via `ScorerStore.load_checkpoint`, and if `loaded.encoder_state_dict is None`, reject with the contract's Phase-A-checkpoint message (FR-014); (3) late-resolve `args.encoder_checkpoint` to the literal default when `None` (Decision §3, mirror of T030 step 6). Depends on T003, T036, T037.

- [X] T039 [US2] In `run_encode_cards` (`src/sealed/infrastructure/cli.py`), when the user supplied `--scorer-checkpoint`, construct `CardPriceTransformerModel` from `TransformerConfig(**loaded.encoder_config)` and call `model.load_state_dict(loaded.encoder_state_dict)` to populate weights — **the Phase B checkpoint is self-contained**, the price-predictor `latest.pt` is not touched in this path. The `EncodeCardsUseCase` itself remains unchanged (it only sees a configured `CardEncoder`) per research.md "Reuse" note. Depends on T038.

**Checkpoint**: User Stories 1 AND 2 work independently. The Phase B checkpoint produced by US1 can be fed straight into `encode-cards --scorer-checkpoint` to refresh the `.npz` cache.

---

## Phase 5: User Story 3 — Verify Phase B against Phase A on the deployment metric (Priority: P2)

**Goal**: Use the existing `evaluate-scorer` to compare Phase A and Phase B checkpoints head-to-head on the same pool set; reverting to Phase A if Phase B regresses requires no code changes (SC-003).

**Independent Test**: Given two checkpoints (`best_phaseA.pt`, `best_phaseB.pt`) and a refreshed `.npz` cache for each, `evaluate-scorer --checkpoint <ckpt> --set <SET>` runs against each and reports win rate, match counts, and per-checkpoint statistics in a directly comparable form.

### Implementation for User Story 3

- [X] T040 [US3] **Pre-verify** that `evaluate-scorer` tolerates Phase B checkpoints before walking the quickstart Step 3 path. (a) Read `evaluate-scorer`'s checkpoint-loading code (`src/sealed/application/evaluate_scorer.py` and the `ScorerStore.load_checkpoint` callsite) and confirm it (i) ignores unknown checkpoint keys (`encoder_state_dict`, `encoder_config`, `train_config`) and (ii) does not crash when those keys are present — the `LoadedScorerCheckpoint` extension in T003 leaves them optional, but `evaluate-scorer` may unpack the payload in a way that breaks. Add a minimal test to `tests/unit/sealed/application/test_evaluate_scorer.py` (or the closest existing file) that hands a Phase B payload to the loader. (b) Then walk the quickstart §Step 3 path: run `python -m sealed evaluate-scorer --checkpoint <phaseB>.pt --set <SET>` against a tiny fixture or a 1-pool smoke run and confirm it reports win-rate / match-count statistics in a directly comparable form to a Phase A run on the same pool set. If pre-verification reveals `evaluate-scorer` actually needs a code change, raise that as a CRITICAL finding and re-plan before declaring US3 complete.

**Checkpoint**: All P1 stories plus the deployment-metric comparison are independently verified. Phase B can be promoted or reverted with cache-only changes (SC-003).

---

## Phase 6: User Story 4 — Detect runaway encoder drift early (Priority: P2)

**Goal**: While Phase B trains, the user observes `embedding_drift` and per-group encoder gradient norm at every epoch boundary, with per-group max-norm 1.0 clipping capping single-step encoder spikes before they reach the optimizer.

**Independent Test**: A short Phase B smoke run (2–3 epochs on a tiny synthetic corpus + tiny encoder) shows `embedding_drift` increasing monotonically from `0.0`, the encoder grad norm logged at end of every epoch, and confirms that artificially inflating one batch's encoder gradients caps the post-clip norm at `1.0`.

### Tests for User Story 4 ✅

- [X] T041 [P] [US4] Add per-group clipping behavior test in `tests/unit/sealed/application/test_train_scorer.py`: with a fabricated encoder gradient whose pre-clip norm is `> 10.0`, after `_train_one_epoch`'s clipping pass the encoder param-group's gradient norm is `≤ 1.0` and the scorer param-group's gradient norm is independently clipped (FR-008). Builds on T008.

- [X] T042 [P] [US4] Add end-of-epoch logging contract test in `tests/unit/sealed/application/test_train_scorer.py`: a 2-epoch Phase B run writes `embedding_drifts` of length 2 and writes both `"scorer"` and `"encoder"` keys in `EpochStats.grad_norms` for each epoch; a Phase A run writes empty `embedding_drifts` and a `"scorer"`-only `grad_norms` (FR-012, contract `train-scorer-cli.md §End-of-Epoch Logging Contract`). Also assert the encoder norm logged is the **pre-clip** total — fabricate an encoder gradient with pre-clip norm `~5.0` and verify the logged value is `~5.0` (not the post-clip `1.0` ceiling).

### Implementation for User Story 4

US4 has no incremental implementation beyond US1's drift metric (T029) and Phase 2's foundational clipping (T008). The two tests above lock in the contract. `_print_epoch_report` already emits the values per T029.

**Checkpoint**: All four user stories are independently functional and testable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Slow integration coverage, help-text audit, documentation, and a final quickstart smoke pass.

- [X] T043 [P] Add slow integration smoke test in `tests/integration/sealed/test_phase_b_smoke.py` (marked `@pytest.mark.integration`). Build a tiny synthetic `match-outcomes.txt` + tiny `output/cardsfolder/` fixture + tiny encoder + tiny scorer. Run Phase A for 1 epoch, then Phase B (`--scorer-checkpoint <phaseA> --encoder-checkpoint <pp> --embedding-lr 1e-7`) for 2 epochs. Verify (a) the Phase B `best_*.pt` contains both `model_state_dict` and `encoder_state_dict`, (b) `encode-cards --scorer-checkpoint <phaseB> --clean` rewrites every fixture `.npz` and the new vectors differ from the pre-run baseline, (c) the run respects `--patience`. (Plan §Phase B integration test.)

- [X] T044 [P] Audit `python -m sealed train-scorer --help` and `python -m sealed encode-cards --help` to confirm every new/changed flag (`--embedding-lr`, `--scorer-checkpoint`, `--encoder-checkpoint`, `--patience` on `train-scorer`; `--encoder-checkpoint`, `--scorer-checkpoint` on `encode-cards`; removal of `--unfreeze-embeddings`, `--val-interval`, `--encoder-path`) has purpose, default, and any mutual-exclusivity / phase semantics in the help string (FR-016).

- [X] T045 Update CLAUDE.md (project root) so the `train-scorer` and `encode-cards` subcommand descriptions match the post-feature behavior: mention `--scorer-checkpoint`, `--encoder-checkpoint`, `--patience` on `train-scorer`; the `--encoder-path` → `--encoder-checkpoint` rename and `--scorer-checkpoint` on `encode-cards`; and the Phase A vs Phase B distinction. (Plan §Constitution Check VI.)

- [ ] T046 Run the quickstart end-to-end against real data per `specs/015-encoder-fine-tuning/quickstart.md` (Phase A → Phase B → `encode-cards --scorer-checkpoint --clean` → `evaluate-scorer`). Confirm SC-001 (no manual intervention between steps), SC-004 (every `.npz` refreshed), and SC-005 (drift observable each epoch). No code changes expected.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup. **BLOCKS all user stories.**
- **User Story 1 (Phase 3)**: Depends on Foundational complete.
- **User Story 2 (Phase 4)**: Depends on Foundational complete. Independent of US1 (its tests use hand-crafted Phase B payloads).
- **User Story 3 (Phase 5)**: Depends on US1 + US2 complete (needs a real Phase B checkpoint and a refreshed `.npz` cache to verify the deployment metric).
- **User Story 4 (Phase 6)**: Depends on Foundational + US1 (US1 implements the drift metric and encoder grad-norm logging that US4's tests assert on).
- **Polish (Phase 7)**: Depends on all four user stories.

### Within Foundational (Phase 2)

- T002, T005, T006 (data shape + dataclass + CLI surface) MUST land before T011, T012, T013, T014, T047 (existing-test updates and the deterministic-split assertion). Within these subgroups [P] markers indicate parallelizable tasks.
- T003 → T004 (load before save).
- T007 (AdamW) → T008 (clipping over `optimizer.param_groups`).
- T010 (build `train_config` dict) depends on T005.

### Within Each User Story

- Tests are written first and MUST fail before implementation (Constitution Principle I).
- `card_encoder.py` extension (T023) before `_train_one_epoch` encoder forward (T027).
- `_TrainingContext` extension (T024) before `_resume_or_build_model` updates (T025).
- `_resume_or_build_model` (T025) and `_build_optimizer` (T026) before encoder forward integration (T027).
- Reference-batch capture (T028) before drift helper (T029).
- CLI validations + resume-precedence resolution (T030) and encoder construction (T031) before Phase B checkpoint persistence (T032).

### Parallel Opportunities

- All [P] tests within a user story can run in parallel (different test files / different test classes within the same file).
- T002, T003 are in different files and can run in parallel.
- T011, T012, T013, T014 are in different test files and can run in parallel.
- T015–T022 are independent test cases (different test functions) and can be drafted in parallel.
- T023 (domain extension) is independent of T024–T032 (application/infrastructure) and can land first.
- T033, T034, T035 are different test files / functions and can run in parallel.
- T041, T042 are independent test functions and can run in parallel.
- T043, T044 are independent and can run in parallel.

---

## Parallel Example: Foundational Phase

```bash
# Three different files, no shared edit surface — run together:
Task: "Remove EmbeddingTable.freeze/unfreeze/is_frozen and add set_text_vectors in src/sealed/infrastructure/match_data_loader.py"  # T002
Task: "Extend LoadedScorerCheckpoint in src/sealed/infrastructure/scorer_store.py"                                                  # T003
Task: "Drop --unfreeze-embeddings/--val-interval references in tests/unit/sealed/application/test_train_scorer.py"                  # T011
```

## Parallel Example: User Story 1 Tests

```bash
# All in different test files / different test functions — run together:
Task: "CLI rejection tests for cross-phase resume / mutual-exclusivity in tests/unit/sealed/infrastructure/test_cli.py"  # T015
Task: "Architecture-inheritance test in tests/unit/sealed/application/test_train_scorer.py"                              # T016
Task: "AdamW two-group dispatch test in tests/unit/sealed/application/test_train_scorer.py"                              # T017
Task: "Within-batch encoder cache test in tests/unit/sealed/application/test_train_scorer.py"                            # T018
Task: "Reference-batch drift test in tests/unit/sealed/application/test_train_scorer.py"                                 # T019
Task: "Patience early-stopping test in tests/unit/sealed/application/test_train_scorer.py"                               # T020
Task: "Phase B checkpoint round-trip test in tests/unit/sealed/infrastructure/test_scorer_store.py"                      # T021
Task: "train_config-in-checkpoint test in tests/unit/sealed/application/test_train_scorer.py"                            # T022
```

---

## Implementation Strategy

### MVP First (User Story 1 + User Story 2)

US1 and US2 are both P1 because Phase B improvements are invisible without `encode-cards` refreshing the `.npz` cache. The MVP slice is: Phase 1 → Phase 2 → US1 + US2 in parallel → Phase 7 §T043 (smoke test).

1. Phase 1: Setup.
2. Phase 2: Foundational (CRITICAL — blocks all stories).
3. Phase 3 (US1) and Phase 4 (US2) — runnable in parallel by two developers.
4. **STOP and VALIDATE**: confirm both stories pass their independent tests.
5. Phase 7 T043 (slow integration smoke test) ties them together.

### Incremental Delivery

1. Setup + Foundational → Foundation ready.
2. US1 → Phase B trains end-to-end → can compare scorer checkpoints offline.
3. US2 → cache refresh → downstream tools see Phase B improvements.
4. US3 → deployment-metric verification → ship/revert decision.
5. US4 → drift observability locked in.
6. Polish → integration smoke + docs + quickstart pass.

### Parallel Team Strategy

With multiple developers post-Foundational:

- Developer A: US1 (`train_scorer.py`, `card_encoder.py`, `match_data_loader.py`)
- Developer B: US2 (`cli.py` `encode-cards` surface)
- Developer C: US4 tests (read-only; depend on US1 implementation landing).
- Developer D (optional): Phase 7 polish in parallel with US3.

US3 is a procedural verification and can be performed by any developer once US1 + US2 land.

---

## Notes

- Tests-first per Constitution Principle I: every implementation task in Phases 3–6 has at least one failing test in the same phase that justifies it.
- Foundational tests (T011–T014) are **updates to existing tests**, not new tests — they keep the suite green after the foundational refactor and are completed *with* the foundational implementation.
- Codebase-Aware Planning (Principle VII): each implementation task above either reuses an existing concept (cited inline) or extends one named in `research.md §Codebase Survey`. No parallel concepts are introduced.
- Pre-feature Phase A checkpoints carry `Adam` optimizer state and are explicitly **not supported** for resume after this feature (spec § Assumptions); no migration shim is required.
- `--val-interval` and `--unfreeze-embeddings` are removed without aliases; existing shell scripts that pass them MUST be updated.
- `Path`-typed fields in `train_config` are serialized as strings (data-model.md §train_config Schema).
