---
description: "Task list for draft-agent gen-2 RL self-play fine-tuning"
---

# Tasks: Draft Agent — RL Self-Play Fine-Tuning (Generation 2)

**Input**: Design documents from `/specs/020-draft-agent-rl/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/train-draft-agent-rl.md, quickstart.md

**Tests**: MANDATORY per Constitution Principle I (Fast Automated Tests). Test tasks are written before the implementation they cover and must FAIL first.

**Organization**: Tasks are grouped by user story. **US1 (the RL trainer) is the entire new-code surface and the MVP.** US2 (yardstick) and US3 (self-play loop) introduce **no code** — they are operator runbooks over existing commands (spec Assumptions; quickstart.md), so their tasks are documentation + end-to-end runbook validation only.

## Path Conventions

Single project, hexagonal: `src/draft/{domain,application,infrastructure}`, tests under `tests/unit/draft/` and `tests/integration/`. Dependency direction unchanged: `draft` → `sealed` → `price_predictor`.

---

## Phase 1: Setup

**Purpose**: Module scaffolding; no new dependencies (reuses existing `torch`/`numpy` + `draft`/`sealed`/`price_predictor`).

- [x] T001 Create `src/draft/application/train_draft_agent_rl.py` skeleton (module docstring citing spec/research, imports from `draft.domain.draft_agent_model`, `draft.domain.draft_geometry`, `draft.domain.draft_state`, `draft.infrastructure.draft_agent_store`, `draft.infrastructure.draft_record_io`, `sealed.infrastructure.converted_card_locator`; a `RANDOM_SEED = 42` constant and `_log` helper mirroring `train_draft_agent.py`). No logic yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared changes the RL trainer depends on — the checkpoint RL-metadata extension and the extracted per-pick state-walk reused from gen-1.

**⚠️ CRITICAL**: US1 cannot be implemented until this phase is complete.

- [x] T002 [P] Extend `DraftAgentStore` with optional `rl_metadata` in `src/draft/infrastructure/draft_agent_store.py`: add an `rl_metadata: dict | None = None` param to `save_checkpoint` (written into the payload only when present) and an `rl_metadata` field on `LoadedDraftAgentCheckpoint` (defaulting to `None` via `payload.get`). Backward compatible — gen-1 checkpoints load unchanged. Prior art: the existing store (data-model §6).
- [x] T003 [P] Extend `tests/unit/draft/test_draft_agent_store.py`: round-trip a checkpoint with `rl_metadata` (generation, reference id, algorithm, gae_lambda, kl_coef, entropy_coef, value_weight, rollout_temperature) and assert a gen-1 checkpoint (no `rl_metadata`) still loads with `rl_metadata is None`. Must FAIL before T002.
- [x] T004 Extract the per-seat state-walk into `src/draft/application/draft_pick_states.py`: a reusable generator yielding each pick's raw state (card embedding-rows, `type_idx`/`packs_ago`/`pick_ago` arrays, the taken-action row index, `pack_number`/`pick_number`) decoupled from `DraftExample`, plus re-export `_leave_one_out_rewards`. Lift the logic verbatim from `train_draft_agent._emit_seat`/`_emit_example` (research §Overlapping vocabulary — extract, don't duplicate). Pure (no torch), uses `DraftGeometry`.
- [x] T005 Refactor `train_draft_agent._Loader` in `src/draft/application/train_draft_agent.py` to build `DraftExample` from the T004 helper (thin adapter); remove the now-duplicated walk body. No behavior change.
- [x] T006 [P] Extend `tests/unit/draft/test_loader_walk.py` with an equivalence assertion: gen-1 examples are byte-identical before/after the T004/T005 extraction (Data Integrity, Principle III). Must FAIL before T005 lands (or pass trivially against the pre-refactor baseline, then guard the refactor).

**Checkpoint**: Checkpoint RL-metadata and the shared state-walk are in place — US1 can begin.

---

## Phase 3: User Story 1 - Fine-tune a new generation from a self-play corpus (Priority: P1) 🎯 MVP

**Goal**: `python -m draft train-draft-agent-rl` performs the on-policy actor-critic update from a reference checkpoint + the corpus it generated, and writes the next-generation checkpoint with RL metadata.

**Independent Test**: Provide a reference checkpoint and a small corpus it generated; run the trainer; confirm a loadable `{timestamp}.pt` with `rl_metadata` (generation index, reference id, hyper-params), a printed loss decomposition + behaviour-anomaly summary, and a non-increasing held-out RL objective.

### Tests for User Story 1 (MANDATORY — write first, ensure they FAIL) ✅

- [x] T007 [P] [US1] Unit test GAE(λ)/return in `tests/unit/draft/test_rl_advantage.py`: over a per-seat value sequence with terminal reward `R` and γ=1, assert λ=1 ⇒ `A_t = R − V(s_t)`, λ=0 ⇒ one-step TD, and intermediate λ blends; trajectory bootstrap `δ_T = R − V(s_T)` (data-model §2; research D2).
- [x] T008 [P] [US1] Unit test loss decomposition in `tests/unit/draft/test_rl_loss.py`: policy/value/entropy/kl terms; PACK-masked softmax at rollout temperature `T`; only `learner_active` picks feed policy/entropy/kl while all `critic_active` feed the value MSE; `0·-inf` NaN-gradient guards on entropy/KL (research D4, D5).
- [x] T009 [P] [US1] Unit test RL loader in `tests/unit/draft/test_rl_loader.py`: per-`(draft,seat)` trajectory grouping in (pack,pick) order; `learner_active = seat∈learner_agents ∧ action embeddable`; `critic_active = non-failed`; failed builds (`deck=[]`/`deck_score=None`) excluded from reward/pod-mean/gradient; `--critic-corpus` contributes critic-only examples (data-model §2,§3; spec FR-013, FR-015).
- [x] T010 [P] [US1] Unit test behaviour-anomaly summary in `tests/unit/draft/test_rl_behaviour_anomaly.py`: mean behaviour log-prob + `frac_below_floor`; warning emitted when the fraction exceeds the threshold; training never aborts (spec FR-009, SC-004; research D6).
- [x] T011 [P] [US1] Unit test CLI validation in `tests/unit/draft/test_rl_cli.py`: `--checkpoint` xor `--resume`; empty `--learner-agents` → exit 2; `--rollout-temperature` missing (required) → exit 2 and `--rollout-temperature <= 0` → exit 2; architecture flag with load → exit 2; missing file → exit 2; resume precedence (CLI > resumed `train_config` > default) via `cli_resume` (contracts/train-draft-agent-rl.md; spec FR-017).
- [x] T012 [P] [US1] Integration smoke test in `tests/integration/test_train_draft_agent_rl_smoke.py` (mark `integration`): train one tiny on-policy corpus for 1 epoch from a tiny reference checkpoint; assert a checkpoint with `rl_metadata` is written and the held-out RL objective is non-increasing (quickstart.md §Minimal single-train smoke).

### Implementation for User Story 1

- [x] T013 [US1] Implement `TrainDraftAgentRLConfig` dataclass in `src/draft/application/train_draft_agent_rl.py` (data-model §1): required `checkpoint`/`learner_agents`/`rollout_temperature` (no default — a forgotten flag must fail fast, not silently train at T=1.0); `drafts_path`, repeatable `critic_corpus`, `gae_lambda=0.95`, `kl_coef`, `entropy_coef`, `entropy_decay_after`, `value_weight=1.0`, gen-1 LR/batch/epoch/val/patience/lr-decay knobs, `resume`. γ fixed at 1.0 (not a field). Mirrors `TrainDraftAgentConfig`.
- [x] T014 [US1] Implement the RL loader in `src/draft/application/train_draft_agent_rl.py`: read on-policy + `--critic-corpus` via `read_records`, build `Trajectory`/`RLExample` from the T004 shared walk with a shared embedding table (load each `.npz` once — Principle VIII I/O batching), tag `learner_active`/`critic_active`/`reward`, group + order picks per `(draft,seat)`. Prior art: `train_draft_agent._Loader` (research §Adjacent prior art).
- [x] T015 [P] [US1] Implement the GAE/advantage pure helper in `src/draft/application/rl_advantage.py`: `advantages(values, reward, gae_lambda, gamma=1.0)` over an ordered trajectory, terminal bootstrap, returns detached `A_t` (+ return `G_t=R`). Pure numpy/torch, no model. (Target of T007.)
- [x] T016 [US1] Add masked-softmax entropy + KL helpers to `src/draft/application/train_draft_agent_rl.py`, adapted (PACK-masked) from `train_picker._policy_entropy`/`_kl_penalty`/`_masked_log_softmax`, keeping the `0·-inf` NaN-gradient guards (research §Adjacent prior art — copy the small REINFORCE helpers).
- [x] T017 [US1] Implement `_collate` (+ `_Batch`) and `_compute_loss` in `src/draft/application/train_draft_agent_rl.py`: collate per gen-1 plus per-pick `advantage` and `pack_mask`; compute `−A·logπ_T(action)` over learner picks, `value_weight·MSE(V, standardize(R))` over critic picks, `−entropy_coef·H(π_T)` and `kl_coef·KL(π_T‖π_ref,T)` over learner picks; all distributions at rollout temperature `T` (research D4, D5; data-model §4).
- [x] T018 [US1] Implement the behaviour-anomaly summary in `src/draft/application/train_draft_agent_rl.py`: batched `no_grad` `log π_ref,T(a_t)` over learner picks → `mean_behaviour_logprob` + `frac_below_floor`; log once; warn (never abort) when above threshold (data-model §5; spec FR-009).
- [x] T019 [US1] Implement warm-start/resume build in `src/draft/application/train_draft_agent_rl.py`: load `--checkpoint` (actor+critic) bootstrap and a **frozen deep-copy KL reference** `π_ref`; reuse the loaded `critic_mean`/`critic_std` (do not recompute); `--resume` path (weights+optimizer+epoch+best+`lr_decay_count`); reuse gen-1 `_check_dims` for the `.npz` width check; xor handling. Prior art: `train_picker._resume_or_build_picker` (reference_state) + `train_draft_agent._resume_or_build` (research D7).
- [x] T020 [US1] Implement the per-epoch advantage precompute in `src/draft/application/train_draft_agent_rl.py`: a batched `no_grad` critic forward over every learner trajectory on the GPU, then T015 GAE → cache detached `A_t` onto each `RLExample`; no per-pick host↔device sync (research D3; Principle VIII GPU batching).
- [x] T021 [US1] Implement the training loop in `src/draft/application/train_draft_agent_rl.py`: AdamW + warmup-then-constant `LambdaLR` + per-group `clip_grad_norm_` + LR-plateau annealing (gen-1 `_make_scheduler`/`_PlateauLR`); val-driven `kl_coef`/`entropy_coef` schedules (`_EntropySchedule` pattern); mini-epoch eval → best/`latest` via `DraftAgentStore.save_checkpoint(..., rl_metadata=…)` with `generation = reference.generation+1`; early stop on patience (research D8, D9; spec FR-018, FR-019, FR-020).
- [x] T022 [US1] Implement `_validate` in `src/draft/application/train_draft_agent_rl.py`: compute the held-out **RL objective** (policy+value+entropy+kl using the epoch's cached val advantages) as the best-checkpoint/early-stop metric, plus per-pack critic MSE; one log line per mini-epoch (research D9; mirrors `train_draft_agent._validate`).
- [x] T023 [P] [US1] Wire the CLI in `src/draft/infrastructure/cli.py`: add the `train-draft-agent-rl` subparser (flags per contracts/train-draft-agent-rl.md; resumable flags `default=None`; architecture flags absent), `run_train_draft_agent_rl` dispatch with startup validation + exit codes, and resume precedence via `cli_resume.resolve_resumable_args`. Mirrors `_build_train_draft_agent_parser`/`run_train_draft_agent`.

**Checkpoint**: US1 is fully functional and independently testable — the MVP. An operator can fine-tune a candidate generation and inspect it.

---

## Phase 4: User Story 2 - Compare generations and decide promotion (Priority: P2)

**Goal**: An apples-to-apples cross-generation yardstick: one greedy fixed-mix `generate-draft-data` run + per-agent `analyze-generated-decks`, with promotion as a manual judgment. **No new code** (spec US2; research §Adjacent prior art).

**Independent Test**: On any two checkpoints, run the greedy fixed-mix generation + per-agent analysis and confirm each agent's mean `deck_score` prints on one shared scale with a clear beats/doesn't verdict the operator can apply.

- [x] T024 [US2] Document the cross-generation yardstick procedure in `README.md` (workflow) and `CLAUDE.md` (draft section): single shared `generate-draft-data --pick-mode argmax` over a fixed mix co-seating the generations + Forge + a random bot, then `analyze-generated-decks --agent <each>` for per-agent mean `deck_score`; promotion is a manual operator judgment (the system does not auto-promote). Source: quickstart.md §Step 4–5 (Principle VI).
- [ ] T025 [US2] Validate the yardstick runbook (no new code): on a small corpus, run `generate-draft-data --pick-mode argmax` with a two-`--agent-checkpoint` mix and `analyze-generated-decks --agent <each>`; confirm per-agent mean `deck_score` prints on the shared scale. Record the result against quickstart.md §Step 4.

**Checkpoint**: An operator can rank two generations on the shared scale and decide promotion.

---

## Phase 5: User Story 3 - Run an operator-driven self-play cycle (Priority: P3)

**Goal**: One end-to-end cycle (freeze → generate sample → train-rl → evaluate greedy → manual promote), runnable with existing commands + the US1 trainer, no orchestration code. **No new code** (spec US3; quickstart.md).

**Independent Test**: Follow the runbook to complete one cycle with no manual file editing, ending with a candidate checkpoint whose `rl_metadata.generation` is incremented.

- [x] T026 [US3] Document the self-play cycle runbook in `README.md` + `CLAUDE.md`: freeze `πₖ`; `generate-draft-data --pick-mode sample --temperature T --seed S` with the learner label + Forge + retained random-bot minority into `drafts-genK.jsonl` (operator-convention provenance); `train-draft-agent-rl --rollout-temperature T` (matching) with optional `--critic-corpus` of prior cycles; greedy yardstick; manual promote. Source: quickstart.md §Step 1–5 (Principle VI).
- [ ] T027 [US3] Validate one full cycle end-to-end at small scale per quickstart.md (sample-mode corpus → `train-draft-agent-rl` → greedy yardstick → analyze); confirm no manual corpus/checkpoint editing is required and the candidate's `rl_metadata.generation` increments over the reference. Additionally **record the SC-007 composition check**: capture the candidate-vs-reference `analyze-generated-decks` composition stats (colour/curve/creature counts) and note that the candidate's distribution has not collapsed to a degenerate one (descriptive, not a gate).

**Checkpoint**: All three stories are independently functional; the lineage can advance a generation per operator cycle.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T028 [P] Update `CLAUDE.md` draft-package section + `README.md` artifact layout: the `train-draft-agent-rl` subcommand (inputs/outputs), the checkpoint `rl_metadata` artifact, and the RL process rationale (REINFORCE+GAE, KL anchor, frozen encoder / Phase A) — Principle VI; spec tense per project convention.
- [ ] T029 [P] Follow-up from research §Third-instance check (non-blocking): extract the two byte-identical trainer atoms (warmup-LR `LambdaLR` lambda and the per-group `clip_grad_norm_` helper), now shared by 5 trainers, into a shared util and rewire all five — or document the deferral with the updated count. (Whole-loop extraction stays deferred.)
- [x] T030 Performance review per Principle VIII over `train_draft_agent_rl.py`: confirm shared embedding table (each `.npz` loaded once), GPU placement of actor/critic/`π_ref`, batched `no_grad` advantage + behaviour-logprob passes, no per-pick `.item()`/`.cpu()` in the loop, streamed corpus read, load-once model/locator. Any optimization beyond the checklist needs a profile.
- [x] T031 [P] Run `ruff check src/ tests/` and fix all findings (no new warnings — Quality Gates).
- [ ] T032 Run quickstart.md validation end to end (minimal single-train smoke + the US3 cycle) and confirm the documented outputs; update quickstart.md if any command/flag drifted.

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: after Setup. **Blocks US1.** (US2/US3 are doc/validation and depend on US1 existing for their *validation* tasks but not their doc tasks.)
- **US1 (Phase 3)**: after Foundational. The MVP and the only new-code story.
- **US2 (Phase 4)** / **US3 (Phase 5)**: doc tasks (T024, T026) can be written any time after US1's CLI exists; validation tasks (T025, T027) require US1 complete.
- **Polish (Phase 6)**: after US1 (T028/T030/T031), with T029 independent and T032 after US2/US3 validation.

### Task-level dependencies

- T002 → T003 (test guards the change); T004 → T005 → T006.
- T013 before T014/T017/T019/T023 (config used). T014 needs T004. T015 before T020. T016 before T017. T019 before T020 (model+ref loaded). T017+T020+T022+T002 before T021. T023 needs T013 + the use-case class (T021).
- Tests T007–T012 are written before their implementation targets and must FAIL first.
- T024/T026 depend on T023 (the command exists to document); T025/T027 depend on the full US1.

### Parallel opportunities

- T002/T003 (store) ∥ T004 (extraction start) — different files.
- All US1 tests T007–T012 in parallel (separate files).
- T015 (`rl_advantage.py`) ∥ the `train_draft_agent_rl.py` body tasks (different file); T023 (`cli.py`) ∥ later trainer-internal tasks once T013 lands.
- Polish T028/T029/T031 in parallel.

> **Same-file caution**: T013, T014, T016, T017, T018, T019, T020, T021, T022 all edit `src/draft/application/train_draft_agent_rl.py` — run them **sequentially** (not [P]) despite belonging to one story.

---

## Parallel Example: User Story 1 tests

```bash
# Write all US1 tests together (they must FAIL before implementation):
Task: "Unit test GAE/advantage in tests/unit/draft/test_rl_advantage.py"
Task: "Unit test loss decomposition in tests/unit/draft/test_rl_loss.py"
Task: "Unit test RL loader in tests/unit/draft/test_rl_loader.py"
Task: "Unit test behaviour-anomaly summary in tests/unit/draft/test_rl_behaviour_anomaly.py"
Task: "Unit test CLI validation in tests/unit/draft/test_rl_cli.py"
Task: "Integration smoke test in tests/integration/test_train_draft_agent_rl_smoke.py"
```

---

## Implementation Strategy

### MVP first (User Story 1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational (store + state-walk extraction) → 3. Phase 3 US1 (the trainer) → **STOP and validate**: fine-tune a candidate on a tiny on-policy corpus, confirm the checkpoint + RL metadata + non-increasing held-out RL objective. This is a usable deliverable on its own (an operator can train gen-2).

### Incremental delivery

- US1 (trainer) → then US2 (yardstick docs + validation, no code) → then US3 (full-cycle runbook docs + validation, no code) → Polish. Each adds operator capability without changing earlier code.

### Notes

- [P] = different files, no incomplete-task dependency. [Story] maps tasks to spec user stories.
- Verify each US1 test FAILS before its implementation; commit after each task or logical group, referencing the task ID.
- US2/US3 add no production code — keep their tasks to documentation + runbook validation (spec Assumptions).
- **Principle VII**: new modules cite prior art — `train_draft_agent_rl.py` mirrors `train_draft_agent.py`/`train_picker.py`; `draft_pick_states.py` is an extraction from `train_draft_agent`; `rl_advantage.py` is genuinely new (GAE not present elsewhere).
- **Principle VIII**: T014/T018/T020/T021/T030 touch data loading / model compute — checked against the performance checklist.
