---

description: "Task list for 021-draft-online-grpo — Draft Agent Online Self-Play GRPO Trainer (Generation 3)"
---

# Tasks: Draft Agent — Online Self-Play GRPO Trainer (Generation 3)

**Input**: Design documents from `/specs/021-draft-online-grpo/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: Per the project constitution (Principle I), test tasks are mandatory.
Every pure/near-pure helper gets a fast unit test; the loop gets one integration
smoke test driven by a fake worker (no JVM).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: `[US1]` / `[US2]` / `[US3]` — maps to the spec's user stories

## Path Conventions

Single project, hexagonal: `src/<package>/{domain,application,infrastructure}/`,
tests at `tests/unit/` + `tests/integration/`, Java under `forge-connector/src/`.
All commands run from the repository root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish a green baseline before the Phase 2 shared-code extraction
touches four existing trainers. No new dependencies are needed — `torch` and
`numpy` are already declared in `pyproject.toml`.

- [X] T001 Establish the pre-refactor baseline: run `pytest tests/unit/draft tests/unit/sealed/application tests/integration/test_generate_draft_data.py tests/integration/test_draft_live_play.py tests/integration/test_draft_supervisor_restart.py -q` and `ruff check src/ tests/`; record any failure before Phase 2 changes shared modules
- [X] T002 [P] Confirm the Java toolchain builds the connector before the D2 worker change: `cd forge-connector && mvn package -DskipTests` produces `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The Principle VII third-instance extractions (plan.md Constitution
Check follow-up (a)) plus the run-configuration entity. Both user stories build
on these; the extraction must land *before* new call sites are written so gen-3
never becomes the fifth copy.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Create `src/price_predictor/infrastructure/torch_training.py` exporting `masked_log_softmax`, `policy_entropy`, `kl_divergence`, `clip_per_group` — bodies lifted verbatim from `src/sealed/application/train_picker.py` (`_masked_log_softmax` L360, `_policy_entropy` L364, `_kl_penalty` L379, `_clip_per_group` L712), preserving the `0 * -inf` NaN-gradient guards and `clip_per_group`'s pre-clip norm return value; module docstring states it is the shared home beside `torch_checkpoint.py`, importable by `sealed` and `draft` (prior art: `price_predictor/infrastructure/torch_checkpoint.py`)
- [X] T004 [P] Repoint `src/sealed/application/train_picker.py` at `price_predictor.infrastructure.torch_training` and delete the local `_masked_log_softmax`/`_policy_entropy`/`_kl_penalty`/`_clip_per_group`; update call sites at L987 (`_policy_entropy`), L999 (`_kl_penalty` → `kl_divergence`), L1005 (`_clip_per_group`)
- [X] T005 [P] Repoint `src/sealed/application/train_scorer.py` at `price_predictor.infrastructure.torch_training.clip_per_group` and delete the local `_clip_per_group` (L817); update the call site at L684
- [X] T006 [P] Repoint `src/draft/application/train_draft_agent.py` at `price_predictor.infrastructure.torch_training.masked_log_softmax` and delete the local `_masked_log_softmax` (L414); update the call site at L437
- [X] T007 [P] Repoint `src/draft/application/train_draft_agent_rl.py` at `price_predictor.infrastructure.torch_training` and delete the local `_masked_log_softmax`/`_policy_entropy`/`_kl_divergence` (L292–L318); update call sites at L458, L463, L466, L627
- [X] T008 Create `src/draft/application/draft_training_common.py` exporting `leave_one_out_rewards(record)` and `length_bucketed_batches(examples, batch_size, rng)` — bodies lifted verbatim from `src/draft/application/train_draft_agent.py` (L124, L289); `length_bucketed_batches` stays generic over any example exposing `n_tokens`; module docstring names it the sibling of the already-extracted `draft_pick_states.py`
- [X] T009 Repoint `src/draft/application/train_draft_agent.py` (L124, L289 definitions; L194, L859 call sites) and `src/draft/application/train_draft_agent_rl.py` (L144, L275 definitions; L228, L925 call sites) at `draft.application.draft_training_common`, deleting both local copies
- [X] T010 [P] Repoint the four test files that import the extracted private names: `tests/unit/draft/test_draft_loss.py` (`_leave_one_out_rewards` → `draft_training_common.leave_one_out_rewards`), `tests/unit/draft/test_length_bucketing.py` (`length_bucketed_batches`), `tests/unit/sealed/application/test_train_picker.py` (`_kl_penalty` → `kl_divergence`, `_policy_entropy` → `policy_entropy`), `tests/unit/sealed/application/test_train_scorer.py` (`_clip_per_group` → `clip_per_group`)
- [X] T011 Verify the extraction is behaviour-neutral: re-run the T001 suites and `ruff check src/ tests/`; the only test edits allowed are the T010 import changes
- [X] T012 Create `src/draft/application/train_draft_agent_online.py` with the `TrainDraftAgentOnlineConfig` dataclass — every field, type, and default from data-model.md § 1 (mirroring `TrainDraftAgentRLConfig` at `train_draft_agent_rl.py:83`), with a docstring listing the deliberately absent gen-2 knobs (`value_weight`, `gae_lambda`, `kl_coef`, `entropy_coef`, `val_fraction`, `patience`, `epochs`, `lr_decay_*`, `resume`, `pick_mode`) per FR-006

**Checkpoint**: Shared helpers have one home each, the existing suites are green, and the gen-3 config entity exists — user story work can begin

---

## Phase 3: User Story 1 - Apply one online GRPO update from a fresh self-play batch (Priority: P1) 🎯 MVP

**Goal**: Given a checkpoint and a small batch of drafts it generated, apply one
pass of the single-term critic-free GRPO update `−A·logπ_T(a|s)` over the
learner-seat picks only, write the next checkpoint, and log the reward /
exploration / movement diagnostics.

**Independent Test**: Drive the round-update entry point with a warm-start
checkpoint and a synthetic batch of `DraftRecord`s (no Forge, no generation):
confirm a loadable checkpoint in gen-1 format carrying
`rl_metadata["algorithm"] == "online-grpo"`, that only learner-label picks moved
the weights, and that the three diagnostic axes are printed. Startup validation
rejects every invalid configuration in data-model § 1.1 before any update.

### Tests for User Story 1 (MANDATORY per Constitution) ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T013 [P] [US1] Advantage tests in `tests/unit/draft/test_online_advantage.py`: round standardisation gives mean≈0/std≈1 over surviving learner seats, one scalar shared by all of a seat's picks (γ=1), failed-build seats and seats whose pod has no other non-failed seat are excluded and counted as dropped, and the degenerate guards (<2 surviving rewards, or `std < 1e-8`) return a no-op round rather than dividing by zero (data-model § 4, FR-008/FR-009/FR-022/FR-023)
- [X] T014 [P] [US1] Loss tests in `tests/unit/draft/test_online_loss.py`: the loss equals `−mean(A · logπ_T(a|s))` over the batch, log-probs are masked to `PACK` positions only, a temperature `T ≠ 1` changes the value as `logits / T` predicts, no critic/KL/entropy term contributes, and the critic head receives no gradient (data-model § 5, FR-010)
- [X] T015 [P] [US1] Round-loader tests in `tests/unit/draft/test_online_loader.py`: only learner-label seats yield `OnlineExample`s, a learner seat with `deck_score is None` yields none, a pick whose taken card is un-embeddable (`action_position == -1`) is dropped, the per-round card table contains exactly the round's cards, and `dropped_seats` matches the exclusions (data-model § 2/§ 3, FR-007/FR-022)
- [X] T016 [P] [US1] Diagnostics tests in `tests/unit/draft/test_online_diagnostics.py`: from the `π_k` no-grad pass, `perplexity == exp(entropy)`, the off-argmax rate counts picks whose recorded action ≠ `argmax π_k` over the same `PACK` set, `mean_logp` averages the taken actions' log-probs, `KL(π_k‖π_{k+1})` is 0 when the weights are unchanged and > 0 after a step (data-model § 6, research D9, FR-015/FR-016)
- [X] T017 [P] [US1] CLI/startup-validation tests in `tests/unit/draft/test_online_cli.py`: each data-model § 1.1 rule (exactly one `--learner NAME=PATH`; learner label in `--mix` and not also `--frozen`; every mix label known; anchor resolvable/unambiguous; `-T` supplied and > 0; scorer exists, picker exists when `--build-method picker`; positive `--drafts-per-round`/`--anchor-window`/`--batch-size`/`--snapshot-every`/`--max-rounds`) exits `2` with a clear message, an architecture/width mismatch exits `6`, and no update or worker launch happens first (FR-024, SC-006, contract § Exit codes)

### Implementation for User Story 1

- [X] T018 [US1] Add the `OnlineExample` dataclass, `_Batch`, and `_collate` to `src/draft/application/train_draft_agent_online.py` per data-model § 3 — a trimmed sibling of `train_draft_agent_rl.py`'s `RLExample` (L122), `_Batch` (L358), `_collate` (L374) with no critic/GAE/`learner_active` fields; collate pads per batch and places tensors on the training device (Principle VIII: no per-item host↔device transfer)
- [X] T019 [US1] Implement the round loader in `src/draft/application/train_draft_agent_online.py`: walk each learner seat of each `DraftRecord` with `draft.application.draft_pick_states.iter_seat_pick_states` (research D6), memoize card rows through a single shared `sealed.infrastructure.converted_card_locator.ConvertedCardLocator`, and build the per-round table over only that round's cards (research D14); mirror `train_draft_agent_rl._Loader` (L157) minus the multi-corpus split
- [X] T020 [US1] Implement reward → advantage in `src/draft/application/train_draft_agent_online.py`: `draft_training_common.leave_one_out_rewards` per record, then round standardisation over the surviving learner seats, returning the degenerate-round no-op signal (fewer than 2 rewards or `std < 1e-8`) per data-model § 4
- [X] T021 [US1] Implement `_compute_loss` in `src/draft/application/train_draft_agent_online.py` using `price_predictor.infrastructure.torch_training.masked_log_softmax` on `logits / T` gathered at each example's `action_token`, weighted by the detached advantage (data-model § 5)
- [X] T022 [US1] Implement the one-pass update in `src/draft/application/train_draft_agent_online.py`: shuffle + `draft_training_common.length_bucketed_batches(examples, batch_size, rng)`, then per minibatch forward → backward → `torch_training.clip_per_group(optimizer, max_norm=…)` (capturing the pre-clip norm) → `optimizer.step()` → `scheduler.step()`; accumulate the round's mean policy loss and mean pre-clip grad norm; discard the batch afterwards (FR-011, research D8)
- [X] T023 [US1] Implement the diagnostics pass in `src/draft/application/train_draft_agent_online.py`: hold a resident `prev_model` loaded with the pre-update `state_dict`, and after the update run **one** batched `no_grad` sweep over the round's examples forwarding both models to accumulate entropy/perplexity/off-argmax/`mean logπ` (from `π_k`) and `KL(π_k‖π_{k+1})` (both), populating `RoundDiagnostics` per data-model § 6 (research D9; Principle VIII: one batched sweep, scalars read once per round). **Both models are in `eval()` for the sweep**: `prev_model` is constructed once and never put in train mode, and the learner is switched back to `eval()` after the update's last backward pass and before the sweep — with dropout active the entropy, off-argmax rate and KL measure noise rather than the policy
- [X] T024 [US1] Implement run setup in `src/draft/application/train_draft_agent_online.py`: load the `--learner` checkpoint via `DraftAgentStore`, validate width/geometry with the `train_draft_agent_rl._check_dims` (L653) convention, select the device with `train_draft_agent._select_device`, build AdamW, and construct the LR scheduler **once** for the whole run via `train_draft_agent._make_scheduler(optimizer, total_steps=warmup_steps, warmup_frac=1.0, controller=None)` so the ramp is exactly `--warmup-steps` optimizer steps then constant, with no plateau controller (research D15, FR-025)
- [X] T025 [US1] Implement checkpoint writing in `src/draft/application/train_draft_agent_online.py` via the unchanged `draft.infrastructure.draft_agent_store.DraftAgentStore`: gen-1 payload with the critic head carried through, `epoch` = round index, `best_val_loss = inf`, `critic_mean`/`critic_std` copied verbatim from the base checkpoint, `train_config` = the resolved § 1 config with paths stringified, and `rl_metadata = {generation: base+1, base_checkpoint, algorithm: "online-grpo", lr, rollout_temperature, drafts_per_round}` (data-model § 8, FR-027/FR-028)
- [X] T026 [US1] Implement `_validate_config` in `src/draft/application/train_draft_agent_online.py` covering every data-model § 1.1 rule in order, raising the typed errors the CLI maps to exit codes; all checks run before the worker launches and before any update (FR-024)
- [X] T027 [US1] Implement the startup echo in `src/draft/application/train_draft_agent_online.py` matching contract § 1 line-for-line (learner + generation transition, frozen/anchor, mix with the ">=1 learner seat forced" note, reward config, rollout config, optimiser config, device + embedding width + anchor window, corpus + checkpoint destinations), noting that Forge-side randomness is unseeded (research D12, FR-013)
- [X] T028 [US1] Implement the round-diagnostics formatter in `src/draft/application/train_draft_agent_online.py`: the four detail lines (`reward` / `explore` / `movement` / `progress`) and the consolidated round-summary line exactly as contract § 2 shows, with the `progress` margin rendered as unavailable until the anchor window is populated (US2 fills it), and the degenerate round replaced by the single `skipped (no signal)` line (FR-014–FR-018, FR-023, SC-003)
- [X] T029 [US1] Wire the `train-draft-agent-online` subparser and `run_train_draft_agent_online` into `src/draft/infrastructure/cli.py` following the `_build_train_draft_agent_rl_parser` (L309) / `run_generate_draft_data` (L533) convention: every flag from contracts/train-draft-agent-online.md with its default, lazy application import inside `run_*`, and the **startup-failure** exit codes the T017 tests exercise (`2` for validation / missing file / bad flag value, `6` for `DraftAgentArchitectureError`); the loop-lifetime codes (`1`, `130`, `0`) land in T046. Register no `--resume` and no gen-2 coefficient flags (research D13, FR-006)

**Checkpoint**: One online GRPO update runs end-to-end from a supplied batch, writes a well-formed gen-3 checkpoint, prints three diagnostic axes, and rejects every invalid startup configuration — User Story 1 is independently testable

---

## Phase 4: User Story 2 - Run the streaming self-play loop with a live progress read (Priority: P1)

**Goal**: One command drives the whole loop — generate `--drafts-per-round`
fresh drafts from the current policy against one resident Forge worker, apply the
US1 update, discard, regenerate — while logging the live anchor margin each round.

**Independent Test**: Run the loop for two rounds against a fake worker: confirm
the worker is launched once and stays resident across rounds, that round 2's
drafts were generated after round 1's update (nothing re-shown), that the
launcher is invoked with `required_agent=<learner label>` (the Java rule it arms
is covered separately by T034), that the corpus accumulates on disk in the
unchanged schema, and that the per-round `progress` line reports the anchor
margin plus each label's windowed mean.

### Tests for User Story 2 (MANDATORY per Constitution) ✅

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T030 [P] [US2] Record-stream tests in `tests/unit/draft/test_record_stream.py`: `iter_records` launches a worker on the first `next()`, yields fully assembled + labeled `DraftRecord`s, stays suspended (worker alive, nothing consumed) between pulls, relaunches after a worker exit and keeps yielding, routes pick requests / abandonments exactly as today, and propagates `MaxConsecutiveFaultsError`; `close()` terminates the worker via the generator's `finally`. Plus two launch/labeler cases: `_default_launch_worker` forwards `required_agent=` to `DraftWorkerConnector.start` when `GenerateDraftDataConfig.required_agent` is set and passes nothing when it is `None`, and `build_labeler(config, locator=…)` uses the supplied locator instead of constructing one (contracts/rollout-stream.md § 1, research D1/D2/D14)
- [X] T031 [P] [US2] Extend `tests/unit/draft/test_agent_pick_service.py`: `AgentPickService.from_model` wraps an already-constructed `DraftAgentModel` by reference (a weight change is visible to the very next pick with no push), does **not** call `.eval()`/`.train()`/`.to()`, and otherwise behaves identically to the path-loading constructor (contracts/rollout-stream.md § 2, research D3, FR-012)
- [X] T032 [P] [US2] Extend `tests/unit/draft/test_agent_registry.py`: `AgentRegistry.build(..., preloaded={label: service})` merges preloaded services, counts their labels as **bound** during FR-003 label validation, runs the same geometry checks (`config.packs == PACKS`, `config.P ≥ pack_size`) over them, and rejects a label bound both as preloaded and as a checkpoint path (research D4)
- [X] T033 [US2] Extend `tests/unit/draft/test_online_diagnostics.py` (the file T016 creates — the one US2 test that shares a US1 file) with `AnchorWindow` cases: bounded `deque(maxlen=anchor_window)` of per-draft label→scores, `label_mean` ignores `None` scores, `margin` is `None` until both the learner and anchor labels have a scored seat in the window, eviction drops the oldest draft, and the best margin + its round index are tracked for the final summary (data-model § 7, FR-017/FR-019/FR-021)
- [X] T034 [P] [US2] Extend `forge-connector/src/test/java/com/pricepredictor/connector/DraftWorkerMainTest.java`: with `-Ddraft.required.agent` set and a mix draw containing none of that label, exactly one uniformly-chosen seat is overwritten with it and the rest of the draw is untouched; a draw that already contains the label is unchanged; absent/blank property leaves sampling byte-for-byte as today (contracts/rollout-stream.md § 3, research D2)
- [X] T035 [P] [US2] Integration smoke test in `tests/integration/test_train_draft_agent_online_smoke.py`: two full rounds against a fake worker (no JVM, no GPU) following the `tests/integration/test_draft_supervisor_restart.py` fake-worker pattern — assert both rounds log all four axes, `models/draft/agent/latest.pt` loads with `rl_metadata["algorithm"] == "online-grpo"`, round 2's records were generated after round 1's update (SC-008), and the corpus file grew by `2 × --drafts-per-round` records

### Implementation for User Story 2

- [X] T036 [US2] Three additive changes to `src/draft/application/generate_draft_data.py`, all behaviour-preserving for existing callers:
  - **(a)** Extract `iter_records(launch, labeler) -> Iterator[DraftRecord]` from `GenerateDraftDataSupervisor._supervise` (L608) as an endless generator that launches/relaunches the worker, routes pick lines, assembles records, and yields them; rewrite `run()` (L550) as its consumer keeping append + count + progress/rate + `--n-drafts` target + resume behaviour byte-for-byte (contracts/rollout-stream.md § 1).
  - **(b)** Give `build_labeler` (L338) an optional locator — `build_labeler(config, *, locator=None)` — constructing its own `ConvertedCardLocator` only when absent, so one memoizing locator can serve the labeler, the pick services, and the trainer (research D14; without this the plan's Performance Review claim is unachievable and every card's `.npz` is decompressed twice).
  - **(c)** Add `required_agent: str | None = None` to `GenerateDraftDataConfig` (L84) and forward it from `_default_launch_worker` (L535) into `DraftWorkerConnector.start(required_agent=…)`, omitting it when `None` — this is the launch-site link that actually arms T039's Java rule; the Java change alone is inert (research D2, FR-003).
  - `tests/integration/test_generate_draft_data.py`, `tests/integration/test_draft_live_play.py`, `tests/integration/test_draft_supervisor_restart.py`, and `tests/unit/draft/test_supervisor_pick_routing.py` stay green unchanged.
- [X] T037 [US2] Add the `AgentPickService.from_model(model, config, locator, *, device, pick_mode, temperature, seed)` alternate constructor in `src/draft/application/agent_pick_service.py`, documenting that the caller owns mode/device placement. The path-loading constructor's public behaviour is unchanged, but factor the shared tail of `__init__` (L57–L77: mode validation, device resolution, RNG, tracker map) into a private initializer both constructors call rather than duplicating it (Principle II) (research D3)
- [X] T038 [US2] Add the optional `preloaded: dict[str, AgentPickService] | None` keyword to `AgentRegistry.build` in `src/draft/application/agent_registry.py` (L67): merge into the service map, treat those labels as bound for FR-003 validation, and run the existing geometry checks over them (research D4)
- [X] T039 [P] [US2] Honour `-Ddraft.required.agent` in `forge-connector/src/main/java/com/pricepredictor/connector/DraftWorkerMain.java`: read the property beside `draft.external.agents` (L94), and after the per-seat `mix.sample(random)` loop (L191–L194) overwrite one uniformly-chosen seat with the required label when no seat carries it; absent/blank ⇒ unchanged behaviour; document the property in the class javadoc property list (L55–L59)
- [X] T040 [P] [US2] Forward the property from `DraftWorkerConnector.start` in `src/draft/infrastructure/draft_worker_connector.py` (L29, `system_properties` built at L57–L61): new optional `required_agent: str | None = None` keyword adding `draft.required.agent` to `system_properties`; omit the entry when None so existing callers are unchanged. T036(c) supplies the caller and T042 supplies the value — all three links are needed for FR-003
- [X] T041 [US2] Implement `AnchorWindow` in `src/draft/application/train_draft_agent_online.py` per data-model § 7 — a `deque(maxlen=anchor_window)` fed one entry per arriving record, `label_mean`, `margin`, `window_drafts`, and best-margin tracking; the anchor label is bound once at startup and never re-bound (FR-021)
- [X] T042 [US2] Implement `TrainDraftAgentOnlineUseCase.execute` in `src/draft/application/train_draft_agent_online.py`:
  - Build **one** `ConvertedCardLocator` and pass it to both `generate_draft_data.build_labeler(config, locator=…)` and `AgentRegistry.build(…, locator=…)` (`--frozen` paths as `agent_checkpoints`, `{learner_label: AgentPickService.from_model(...)}` as `preloaded`, `pick_mode="sample"` at `-T` for every model seat per research D5).
  - Construct the `GenerateDraftDataConfig` the supervisor needs from the resolved online config, setting every field **explicitly** rather than relying on its defaults (`GenerateDraftDataConfig.build_method` defaults to `"picker"`; gen-3's default is `"greedy"`): `build_method`, `picker_checkpoint`, `scorer_checkpoint`, `cards_path`, `agent_mix`, `agent_checkpoints`, `set_code`, `seed`, `max_consecutive_faults`, `pick_mode="sample"`, `temperature=T`, and **`required_agent=learner_label`** (T036(c) → T040 → the JVM property); then instantiate `GenerateDraftDataSupervisor` with the shared labeler + registry.
  - Open the corpus once in **`"a"` mode**, line-buffered — never `"w"`: `output/draft/drafts.jsonl` is the canonical shared corpus and truncating it destroys the gen-1/live-play data (FR-020).
  - Launch `iter_records` once, then per round pull `--drafts-per-round` records with `model.eval()`, append each record as it arrives, feed the anchor window, switch to `model.train()` for the US1 update, and continue until `--max-rounds` or interrupt (FR-005, FR-012, FR-020, FR-029, FR-031, research D1/D2/D14).
- [X] T043 [US2] Implement the checkpoint cadence in `src/draft/application/train_draft_agent_online.py`: write `models/draft/agent/latest.pt` every round and `models/draft/agent/{timestamp}.pt` every `--snapshot-every` rounds plus once at run end/interrupt, logging each write as contract § 3 shows; no best-checkpoint or early-stop guard (FR-026, research D11)
- [X] T044 [US2] Implement clean shutdown + the final summary in `src/draft/application/train_draft_agent_online.py`: reuse the supervisor's signal-handling convention (`_install_signal_handlers`, `generate_draft_data.py:774`), finish the in-flight round's bookkeeping, `close()` the record generator to terminate the JVM, write the final snapshot, and print the contract § 4 block (rounds, total drafts, learner picks, wall-clock, latest + final snapshot paths, best anchor margin and its round) (FR-019)
- [X] T045 [US2] Complete the per-round `progress` line and the summary line's margin field in `src/draft/application/train_draft_agent_online.py` from the anchor window: margin, every label's raw windowed mean, and the window draft count — computed in-loop with no pause and no second command (FR-017, SC-004/SC-005)
- [X] T046 [US2] Add the **loop-lifetime** exit codes on top of T029's startup codes in `run_train_draft_agent_online` (`src/draft/infrastructure/cli.py`): `MaxConsecutiveFaultsError` → 1, `KeyboardInterrupt` → 130 before the loop starts and 0 after a clean shutdown with the final summary printed, and 0 on reaching `--max-rounds`. T029 already maps `AgentRegistryError`/`AgentMixError`/`FileNotFoundError`/validation → 2 and `DraftAgentArchitectureError` → 6 (contract § Exit codes, FR-031)
- [X] T047 [US2] Rebuild the connector JAR (`cd forge-connector && mvn package -DskipTests`) and run `cd forge-connector && mvn test` so the T034 case exercises the shipped worker

**Checkpoint**: The full loop runs in one process against one resident worker, streams the corpus to disk, and prints a live anchor margin every round — User Stories 1 AND 2 both work

---

## Phase 5: User Story 3 - Decide when to pause and whether to promote (Priority: P2)

**Goal**: Make the cross-generation yardstick and the promotion decision runnable
entirely from existing commands — this story adds **no new code** (FR-030).

**Independent Test**: Take a gen-3 candidate and its base, run the greedy
fixed-mix `generate-draft-data` pass followed by one
`analyze-generated-decks --agent <label>` per agent, and confirm each agent's mean
`deck_score` is reported on one shared absolute scale.

- [X] T048 [US3] Verify the yardstick needs no new flags: check that `generate-draft-data` in `src/draft/infrastructure/cli.py` (L167) already accepts repeated `--agent-checkpoint LABEL=PATH`, `--agent-mix`, `--pick-mode argmax`, `--build-method greedy`, and `--output-path`, and that `analyze-generated-decks` accepts `--drafts-path` + `--agent`; record any gap as a defect rather than adding an evaluation engine (FR-030, spec Out of Scope)
- [X] T049 [US3] Document the gen-3 yardstick + promotion runbook in `README.md` beside the existing gen-2 RL section (L828): the greedy fixed-mix generation command, the per-agent analysis loop, and that promotion is a manual judgment with the anchor kept fixed across the campaign — cross-referencing [quickstart.md](quickstart.md) steps 3–5

**Checkpoint**: All user stories are independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T050 [P] Add the `train-draft-agent-online` subcommand to `README.md`: flags, the four diagnostic axes, the required JAR rebuild for `-Ddraft.required.agent`, and the shared-`latest.pt` caveat during a run
- [X] T051 [P] Add the `train-draft-agent-online` bullet to the `draft` subcommand list in `CLAUDE.md` (timeless present tense, no benchmark numbers), and note the two new shared modules `price_predictor/infrastructure/torch_training.py` and `draft/application/draft_training_common.py` in the key-modules lists for `price_predictor` and `draft`
- [X] T052 Performance review per Principle VIII against plan.md's Performance Review: confirm one shared memoizing `ConvertedCardLocator`, models/`prev_model`/scorer on CUDA when available, batches collated on-device, one batched `no_grad` diagnostics sweep (not per-pick math), the corpus handle opened once, and no per-item host↔device transfer outside `AgentPickService._select`'s inherent per-pick readout
- [X] T053 Run the full gate: `pytest` (unit + integration) and `ruff check src/ tests/` from the repo root, plus `cd forge-connector && mvn test`; fix every failure, warning, and IDE type diagnostic introduced by this feature
- [X] T054 Validate [quickstart.md](quickstart.md) end-to-end: the smoke run in its final section (`pytest tests/integration/test_train_draft_agent_online_smoke.py -q`) passes, and the Step 1 command line matches the flags actually registered in `src/draft/infrastructure/cli.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately
- **Foundational (Phase 2)**: depends on Phase 1; **blocks both user stories** (gen-3 must import the extracted helpers, not copy them a fifth time)
- **User Story 1 (Phase 3)**: depends on Phase 2 only
- **User Story 2 (Phase 4)**: depends on Phase 2; its round loop (T042) consumes the US1 update, so T042/T045 depend on Phase 3 being complete. T030–T032 and T036–T040 (the generation-side extensions) are independent of US1 and can proceed in parallel with Phase 3.
- **User Story 3 (Phase 5)**: no code dependency — documentation + verification only; can run any time after Phase 2
- **Polish (Phase 6)**: depends on Phases 3–5

### Within Phase 2

- T003 → T004, T005, T006, T007 (all four repoint the module T003 creates)
- T008 → T009
- T004–T007 + T009 → T010 → T011
- T012 is independent of the extractions

### Within User Story 1

- Tests T013–T017 first (must fail)
- T018 → T019 → T020 → T021 → T022 (loader → advantage → loss → update, same file, sequential)
- T023 depends on T022 (the pass needs pre/post-update weights)
- T024 → T022/T025; T026 → T027 → T029
- T028 depends on T020/T022/T023 (it formats their outputs)

### Within User Story 2

- Tests T030–T035 first (must fail)
- T036, T037, T038 are independent of each other but all feed T042
- **The `-Ddraft.required.agent` chain is T039 → T040 → T036(c) → T042, then T047**: Java rule → connector kwarg → config field + launcher forwarding → the learner label actually supplied → JAR rebuilt. Every link is required for FR-003; any one of them missing leaves learner-free pods being played with no error
- T036(b) (the `build_labeler` locator parameter) must land before T042, or the plan's one-shared-locator Performance Review claim silently fails
- T041 → T045; T042 → T043 → T044; T046 depends on T029
- T035 (integration smoke) passes only once T042–T046 land

### Parallel Opportunities

- T002 runs alongside T001
- T004, T005, T006, T007 in parallel once T003 lands; T010 in parallel across its four files
- All of T013–T017 in parallel (five distinct new test files)
- All of T030–T032, T034, T035 in parallel (distinct files); T033 waits on T016's file
- T039 (Java) and T040 (connector) in parallel with the Python loop work — but T040 lands before T036(c), which calls it
- T050 and T051 in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all five User Story 1 test files together:
Task: "Advantage tests in tests/unit/draft/test_online_advantage.py"
Task: "Loss tests in tests/unit/draft/test_online_loss.py"
Task: "Round-loader tests in tests/unit/draft/test_online_loader.py"
Task: "Diagnostics tests in tests/unit/draft/test_online_diagnostics.py"
Task: "CLI/startup-validation tests in tests/unit/draft/test_online_cli.py"
```

## Parallel Example: Phase 2 extraction

```bash
# After T003 creates price_predictor/infrastructure/torch_training.py:
Task: "Repoint src/sealed/application/train_picker.py"
Task: "Repoint src/sealed/application/train_scorer.py"
Task: "Repoint src/draft/application/train_draft_agent.py"
Task: "Repoint src/draft/application/train_draft_agent_rl.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1: Setup (green baseline + JAR builds)
2. Phase 2: Foundational — **critical**, blocks both stories
3. Phase 3: User Story 1 — the update rule, checkpoint, diagnostics, CLI, validation
4. **STOP and VALIDATE**: drive one round from a synthetic batch; confirm the checkpoint loads with `rl_metadata["algorithm"] == "online-grpo"` and the three axes print

### Incremental Delivery

1. Setup + Foundational → shared helpers have one home, existing suites green
2. + User Story 1 → a single online GRPO update is runnable and testable (MVP)
3. + User Story 2 → the streaming loop with a resident worker and a live anchor margin (the shippable feature)
4. + User Story 3 → the promotion runbook documented over existing commands

### Suggested MVP Scope

Phases 1–3 (T001–T029). That delivers the whole update rule, the checkpoint
contract, three of the four diagnostic axes, and fail-fast startup validation —
everything except generation, the resident worker, and the anchor margin.

---

## Notes

- `[P]` = different files, no dependencies on incomplete tasks
- Verify each story's tests fail before implementing it
- Commit after each task or logical group
- **Principle VII**: every new entity above cites its prior art — `OnlineExample`/`_collate` ← `train_draft_agent_rl.RLExample`/`_collate`; the round loader ← `train_draft_agent_rl._Loader`; the CLI subparser ← `_build_train_draft_agent_rl_parser`; the extracted helpers are moves, not new logic. No parallel concept is introduced.
- **Principle VIII**: T018, T019, T022, T023, T036(b), T042 all touch data loading or model compute and are checked against the performance checklist in T052
- The `-Ddraft.required.agent` change means the fat JAR **must** be rebuilt (T047) before the first real gen-3 run
- Two failure modes here are silent rather than loud, so they get explicit tasks and tests: an unset `required_agent` (learner-free pods play and are trained on as if valid — T036(c)/T042, tested in T030) and a corpus handle opened `"w"` (the canonical `drafts.jsonl` is truncated — T042)
