---
description: "Task list for Draft agent — live Forge integration"
---

# Tasks: Draft agent — live Forge integration

**Input**: Design documents from `specs/019-draft-live-play/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, contracts/pick-protocol.md, quickstart.md

**Tests**: Mandatory per Constitution Principle I (Fast Automated Tests). The
gating online-tracker↔`build_state` equivalence test (SC-003) is load-bearing.

**Organization**: Tasks are grouped by user story. This feature *extends* the
existing `src/draft/` package and the existing `DraftWorkerMain` (no new package,
no new subcommand). Every task cites its prior art per Principle VII.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1 / US2 / US3 (omitted for Setup / Foundational / Polish)
- Exact file paths included in each description

## Path Conventions

Single-project hexagonal layout (plan.md §Project Structure): `src/draft/{domain,
application,infrastructure}`, Java worker under
`forge-connector/src/main/java/com/pricepredictor/connector/`, Python tests under
`tests/unit/draft/` and `tests/integration/`, Java tests under
`forge-connector/src/test/java/com/pricepredictor/connector/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the two new module stubs the feature adds; everything else is
an extension of existing files.

- [ ] T001 [P] Create empty module stubs with docstrings referencing data-model.md §2.4–2.6: `src/draft/domain/online_draft_state.py` (OnlineDraftStateTracker), `src/draft/application/agent_pick_service.py` (AgentPickService + PickFault), `src/draft/application/agent_registry.py` (AgentRegistry). No new package — these slot into the existing `draft` hexagonal layout (plan.md §Structure Decision).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The pure-domain online state reconstruction and the wire-protocol
message types — both required by *every* user story before any seat can be
agent-piloted.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T002 [P] Implement `OnlineDraftStateTracker` in `src/draft/domain/online_draft_state.py`: per-`(draft_id, seat)` walk holding `pool` / `last_seen` / `passed` / per-pack last-seen contents; `observe(request) -> DraftState` advancing the wheel-diff + pack-flush + `_recency` exactly as `draft_state.build_state` (research D3); `commit(card)` after the supervisor selects. REUSE `CardInstance`/`DraftState`/`_recency` from `src/draft/domain/draft_state.py` and pod/pack/pick/`direction`/wheel conventions from `src/draft/domain/draft_geometry.py` (prior art: research §codebase survey). Pure domain — no torch, no IO.
- [ ] T003 [P] Gating equivalence test in `tests/unit/draft/test_online_draft_state.py`: build a finished synthetic `DraftRecord`, replay each model-seat's pick-requests through the tracker, and assert the emitted `DraftState` (typed-token multiset + per-instance `(packs_ago, pick_ago)` + `pack_actions` + revealed `target`) equals `draft_state.build_state(record, geometry, seat, pack, pick)` at **every** `(seat, pack, pick)` (SC-003, research D3).
- [ ] T004 Define live-pick message types + sentinel parsers in `src/draft/application/generate_draft_data.py`: `PickRequest` (parse `<<DRAFT-PICK-REQUEST>>`), `PickResponse` (serialize `<<DRAFT-PICK-RESPONSE>>` with `pick` xor `abort`), `AbandonedNotice` (parse `<<DRAFT-ABANDONED>>`), per data-model.md §2.1–2.3 + contracts/pick-protocol.md. REUSE the existing `parse_sentinel_line` pattern (research §adjacent prior art); add sibling parsers only.
- [ ] T005 [P] Unit test in `tests/unit/draft/test_pick_protocol.py` for the message round-trip + validation: well-formed `<<DRAFT-PICK-REQUEST>>` parses; `PickResponse` enforces exactly one of `pick`/`abort`; request field validation (`pod_size == 8`, `1 ≤ pack_number ≤ packs`, non-empty `pack`) per data-model.md §2.1.

**Checkpoint**: Online state reconstruction is pinned to `build_state`; the wire
messages parse/serialize. User-story work can begin.

---

## Phase 3: User Story 1 - Pilot a trained agent as a live draft seat (Priority: P1) 🎯 MVP

**Goal**: Bind one checkpoint to a mix label and run `generate-draft-data`; agent
seats pick via the trained policy, each completed draft is labeled/scored and
appended to `drafts.jsonl`, faults abandon the whole draft (never substitute),
and a deterministic fault auto-aborts the run.

**Independent Test**: Run the command with `--agent-checkpoint draft-agent=<ckpt>`
and a mix assigning `draft-agent` to ≥1 seat; confirm new records appear whose
agent seats carry that label with built+scored decks and the policy's own picks.

### Tests for User Story 1 (MANDATORY) ✅

> Write these FIRST and ensure they FAIL before implementation.

- [ ] T006 [P] [US1] Unit test in `tests/unit/draft/test_agent_pick_service.py`: argmax pick masks to PACK and returns a held-pack card; individual un-embeddable cards are dropped (normal, FR-... edge case); a request whose PACK is *entirely* un-embeddable raises `PickFault`; a `pick_number` exceeding the checkpoint's `P` is a `PickFault` (data-model.md §2.5, §3).
- [ ] T007 [P] [US1] Unit test in `tests/unit/draft/test_agent_registry.py`: `LABEL=PATH` parsing incl. bare-`PATH`→`draft-agent`; unknown mix label (neither Forge built-in nor bound) fails fast exit 2 (FR-011); `config.packs ≠ live PACKS` or `config.P < pack size` fails fast exit 2 (FR-012); missing checkpoint file fails fast (contracts/cli.md §Validation).
- [ ] T008 [P] [US1] Unit test in `tests/unit/draft/test_supervisor_pick_routing.py` driving the supervisor read loop with a **fake worker** stream: a `<<DRAFT-PICK-REQUEST>>` is answered with the registry's pick; a Python-side `PickFault` sends `abort:true` and the draft is dropped (no record, SC-002); a `<<DRAFT-ABANDONED>>` is logged + counted; the consecutive-fault counter resets on any completed `<<DRAFT-EVENT-JSON>>`; `--max-consecutive-faults` consecutive faults raises a fatal nonzero-exit error (FR-016, SC-008); with an empty registry the loop emits no responses and behaves identically to gen-1 (SC-004).
- [ ] T009 [P] [US1] Java unit test (non-Forge) in `forge-connector/src/test/java/com/pricepredictor/connector/DraftWorkerMainTest.java`: `<<DRAFT-PICK-REQUEST>>` payload formatting (data-model.md §2.1 fields); response routing-field validation (`draft_id`/`seat`/`pack_number`/`pick_number` must match outstanding request; `pick` must be a held-pack card); mismatch ⇒ `<<DRAFT-ABANDONED>>` emitted; `abort:true` ⇒ draft dropped.
- [ ] T025 [P] [US1] Extend `tests/unit/draft/test_draft_resume.py`: a corpus containing records whose seats carry a model label is counted by `--resume` toward `--n-drafts` exactly as gen-1 (FR-014), and an abandoned (pick-fault) draft leaves no partial record that resume would miscount (SC-005). REUSE the gen-1 resume-count path in `src/draft/infrastructure/draft_record_io.py`.

### Implementation for User Story 1

- [ ] T010 [US1] Implement `AgentPickService` in `src/draft/application/agent_pick_service.py`: holds one `DraftAgentModel` (moved to CUDA-if-available once), the per-label `OnlineDraftStateTracker` registry, and the `ConvertedCardLocator`; `pick(request) -> str` = tracker.observe → embed (drop un-embeddable via the locator's per-name memo) → one-example tensorization on the model device → forward → mask logits to PACK → argmax → name → tracker.commit; raise `PickFault` when no PACK card is embeddable (data-model.md §2.5, research D4). Prior art: gen-1 `_PickerLabeler` device co-location; reimplement a minimal one-example tensorizer (research §adjacent prior art "Tiny single-example tensorization"). Principle VIII: model/locator load-once, memoized `.npz`, single host↔device sync per pick.
- [ ] T011 [US1] Implement `AgentRegistry` in `src/draft/application/agent_registry.py`: build label→`AgentPickService` map from `agent_checkpoints`; validate every mix label is a Forge built-in or bound (FR-011) and each checkpoint's `config.packs`/`config.P` against live geometry (FR-012, research D5; mirror `train_draft_agent._check_dims`); expose the external-label set for the worker. Loads checkpoints via the reused `DraftAgentStore` (`src/draft/infrastructure/draft_agent_store.py`).
- [ ] T012 [US1] Extend CLI in `src/draft/infrastructure/cli.py`: add `--agent-checkpoint LABEL=PATH` (repeatable) and `--max-consecutive-faults` (default 5); populate the new `GenerateDraftDataConfig` fields (data-model.md §4); validation failures exit `2` (contracts/cli.md). Argmax-only here; `--pick-mode`/`--temperature`/`--seed` land in US3.
- [ ] T013 [US1] Extend the connector in `src/draft/infrastructure/draft_worker_connector.py`: open the worker with `stdin=PIPE`, redirect worker stderr to a per-run log `output/draft/worker-<run_id>.log` (research D8, FR-015), and pass `-Ddraft.external.agents=<labels>` (contracts/pick-protocol.md §Worker JVM properties). Empty external set ⇒ unchanged gen-1 launch.
- [ ] T014 [US1] Extend `DraftWorkerMain.java` in `forge-connector/src/main/java/com/pricepredictor/connector/DraftWorkerMain.java`: allocate `draft_id` at draft start and thread it through the pick loop + transcript (research D2); branch `decidePick` for external labels to emit `<<DRAFT-PICK-REQUEST>>` and block on stdin for `<<DRAFT-PICK-RESPONSE>>` under strict synchrony (≤1 outstanding); validate routing fields + held-pack membership; on mismatch/garbled response emit `<<DRAFT-ABANDONED>>` and drop the draft; on `abort:true` drop the draft; on stdin EOF exit (supervisor restart path). Keep UTF-8 sentinel-on-stdout / Forge-chatter-to-stderr discipline.
- [ ] T015 [US1] Extend the supervisor in `src/draft/application/generate_draft_data.py`: build the `AgentRegistry` (skip entirely when `agent_checkpoints` is empty, SC-004); in the single read loop route a parsed `PickRequest` → `registry.pick` → write `PickResponse` (`pick` or `abort` on `PickFault`); reset per-draft trackers on completion; handle `AbandonedNotice` (log prominently + increment the consecutive-fault counter); reset the counter on any completed draft; raise a fatal nonzero-exit error at `--max-consecutive-faults` (FR-016, research D7); add startup + per-draft progress logging of model seats and a prominent `ERROR` on abandonment (FR-013). REUSE gen-1 labeling/scoring/JSONL-append/run-id/crash-restart/`--resume` count unchanged (FR-014); confirm resume counts pre-existing model-labeled records toward `--n-drafts` (no re-piloting of already-recorded drafts).

**Checkpoint**: A single trained agent can pilot live seats; faults abandon
cleanly; deterministic faults auto-abort. US1 is an independently testable MVP.

---

## Phase 4: User Story 2 - Measure agent strength against Forge in the same pod (Priority: P2)

**Goal**: A mixed pod scores agent and Forge seats on the same scale from the
same boosters, so a per-draft agent-minus-Forge delta is computable with no
cross-pod normalization.

**Independent Test**: Run a mix with both agent and Forge labels; confirm each
completed record holds both seat kinds from the same boosters, each with a
comparable `deck_score`.

> No new production code: gen-1 labeling already scores **every** seat regardless
> of who piloted it (research §adjacent prior art "Deck build + score per seat").
> This story is verified, not built.

- [ ] T016 [P] [US2] Unit test in `tests/unit/draft/test_mixed_pod_scoring.py`: assemble a completed mixed-pod `DraftRecord` (agent + `forge-full` seats) and assert every seat carries a `deck` + `deck_score` on the same scale from the shared `boosters`, so a per-draft agent-minus-Forge `deck_score` delta is well-defined (SC-006). REUSE the gen-1 record-assembly path exercised in `tests/unit/draft/test_record_assembly.py`.

**Checkpoint**: Mixed-pod records yield a within-pod agent-vs-Forge scoreboard.

---

## Phase 5: User Story 3 - Configure rollouts: rival checkpoints and pick determinism (Priority: P3)

**Goal**: Bind multiple labels to different checkpoints in one run, and choose
between argmax and seeded temperature-sampled picks.

**Independent Test**: Run two labels bound to two checkpoints and confirm each
checkpoint's seats record under its own label; run twice with `--pick-mode sample
--seed S` and confirm identical agent-seat picks.

### Tests for User Story 3 (MANDATORY) ✅

- [ ] T017 [P] [US3] Unit test in `tests/unit/draft/test_agent_pick_modes.py`: `sample` mode draws from `softmax(logits/temperature)` over PACK positions and is **identical across two services seeded the same** (SC-007, FR-005); `--temperature ≤ 0` with `--pick-mode sample` fails fast exit 2 (contracts/cli.md); two distinct `--agent-checkpoint` labels build distinct services in the registry (FR-010).

### Implementation for User Story 3

- [ ] T018 [US3] Extend `AgentPickService` in `src/draft/application/agent_pick_service.py` with a `sample` pick mode: sample the PACK-masked `softmax(logits/temperature)` using a per-service RNG seeded from `--seed` (research D4, FR-005, SC-007). Argmax path from T010 unchanged.
- [ ] T019 [US3] Extend the CLI/config in `src/draft/infrastructure/cli.py` (and thread through `src/draft/application/generate_draft_data.py`): add `--pick-mode {argmax,sample}` (default `argmax`), `--temperature` (default 1.0), `--seed`; `--temperature ≤ 0` with `sample` exits 2; pass mode/temperature/seed into the registry so each service is constructed accordingly (data-model.md §4). Multi-`--agent-checkpoint` binding is already handled by the registry (T011) — confirm end-to-end.

**Checkpoint**: Rival checkpoints and reproducible sampled rollouts are
configurable; all three stories independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T020 [P] Update the `generate-draft-data` documentation in **both** `CLAUDE.md` and the root `README.md`: the model-pilot flags (`--agent-checkpoint`, `--pick-mode`, `--temperature`, `--seed`, `--max-consecutive-faults`), the pick side-channel, fault-abandon/auto-abort semantics, and the per-run worker stderr log (Constitution VI Quality Gate — docs complete for changed CLI commands; FR-013/FR-015).
- [ ] T021 [P] Forge-dependent integration smoke test in `tests/integration/test_draft_live_play.py` (pytest `integration` marker): run one live-JVM draft with ≥1 model seat and assert a labeled record with the agent label + scored decks is appended (US1 end-to-end). Pair with the Forge-tagged Java protocol test (`@Tag("integration")`) in `DraftWorkerMainTest.java`.
- [ ] T022 Performance review per Principle VIII over the new code (`agent_pick_service.py`, supervisor loop): confirm model/registry/locator load-once, `.npz` embeddings memoized, single host↔device sync per pick, no batching needed under strict synchrony (plan.md §Performance Review).
- [ ] T023 Run the `specs/019-draft-live-play/quickstart.md` workflows 1–3 and the SC-004 no-`--agent-checkpoint` regression check; record results.
- [ ] T024 [P] Record the non-blocking follow-up surfaced in plan.md §Codebase Survey / research.md §third-instance check — evaluate factoring a shared wheel-diff/pack-flush/recency kernel across `draft_state.build_state`, `train_draft_agent._Loader._emit_seat`, and `online_draft_state.OnlineDraftStateTracker` (alongside the gen-1 `train_common` follow-up) — as a tracked backlog note. Not a gate for this feature.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup; **blocks all user stories**. T002/T003 (tracker + gating test) and T004/T005 (protocol messages) are the prerequisites every pick path needs.
- **US1 (Phase 3)**: depends on Foundational. The MVP.
- **US2 (Phase 4)**: depends on US1 producing records; adds only a verification test (no new code).
- **US3 (Phase 5)**: depends on US1 (extends `AgentPickService` + CLI); independent of US2.
- **Polish (Phase 6)**: after the desired stories land.

### Within User Story 1

- Tests T006–T009 + T025 before implementation T010–T015.
- T010 (pick service) before T011 (registry wraps services) and before T015 (supervisor calls registry).
- T004 (message types) before T014/T015 (worker + supervisor use them).
- T012 (CLI flags / config fields) before T015 (supervisor reads config).
- T013 (connector wiring) + T014 (worker protocol) before T015 end-to-end routing.

### Within User Story 3

- T017 (test) before T018/T019.
- T018 (sample mode in service) and T019 (CLI flags + threading) can proceed together but T019 depends on T018's constructor signature.

### Parallel Opportunities

- T002 and T004 are different files → parallel; T003 and T005 (their tests) parallel.
- US1 tests T006, T007, T008, T009, T025 are different files → all parallel.
- T010 and T014 (Python service vs Java worker) touch different files → parallel after Foundational; T011 depends on T010.
- Polish T020, T021, T024 are parallel (different files).

---

## Parallel Example: User Story 1 tests

```bash
# Launch the US1 test files together (all fail before implementation):
Task: "Unit test agent pick service in tests/unit/draft/test_agent_pick_service.py"
Task: "Unit test agent registry in tests/unit/draft/test_agent_registry.py"
Task: "Unit test supervisor pick routing in tests/unit/draft/test_supervisor_pick_routing.py"
Task: "Java pick-protocol test in forge-connector/.../DraftWorkerMainTest.java"
Task: "Resume count with model labels in tests/unit/draft/test_draft_resume.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → Phase 2 Foundational (tracker pinned to `build_state`, messages parse).
2. Phase 3 US1 → a single agent pilots live seats, faults abandon cleanly, drafts are recorded.
3. **STOP and VALIDATE** US1 independently (run the command with one checkpoint; inspect new records; force a fault and confirm abandonment + auto-abort).

### Incremental Delivery

- US1 → MVP self-play corpus generator.
- US2 → within-pod agent-vs-Forge scoreboard (verification only).
- US3 → rival checkpoints + seeded sampled rollouts.

---

## Notes

- [P] = different files, no dependencies; [Story] maps each task to a user story.
- The gating test (T003, SC-003) is the single load-bearing correctness property — do not skip or weaken it.
- SC-004 (zero behavior/format change with no model labels) must hold after every US1 task; the empty-registry path stays gen-1-identical.
- Principle VII: every new entity/service task above cites its prior art; the two genuinely new pieces (T002 tracker, T010 pick service) are justified in research.md §codebase survey.
- Principle VIII: T010 and T022 are the data-loading/model-compute checkpoints.
- Commit after each task or logical group.
</content>
</invoke>
