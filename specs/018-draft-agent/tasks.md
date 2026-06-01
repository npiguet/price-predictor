# Tasks: Draft agent — imitation policy + critic (generation 1)

**Input**: Design documents from `specs/018-draft-agent/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (cli.md, drafts-jsonl.md, worker-protocol.md), quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), test tasks are MANDATORY and included for every story.

**Organization**: Tasks are grouped by user story (P1 → P2 → P3) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1, US2, US3 (setup/foundational/polish carry no story label)
- Exact file paths are included in every task.

## Path Conventions

Single-project hexagonal package `src/draft/` (sibling of `src/sealed/`), Java
worker in the existing `forge-connector` module, tests under `tests/`. Paths
follow plan.md §Project Structure.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `draft` package skeleton and entry point.

- [ ] T001 Create the `src/draft/` package skeleton — `src/draft/__init__.py`, `src/draft/domain/__init__.py`, `src/draft/application/__init__.py`, `src/draft/infrastructure/__init__.py`, `src/draft/scripts/__init__.py` (empty package markers, mirroring `src/sealed/` layout per plan.md §Project Structure).
- [ ] T002 [P] Create the test package marker `tests/unit/draft/__init__.py` (mirrors `tests/unit/sealed/`).
- [ ] T003 Create `src/draft/__main__.py` dispatching `python -m draft <subcommand>` to `draft.infrastructure.cli` (copy the structure of `src/sealed/__main__.py`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Geometry + JSONL IO + CLI skeleton shared by both pipeline stories. FR-016 geometry is the one genuinely-new piece of domain logic (research §D4) and underpins both data-gen pool reconstruction (US1) and training state reconstruction (US2).

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Implement FR-016 booster↔seat/pick geometry in `src/draft/domain/draft_geometry.py`: derived sizes (`pod_size`, `packs`, `pack_size P`), forward map (`pack_number(k)`, `opening_seat(k)`, seat-of-pick-`j` with `dir_p`), the inverse map (`(seat s, pack p, pick i) → booster index k`, offset `j`, legal actions `picks[j:]`, taken `picks[j]`), and a "reconstruct one seat's full drafted pool" helper used by the US1 supervisor. Pure functions, no torch. Prior art: none — new domain logic (research §D4); follows `sealed/domain` purity conventions.
- [ ] T005 [P] Unit tests for geometry round-trip (SC-002) in `tests/unit/draft/test_draft_geometry.py`: forward/inverse consistency for all `(s,p,i)`, `dir_p` direction for packs 1 & 3 vs pack 2, derived-size extraction, and hand-worked records.
- [ ] T006 Implement JSONL corpus IO in `src/draft/infrastructure/draft_record_io.py`: append one compact record per line, line-by-line read that tolerates+skips a trailing partial final line, and a `--resume` complete-record count (FR-012, FR-013). Prior art: `sealed/infrastructure/pool_file_reader.count_complete_lines_and_truncate_partial` + `cards_played_reader` partial-line tolerance (research §Adjacent prior art).
- [ ] T007 [P] Unit tests for record IO in `tests/unit/draft/test_draft_record_io.py`: write→read round-trip, trailing-partial-line tolerance, and resume count of complete records.
- [ ] T008 Create the top-level argparse skeleton in `src/draft/infrastructure/cli.py` (`build_parser()` + `main()` with subparsers, `set_defaults(func=…)` dispatch), mirroring `src/sealed/infrastructure/cli.py`. Subcommand parsers are added by their respective story tasks (T017, T028).

**Checkpoint**: Geometry + IO + CLI skeleton ready; both user stories can begin.

---

## Phase 3: User Story 1 - Generate a labeled draft corpus (Priority: P1) 🎯 MVP

**Goal**: A `generate-draft-data` CLI that drives Forge's draft AI for all pod seats via a supervised Java worker, builds+scores a deck per seat with the frozen picker/scorer, and appends one self-contained JSON record per draft.

**Independent Test**: Run `python -m draft generate-draft-data --n-drafts N` and confirm `output/draft/drafts.jsonl` gains N parseable, self-contained records (per-seat agents/decks/scores + full per-booster pick transcript), reconstructable per FR-016; a mid-run worker crash does not abort the run.

### Tests for User Story 1 (MANDATORY) ✅

> Write these first; ensure they fail before implementation.

- [ ] T009 [P] [US1] Unit test for `--agent-mix` categorical sampling (independent per-seat draw, weights honored, identifier recorded verbatim — FR-006) in `tests/unit/draft/test_agent_mix.py`.
- [ ] T010 [P] [US1] Unit test for supervisor stdout handling — keep only `<<DRAFT-EVENT-JSON>>` lines, defensively `json.loads` the suffix, skip non-sentinel/parse-failure lines (FR-010, worker-protocol.md) — in `tests/unit/draft/test_draft_supervisor.py` (drive a pure parse/dispatch helper, no live JVM).
- [ ] T011 [P] [US1] Unit test for record assembly — failed deck build ⇒ `deck=[]` / `deck_score=null`, well-formed record fields (FR-007, FR-014) — in `tests/unit/draft/test_record_assembly.py`.
- [ ] T012 [P] [US1] JUnit test for `DraftWorkerMain` (Forge-dependent, `@Tag("integration")`) asserting a fully-drained 8-seat / 3-pack transcript with one flushed sentinel line per draft, in `forge-connector/src/test/java/com/pricepredictor/connector/DraftWorkerMainTest.java`.
- [ ] T013 [P] [US1] Integration smoke test (Forge-dependent, `integration` marker) for the end-to-end `generate-draft-data` pipeline producing parseable records, in `tests/integration/test_generate_draft_data.py`.

### Implementation for User Story 1

- [ ] T014 [US1] Implement `DraftWorkerMain.java` in `forge-connector/src/main/java/com/pricepredictor/connector/DraftWorkerMain.java`: drive Forge `BoosterDraft` + `LimitedPlayer` for all pod seats, apply `forge-r30`/`forge-r100` uniform-random pick overrides, emit one flushed `<<DRAFT-EVENT-JSON>>` line per completed draft (boosters with per-booster `set_code` + drained `picks`, per-seat `agent`; no deck/score), diagnostics to stderr (FR-006, FR-010, FR-015, research §D1). Prior art: `PoolMain`/`MatchWorkerMain` + `ForgeEnvironmentInitializer` + `MatchGenerator.computeEligibleSets()`.
- [ ] T015 [US1] Implement `src/draft/infrastructure/draft_worker_connector.py` to launch/kill `DraftWorkerMain` and stream its stdout, reusing `price_predictor/infrastructure/forge_jvm.py` (`build_jvm_command`, `build_forge_classpath`, `run_forge_worker`, `kill_process_tree`) exactly as `MatchWorkerConnector`/`PoolConnector` do (research §Adjacent prior art).
- [ ] T016 [US1] Implement the supervisor `src/draft/application/generate_draft_data.py`: one `run_id` UUID at startup (FR-005); spawn+restart the worker on crash (FR-011, SC-003); filter/parse sentinel lines (FR-010); per parsed transcript reconstruct each seat's full pool via `draft_geometry` (T004), build a deck per seat with `--build-method` (picker default via `deck_assembly.load_pool_embeddings` + `picker_model.decompose_picks` + `manabase.compute_basic_lands`; `greedy` via `GreedyDeckBuilder`), score the non-basic subset with the frozen scorer (`score_decks`), and append the completed record via `draft_record_io` (T006); failed build ⇒ `deck=[]`/`deck_score=null`; SIGINT clean stop (FR-007, FR-008, FR-011, FR-012, FR-014). Prior art: `sealed/application/match_outcomes.py` `MatchOutcomeSupervisor`, `evaluate_scorer.score_decks`, `pick_decks.py`.
- [ ] T017 [US1] Wire the `generate-draft-data` subparser into `src/draft/infrastructure/cli.py` with all flags + defaults from contracts/cli.md (`--n-drafts` required, `--set`, `--agent-mix`, `--scorer-checkpoint`, `--build-method`, `--picker-checkpoint`, `--cards-path`, `--output-path`, `--resume`) and `set_defaults(func=…)` calling T016.

**Checkpoint**: `generate-draft-data` produces a self-contained `drafts.jsonl`; US1 is independently testable (MVP).

---

## Phase 4: User Story 2 - Train the two-headed draft agent (Priority: P2)

**Goal**: A `train-draft-agent` CLI that turns each `(draft, seat, pack, pick)` into a typed-token training example, trains the imitation policy (whitelisted seats) and the MC-regression critic (all non-failed seats) jointly, logs the per-epoch loss split + val metrics, and saves the best checkpoint.

**Independent Test**: Run `python -m draft train-draft-agent` on a small corpus and confirm it writes `{timestamp}.pt` + `latest.pt` under `models/draft/agent/`, logs per-epoch loss decomposition + val imitation top-1/top-3 + per-pack critic MSE, selects best by val loss, and the checkpoint reloads to produce picks + critic scalars on held-out states.

### Tests for User Story 2 (MANDATORY) ✅

> Write these first; ensure they fail before implementation.

- [ ] T018 [P] [US2] Unit tests for typed-token state reconstruction in `tests/unit/draft/test_draft_state.py`: `[CONTEXT][POOL][PACK][PASSED][TAKEN]` layout (FR-017), every observed instance in exactly one type (FR-018), wheel-diff `PASSED→TAKEN` + survivors re-enter `PACK` (FR-019a), pack-end flush (FR-019b), and recency `packs_ago∈{0,1,2}` / `pick_ago` freeze-at-`packs_ago≥1` (FR-021), on hand-worked records incl. the wheel.
- [ ] T019 [P] [US2] Unit tests for the model in `tests/unit/draft/test_draft_agent_model.py`: default no-projection width `d_model = embedding_dim + 4 + d(packs_ago) + d(pick_ago)` and non-default `--d-model` inserts one `Linear` (FR-025); policy head produces one logit per `PACK` token with masked-softmax over PACK only (FR-027); critic head produces one scalar on the `CONTEXT` token (FR-028); padding masked everywhere (FR-023); `d_model % n_heads != 0` raises a fast architecture error (FR-026, SC-006).
- [ ] T020 [P] [US2] Unit tests for loss + targets in `tests/unit/draft/test_draft_loss.py`: imitation CE active on whitelisted seats only while critic stays active on all non-failed seats (FR-033); leave-one-out pod-relative reward excludes failed seats from the mean (FR-032); critic-target z-scoring over the training split and de-standardization round-trip (FR-032).
- [ ] T021 [P] [US2] Unit tests for the draft-disjoint split + checkpoint store in `tests/unit/draft/test_draft_agent_store.py`: all picks of a `draft_id` land on one side, first `--val-fraction` of distinct IDs with `random_seed=42` (FR-035); checkpoint round-trip carries `config` + standardization mean/std + epoch + best-val and reloads to produce picks/critic (FR-040, SC-004); architecture flags rejected with `--resume`/`--checkpoint` (FR-039, SC-006).
- [ ] T022 [P] [US2] Integration smoke test (`integration` marker) training to completion on a tiny fixture corpus and reloading the best checkpoint, in `tests/integration/test_train_draft_agent.py`.

### Implementation for User Story 2

- [ ] T023 [P] [US2] Implement typed-token state assembly in `src/draft/domain/draft_state.py`: build `[CONTEXT][POOL][PACK][PASSED][TAKEN]` instance sets + per-instance `(packs_ago, pick_ago)` for a target `(seat s, pack p, pick i)` by walking the seat's boosters with wheel-diff + pack-end-flush transitions, using `draft_geometry` (T004) (FR-017–FR-021). Pure domain logic.
- [ ] T024 [P] [US2] Implement `src/draft/domain/draft_agent_model.py`: `DraftAgentConfig` (data-model §4 fields incl. derived `embedding_dim`, `P`, recency widths) with fail-fast `d_model % n_heads` validation, and `DraftAgentModel` — optional input projection (`Identity` by default), `n_layers × SAB` trunk (`from sealed.domain.scorer_model import SAB`), per-`PACK`-token policy head, `CONTEXT`-token critic head, learned `packs_ago`/`pick_ago`/`pack_number`/`pick_number` tables, frozen card embeddings, type as a 4-dim one-hot (FR-020, FR-022, FR-024–FR-029). Prior art: `sealed/domain/picker_model.py` (template; new sibling justified per research §Overlapping vocabulary).
- [ ] T025 [P] [US2] Implement `src/draft/infrastructure/draft_agent_store.py` mirroring `sealed/infrastructure/picker_store.py`: save/load `{timestamp}.pt` + `latest.pt` under `models/draft/agent/` with `model_state_dict` + `config` + `epoch` + `best_val_loss` + training metadata incl. critic-target standardization mean/std (FR-040, FR-041). Prior art: `price_predictor/infrastructure/torch_checkpoint.py`, `PickerStore`.
- [ ] T026 [US2] Implement the corpus loader inside `src/draft/application/train_draft_agent.py`: read `drafts.jsonl` (T006), build one `(draft,seat,pack,pick)` example per pick via `draft_state` (T023), resolve card vectors through `ConvertedCardLocator` and warn+drop missing-`.npz` picks (≤20 names + total, FR-038), compute leave-one-out critic targets + training-split z-scoring (FR-032), apply the draft-disjoint split (FR-035), and pad per batch with attention masks (length bucketing permitted, FR-023, FR-036).
- [ ] T027 [US2] Implement the joint training loop in `src/draft/application/train_draft_agent.py`: loss `imitation_weight·CE(policy,taken)` over whitelisted seats + `critic_weight·MSE(critic,std-reward)` over non-failed seats (FR-033); AdamW + per-group max-norm clip + linear-warmup-then-constant LR (FR-034); per-epoch loss-split log + val imitation top-1/top-3 + per-`pack_number` critic MSE (FR-037, SC-005); best-by-val-loss checkpoint + `latest.pt` (FR-036); `--resume` (weights+optimizer+epoch+best-val) and `--checkpoint` (weights-only bootstrap), mutually exclusive, architecture flags forbidden, `--epochs`/`--patience` early stop (FR-039, SC-006). Prior art: `train_picker.py`/`train_scorer.py`/`train_encoder.py` warmup+clip+resume+early-stop helpers (research §Third-instance check — follow, do not extract).
- [ ] T028 [US2] Wire the `train-draft-agent` subparser into `src/draft/infrastructure/cli.py` with all flags + defaults from contracts/cli.md (`--drafts-path`, `--cards-path`, `--d-model`, `--n-layers`, `--n-heads`, `--ff-dim`, `--dropout`, `--imitation-weight`, `--critic-weight`, `--imitation-agents`, `--lr`, `--warmup-frac`, `--batch-size`, `--max-grad-norm`, `--epochs`, `--val-fraction`, `--patience`, `--resume`, `--checkpoint`), with startup fail-fast on `d_model % n_heads != 0` and on architecture flags supplied with `--resume`/`--checkpoint` (SC-006), and `set_defaults(func=…)` calling T027.

**Checkpoint**: `train-draft-agent` produces a reloadable best checkpoint; US1 and US2 both work independently.

---

## Phase 5: User Story 3 - Validate the picker as a label-builder (Priority: P3)

**Goal**: A one-off diagnostic *script* (not a subcommand) that builds a few hundred drafted pools both ways (picker + SA), scores both with the frozen scorer, and reports the gating picker-vs-SA Spearman, the SA−picker gap distribution, and the SA-vs-SA reference ceiling.

**Independent Test**: Run `python -m draft.scripts.validate_builder` over a few hundred pools and confirm it prints the picker-vs-SA Spearman, the score-gap median/spread, and the SA-vs-SA reference correlation — enough to decide `picker` vs `greedy`.

### Tests for User Story 3 (MANDATORY) ✅

> Write these first; ensure they fail before implementation.

- [ ] T029 [P] [US3] Unit test for the diagnostic statistics in `tests/unit/draft/test_validate_builder.py`: Spearman correlation, SA−picker gap median/spread, and SA-vs-SA reference computed correctly on synthetic score arrays (FR-042, SC-007), driving the pure stats helper without Forge/torch.

### Implementation for User Story 3

- [ ] T030 [US3] Implement the diagnostic logic in `src/draft/application/validate_builder.py`: over drafted pools (read from a `drafts.jsonl` via T004/T006 or fresh pools), build each pool with both the picker and `GreedyDeckBuilder`, score both with `score_decks`, and compute the picker-vs-SA Spearman, SA−picker gap median/spread, and SA-vs-SA reference across independent SA restarts (FR-042). Prior art: `deck_assembly.load_pool_embeddings`, `picker_model`, `greedy_deck_builder`, `evaluate_scorer.score_decks` (research §D7).
- [ ] T031 [US3] Implement the ~40-line entry `src/draft/scripts/validate_builder.py` parsing `--pools-from` / `--fresh-pools --set --n-pools` and printing the three reported numbers, calling T030 (quickstart §1, contracts/cli.md §Builder-validation).

**Checkpoint**: All three user stories independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, follow-ups, and end-to-end validation.

- [ ] T032 [P] Update `CLAUDE.md`: add the `draft` package to Architecture (alongside `price_predictor`/`sealed`), document the `generate-draft-data` + `train-draft-agent` subcommands, the `drafts.jsonl` format, the `DraftWorkerMain` Java main class, and the `models/draft/agent/` + `output/draft/` artifact layout (Principle VI; plan.md Constitution Check).
- [ ] T033 [P] Re-evaluate extracting a `train_common` helper (warmup `LambdaLR` lambda + per-group clip + resume/bootstrap guard + best/latest persistence) now that `train-draft-agent` is the fourth trainer — extract only the genuinely identical pieces, or record the decision to defer, resolving the `src/sealed/application/train_picker.py` `TODO(shared-trainer)` (research §Third-instance check; non-blocking follow-up surfaced in plan.md).
- [ ] T034 [P] `ruff check src/draft tests` clean and run `pytest tests/unit/draft/` to confirm the fast suite is green.
- [ ] T035 Run the quickstart.md end-to-end validation (build forge-connector JAR, generate a small corpus, train, inspect checkpoint, run the builder diagnostic) and confirm the SC-001…SC-007 checklist in quickstart.md §Validating against the success criteria.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately.
- **Foundational (Phase 2)**: depends on Setup — **blocks all user stories** (geometry T004, IO T006, CLI skeleton T008 are shared).
- **User Stories (Phase 3–5)**: each depends only on Foundational; US1 → US2 → US3 by priority, but US2/US3 do not depend on US1's code (they share only foundational modules), so they can proceed in parallel if staffed. US2 and US3 both reuse the foundational geometry/IO; neither imports US1's supervisor.
- **Polish (Phase 6)**: depends on the targeted stories being complete.

### Within Each User Story

- Tests (written first, failing) → implementation.
- Domain (state/model) before application (loader/loop) before CLI wiring.
- US1: worker (T014) and connector (T015) before supervisor (T016) before CLI (T017).
- US2: state/model/store (T023–T025) before loader/loop (T026–T027) before CLI (T028).
- US3: logic (T030) before script entry (T031).

### Parallel Opportunities

- Setup: T002 ∥ (T001→T003).
- Foundational: T004+T005 ∥ T006+T007 (geometry and IO are independent files); T008 after the parser-less modules exist.
- US1 tests T009–T013 all [P]; US2 tests T018–T022 all [P]; US3 test T029 [P].
- US2 implementation T023 ∥ T024 ∥ T025 (three independent files) before T026.
- Across stories: once Foundational lands, US1 / US2 / US3 can be built by separate developers (they touch disjoint files except the shared `cli.py`, edited by T017/T028 sequentially).
- Polish: T032 ∥ T033 ∥ T034 before T035.

---

## Parallel Example: User Story 2

```bash
# Tests first (all parallel — different files):
Task: "Unit tests for state reconstruction in tests/unit/draft/test_draft_state.py"
Task: "Unit tests for the model in tests/unit/draft/test_draft_agent_model.py"
Task: "Unit tests for loss + targets in tests/unit/draft/test_draft_loss.py"
Task: "Unit tests for split + store in tests/unit/draft/test_draft_agent_store.py"

# Then the three independent implementation files in parallel:
Task: "Implement src/draft/domain/draft_state.py"
Task: "Implement src/draft/domain/draft_agent_model.py"
Task: "Implement src/draft/infrastructure/draft_agent_store.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → Phase 2 Foundational (geometry + IO + CLI skeleton).
2. Phase 3 US1 → **stop and validate**: `generate-draft-data --n-drafts N` yields N self-contained, FR-016-reconstructable records and survives a worker crash. This corpus is independently valuable.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 (corpus) → test independently → the dataset exists.
3. US2 (trainer) → test independently → the model artefact exists.
4. US3 (builder diagnostic) → test independently → label-builder decision de-risked (run it *before* a large US1 corpus run when desired, since it depends only on foundational + sealed reuse).
5. Polish (docs + trainer-extraction re-eval + quickstart validation).

---

## Notes

- [P] = different files, no incomplete-task dependencies; [Story] maps each task to US1/US2/US3 for traceability.
- One-way dependency rule (FR-002): `draft` imports `sealed`/`price_predictor`, never the reverse — enforce in every new module.
- The only genuinely new logic is the FR-016 geometry (T004) + typed-token state (T023) + two-headed model (T024) + Java worker (T014); everything else reuses `sealed`/`price_predictor` per research §Codebase Survey.
- Verify each story's tests fail before implementing; commit after each task or logical group.
