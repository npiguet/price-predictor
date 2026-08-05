# Research: Draft Agent — Online Self-Play GRPO Trainer (Generation 3)

**Feature**: 021-draft-online-grpo | **Date**: 2026-08-05
**Inputs**: [spec.md](spec.md); normative source
[`specs/2026-08-04-draft-agent-gen3-online-grpo.md`](../2026-08-04-draft-agent-gen3-online-grpo.md);
rationale
[`experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md`](../../experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md)
and [`experiments/2026-06-10-draft-agent-gen2-design.md`](../../experiments/2026-06-10-draft-agent-gen2-design.md).

The feature is **one new application use case** (`train-draft-agent-online`) that
fuses two existing halves that today live in separate processes: the live-play
draft-data supervisor (Forge worker + pick side-channel + deck labeling) and a
draft-agent trainer (state walk + batched policy forward + AdamW loop). The new
code is the round loop that alternates them in one process, the single-term GRPO
update, and the four-axis diagnostics. Everything else is reuse or a small,
named extension.

## Codebase Survey

### Overlapping domain vocabulary

| Existing concept | File / symbol | Decision |
|---|---|---|
| Two-headed agent (policy + critic) | `draft/domain/draft_agent_model.py` — `DraftAgentModel`, `DraftAgentConfig` | **Reuse unchanged.** The learner is the same model; only the policy head is trained. The critic head is carried through untouched (spec FR-027). |
| Typed-token per-pick state | `draft/domain/draft_state.py`, `draft/domain/draft_geometry.py`, `draft/application/draft_pick_states.py` — `iter_seat_pick_states` | **Reuse unchanged.** Every training state is the same `(seat, pack, pick)` state the gen-1 imitation loader and the gen-2 RL loader build (spec FR-007). |
| Online (request-stream) state reconstruction | `draft/domain/online_draft_state.py` — `OnlineDraftStateTracker` | **Reuse unchanged.** Used at *pick* time inside the pick service; the trainer separately rebuilds states from the finished record (D6). |
| Live policy inference for one label | `draft/application/agent_pick_service.py` — `AgentPickService` | **Extend (additive constructor).** Add `AgentPickService.from_model(...)` so a service can wrap an already-constructed `DraftAgentModel` instead of loading a checkpoint path — the mechanism by which the learner label is piloted by the live in-training policy (D3, spec FR-012). Existing path-based construction unchanged. |
| Label → service registry + label validation | `draft/application/agent_registry.py` — `AgentRegistry`, `FORGE_BUILTINS`, `parse_agent_checkpoints` | **Extend (additive kwarg).** Add an optional `preloaded: dict[str, AgentPickService]` to `AgentRegistry.build` so the learner's live-model service joins the frozen path-loaded ones and counts as a bound label during validation (D4). `parse_agent_checkpoints` is reused verbatim for `--learner` / `--frozen` `NAME=PATH` parsing. |
| Agent-mix categorical | `draft/application/agent_mix.py` — `parse_agent_mix`, `format_agent_mix` | **Reuse unchanged** for `--mix`. |
| Draft-data supervisor (worker lifetime, pick routing, record assembly, fault counting) | `draft/application/generate_draft_data.py` — `GenerateDraftDataSupervisor`, `assemble_record`, `build_labeler` | **Extract + reuse.** The worker-driving body of `_supervise` becomes a public `iter_records()` generator that yields assembled `DraftRecord`s forever and restarts a crashed worker; `run()` is rewritten as its consumer (counting, appending, progress, target). The online loop holds the same generator **suspended** between rounds — that suspension *is* the resident worker (D1, spec FR-005). No parallel supervisor is introduced. |
| Pod-relative leave-one-out reward | `train_draft_agent._leave_one_out_rewards` **and** `train_draft_agent_rl._leave_one_out_rewards` (already duplicated) | **Extract to a shared module** (third-instance rule; see below). Same formula and failed-build handling the gen-3 reward needs (spec FR-008). |
| Length-bucketed batching | `train_draft_agent.length_bucketed_batches` **and** `train_draft_agent_rl.length_bucketed_batches` (already duplicated) | **Extract to the same shared module.** Reused verbatim for the round's one pass. |
| Masked policy distribution helpers | `train_picker._masked_log_softmax/_policy_entropy/_kl_penalty`, `train_draft_agent_rl._masked_log_softmax/_policy_entropy/_kl_divergence`, `train_draft_agent._masked_log_softmax` | **Extract to a shared torch module** (third-instance rule). The gen-3 loss and the exploration/movement diagnostics need exactly these, with the same `0·-inf` NaN-gradient guards. |
| LR scaffolding (warmup-then-constant, per-group clip, plateau controller) | `train_draft_agent._make_scheduler`, `_PlateauLR`, `_reapply_resume_lr`, `_select_device`; `train_picker._clip_per_group`, `train_scorer._clip_per_group` | **Import (as gen-2 already does)** for the scheduler/device helpers; **extract** `clip_per_group` (already duplicated twice, inlined twice more). Gen-3 uses warmup-then-constant with **no** plateau annealing (spec FR-026 forbids a held-out-loss guard, which is what drives the plateau counter). |
| Checkpoint persistence + RL metadata | `draft/infrastructure/draft_agent_store.py` — `DraftAgentStore`, `LoadedDraftAgentCheckpoint.rl_metadata` | **Reuse unchanged.** `rl_metadata` is already a free-form dict; gen-3 writes `algorithm="online-grpo"` plus its own hyper-parameters (spec FR-027). No store change needed. |
| Corpus IO | `draft/infrastructure/draft_record_io.py` — `append_record`, `read_records` | **Reuse unchanged.** One append-mode handle held open for the run (spec FR-020). |
| Per-agent deck-score summary | `draft/application/analyze_generated_decks.py` — `deck_score_summary` | **Reuse (post-hoc read + yardstick).** The *live* anchor margin is computed in-loop from `deck_score`s already in hand; the same figure is recomputable from the corpus with this command (spec FR-020, US3). |
| Forge worker launcher | `draft/infrastructure/draft_worker_connector.py` — `DraftWorkerConnector.start` | **Extend (one optional kwarg).** Forward the new `-Ddraft.required.agent` system property (D2). |
| Java draft worker | `forge-connector/.../DraftWorkerMain.java` — per-seat `mix.sample(random)` | **Extend (~6 lines).** Honour `-Ddraft.required.agent`: after sampling the pod's agents, if none carries the required label, overwrite one uniformly-chosen seat with it (spec FR-003's "force one seat to the learner"). |

No parallel domain concept is introduced and no rename is needed: gen-3 reuses
`DraftRecord`/`Seat`/`Booster`, the typed-token state, the mix-label vocabulary,
and the checkpoint format as-is.

### Adjacent prior art

| Sub-problem | Existing solution | Decision |
|---|---|---|
| Driving Forge for drafts, restarting a crashed worker | `GenerateDraftDataSupervisor._supervise` | **Reuse via extraction** (D1). Not reimplemented. |
| Answering live pick requests / fault handling / consecutive-fault abort | `GenerateDraftDataSupervisor._route_pick_line`, `_answer_pick`, `_register_fault`; `contracts/pick-protocol.md` (spec 019) | **Reuse unchanged.** Gen-3 inherits the "abandon the draft, never substitute" discipline and `--max-consecutive-faults`. |
| Building + scoring each seat's pool (the reward source) | `generate_draft_data.build_labeler`, `_GreedyLabeler`, `_PickerLabeler` (batched per pod) | **Reuse unchanged.** `--scorer-checkpoint` / `--build-method` / `--picker-checkpoint` / `--cards-path` mean exactly what they mean for `generate-draft-data` (spec FR-005). |
| Shared card-embedding table + per-pick collate | `train_draft_agent_rl._Loader`, `_collate`, `RLExample` | **Mirror structurally** (a smaller sibling): the gen-3 loader drops `critic_active`, `value`, GAE fields and the multi-corpus split, keeping `card_idx`-into-shared-table memoization and the length-bucketed collate. |
| `.npz` card vectors | `sealed/infrastructure/converted_card_locator.py` — `ConvertedCardLocator.load_embedding` (memoized by name) | **Reuse unchanged.** One locator instance shared by the labeler, the pick services, and the trainer, so each card's `.npz` is decompressed once per run (D14). |
| Batched policy forward at temperature `T` over PACK positions | `agent_pick_service._select` (single pick), `train_draft_agent_rl._compute_loss` (batched) | **Reuse the batched form.** The update and the diagnostics pass are the same masked-softmax-over-`PACK`-at-`T` computation, batched. |
| Checkpoint round-trip + architecture/width fail-fast | `train_draft_agent_rl._check_dims`, `AgentRegistry.build` geometry checks | **Reuse.** Same messages/behaviour for the learner's warm start (spec FR-024). |
| CLI subparser + lazy import + `set_defaults(func=…)` + exit codes | `draft/infrastructure/cli.py` | **Reuse the convention** for `train-draft-agent-online`. `cli_resume.resolve_resumable_args` is **not** used — gen-3 has no `--resume` (each run is a fresh online run from `--learner`'s warm start). |

### Convention alignment

Mirror the `train-draft-agent-rl` sibling (spec 020) exactly:

- New use case at `src/draft/application/train_draft_agent_online.py`; CLI
  subparser `train-draft-agent-online` in `src/draft/infrastructure/cli.py`
  (lazy application import inside `run_*`).
- Pure/near-pure helpers get one fast unit-test file each under
  `tests/unit/draft/`; the end-to-end path gets one integration smoke test under
  `tests/integration/` with a fake worker (the pattern of
  `tests/unit/draft/test_supervisor_pick_routing.py` and
  `tests/integration/test_draft_supervisor_restart.py` — no JVM required).
- Checkpoints via `DraftAgentStore` to `models/draft/agent/`.
- Dependency direction unchanged: `draft` → `sealed` → `price_predictor`, never
  the reverse. The extracted torch helpers therefore land in `price_predictor`
  (the only package `sealed` may import from).
- Java changes stay inside `forge-connector`, tested by a JUnit test alongside
  `DraftWorkerMainTest.java`.

### Third-instance check

Four helpers are already duplicated in the codebase and gen-3 would be the next
copy. Per Principle VII these are **extracted now**, not copied again:

| Helper | Current copies | Extract to |
|---|---|---|
| `masked_log_softmax`, `policy_entropy`, `kl_divergence` | `train_picker` (as `_kl_penalty`), `train_draft_agent_rl`, `train_draft_agent` (log-softmax only) | `src/price_predictor/infrastructure/torch_training.py` (**new**; sits beside the existing `torch_checkpoint.py`, importable by both `sealed` and `draft`) |
| `clip_per_group` | `train_picker`, `train_scorer` (+ inlined in both draft trainers) | same module |
| `leave_one_out_rewards` | `train_draft_agent`, `train_draft_agent_rl` | `src/draft/application/draft_training_common.py` (**new**, sibling of the already-extracted `draft_pick_states.py`) |
| `length_bucketed_batches` | `train_draft_agent`, `train_draft_agent_rl` | same module |

Scope control: the extraction is mechanical (identical bodies), and the existing
call sites are updated in the same change so the codebase converges rather than
gaining a fifth copy. Existing unit tests import several of these by their
current private module names
(`tests/unit/draft/test_draft_loss.py`,
`tests/unit/draft/test_length_bucketing.py`,
`tests/unit/sealed/application/test_train_picker.py`,
`tests/unit/sealed/application/test_train_scorer.py`) — those imports are
repointed at the shared module in the same task.

**Still deferred** (unchanged standing decision from specs 017/018/020): the
whole training-loop scaffold is *not* extracted. Each trainer's
loss/metric/dataset bodies genuinely diverge, and gen-3's loop diverges the most
(no val split, no early stop, no epochs — rounds interleaved with generation).
`_make_scheduler` / `_PlateauLR` / `_select_device` continue to be imported from
`train_draft_agent`, as `train_draft_agent_rl` already does.

## Technical decisions

### D1 — The loop: a suspended record generator owning one resident worker

**Decision**: Extract the worker-driving body of
`GenerateDraftDataSupervisor._supervise` into a public generator

```python
def iter_records(self, launch, labeler) -> Iterator[DraftRecord]
```

that launches a worker, routes pick lines, assembles each transcript into a
`DraftRecord`, yields it, and relaunches on worker exit — forever, until
shutdown. `run()` becomes its consumer (append + count + progress + target).
The online trainer creates the generator once and pulls `--drafts-per-round`
records per round, leaving it **suspended** in between; `close()` at run end
terminates the JVM via the generator's `finally`.

**Rationale**: suspension is exactly the resident-worker semantics FR-005 asks
for, with zero new process-management code. Back-pressure is automatic and safe:
because every pod has ≥1 learner seat (D2) and the pick protocol is strictly
synchronous (spec 019 contract), the worker blocks on stdin at the first learner
pick of the *next* draft the moment we stop reading, so it can neither run ahead
nor fill the pipe. It also means no draft ever straddles a weight swap: a draft
begun after the boundary has *all* of its learner picks answered by the updated
policy.

**Alternatives rejected**: (a) spawning `generate-draft-data` per round — pays
Forge's ~20 s JVM startup every round (explicitly rejected by the spec);
(b) a second, parallel supervisor class for streaming — duplicates the
restart/fault/labeling logic that is already tested.

### D2 — Guaranteeing ≥1 learner seat per pod: a forced seat in the Java worker

**Decision**: New JVM system property `-Ddraft.required.agent=<label>`, forwarded
by `DraftWorkerConnector.start`. After `DraftWorkerMain` samples the pod's eight
agents from the mix, if none equals the required label it overwrites one
uniformly-chosen seat with it. Absent/blank ⇒ today's behaviour exactly.

**Rationale**: this is the spec's "force one seat to the learner" (FR-003), done
where the sampling actually happens. It is ~6 lines, needs no protocol change,
and is unit-testable in `DraftWorkerMainTest`.

**Alternative rejected**: discarding learner-free transcripts on the Python side.
The pod would already have been drafted and (worse) the eight decks built and
scored before we could tell — and FR-003 requires such a pod never be played.
It would also silently distort the realised mix.

**Plumbing** (the whole chain, so no link is left dangling):
`GenerateDraftDataConfig` gains an optional `required_agent: str | None = None`;
`GenerateDraftDataSupervisor._default_launch_worker`
(`generate_draft_data.py:535`) forwards it to
`DraftWorkerConnector.start(required_agent=…)`, which adds the property only when
non-`None`; the online trainer sets it to the learner label on the config it
builds. Existing callers pass nothing and keep byte-for-byte behaviour. The Java
change alone is inert — the launch site is what arms it.

**Cost**: the fat JAR must be rebuilt (`cd forge-connector && mvn package
-DskipTests`) before the first gen-3 run; called out in quickstart.md.

### D3 — The learner seat is piloted by the live in-training model instance

**Decision**: `AgentPickService.from_model(model, config, locator, …)` wraps an
existing `DraftAgentModel` rather than loading a checkpoint. The trainer passes
the very model it optimises, so "push the updated weights into the pick service"
(FR-012) is structurally a no-op — the service always reads current weights. The
trainer owns mode/device: `model.eval()` for the generation phase,
`model.train()` for the update phase. `from_model` does not call `.eval()`/`.to()`
itself (documented precondition), unlike the path-loading constructor which owns
a model nobody else touches.

**Rationale**: the strongest possible form of FR-012's on-policy guarantee — there
is no copy that can go stale — and it removes a per-round 50+ MB `state_dict`
clone.

**Alternative rejected**: `service.load_state_dict(model.state_dict())` each
round. Identical behaviour, extra copy, and one more way to forget the push.

### D4 — Registry composition: frozen labels from paths, learner pre-built

**Decision**: `AgentRegistry.build(frozen_paths, mix_labels, …, preloaded={learner_label: learner_service})`.
`preloaded` services are merged into the service map and treated as bound labels
for FR-003 validation; the geometry checks (`config.packs == PACKS`,
`config.P ≥ pack_size`) run over preloaded services too.

**Rationale**: keeps one place that owns "every mix label is a built-in, a frozen
agent, or the learner", instead of a second validation path in the trainer.
Avoids loading the learner checkpoint twice (once into the trainer, once into a
throwaway service).

### D5 — Every model-piloted seat samples at the same `-T`

**Decision**: The learner *and* the frozen agents are served in `pick_mode="sample"`
at `--rollout-temperature`. There is no per-category pick-mode flag.

**Rationale**: (a) FR-004 requires the learner's rollouts to be sampled at `T`;
(b) an anchor evaluated at argmax while the learner is sampled would put a fixed
sampling handicap on only one side of the margin — like-for-like sampling makes
`mean(learner) − mean(anchor)` a comparison of two policies under identical
conditions; (c) it matches `AgentRegistry.build`'s existing one-mode-for-all-services
signature. The anchor's absolute level is irrelevant — FR-021 only requires it not
to move, and it does not.

### D6 — Training states are rebuilt from the finished record, not captured at pick time

**Decision**: After a round's records arrive, walk each learner seat with
`iter_seat_pick_states` (the shared gen-1/gen-2 walk) to get
`(state, action_position)` per pick. The action is the card the booster geometry
records as taken — which for a model seat *is* the policy's own choice.

**Rationale**: reuses the walk that is already pinned by the gen-1 equivalence
test, keeps zero new state-reconstruction code, and needs no plumbing from the
pick service into the trainer. The recomputed `log π` matches the sampling-time
`log π` exactly for the round's **first** minibatch — the update runs immediately
after the round with no intervening weight change, and both forwards evaluate the
same eval-mode model at the same `T`. Later minibatches within the pass see
already-stepped weights; that within-pass drift is D8's accepted trade-off and is
exactly what the KL-to-previous-round diagnostic (D9) exists to size.

**Alternative rejected**: capturing states/logits inside `AgentPickService` and
threading them out. Correct but couples a shared inference module to the
trainer, and the two PACK orderings would have to be kept aligned by hand.

### D7 — Advantage: round-standardised pod-relative leave-one-out reward

**Decision**: `R_seat = deck_score_seat − mean(deck_score of the other non-failed
seats in its pod)` (the extracted `leave_one_out_rewards`), then
`A_seat = (R_seat − mean_round) / std_round` over the round's surviving learner
seats. One scalar shared by all of that seat's picks; detached. No critic, no
GAE, no discounting (γ=1, terminal reward).

**Guards** (FR-023): fewer than two surviving learner rewards, or
`std_round < 1e-8`, ⇒ the round is a **no-op** — no optimizer step — logged as
`skipped (no signal)`. The drafts are still recorded and still feed the anchor
window.

**Rationale**: the spec's contract verbatim; the leave-one-out term is the
RLOO/group baseline (pod = group) and the standardisation is a scale-only
rescale so `--lr` means the same thing across rounds and sets.

### D8 — "One pass" = minibatch SGD over the round's learner picks

**Decision**: One epoch over the round's learner picks, shuffled and
length-bucketed at `--batch-size` (default 32) — ≈56 optimizer steps for 10
drafts at a ~50 % learner mix — then the batch is discarded. Each step: forward,
`−(A · logπ_T(a))` mean over the batch's learner picks, backward, per-group
max-norm clip, `optimizer.step()`, `scheduler.step()`.

**Rationale**: the design doc states this explicitly ("take one pass over them
(~56 minibatch steps at batch 32)"). Batch 32 is the fixed-and-forget value that
fits the 8 GB VRAM budget.

**Alternative rejected**: accumulating the whole round into a single optimizer
step. It is the purest on-policy form, but rounds cost 1–3 minutes wall-clock
(generation + 8 greedy builds per draft), so a session fits a few hundred rounds
— i.e. a few hundred total gradient steps, far too few to move the policy. The
resulting mild within-pass off-policyness is what the KL-to-previous-round
diagnostic (D9) exists to surface, and `--lr` is the lever.

### D9 — All exploration + movement diagnostics from one post-update no-grad pass

**Decision**: Keep a second resident `DraftAgentModel` (`prev_model`). Before the
update, `prev_model.load_state_dict(model.state_dict())` — so `prev_model` **is**
πₖ, the policy that generated the round. After the update, run one batched
`no_grad` pass over the round's learner picks forwarding both models, and
accumulate:

| Quantity | From | Feeds |
|---|---|---|
| `H(πₖ)`, `exp(H)` | `prev_model` | exploration (FR-015) |
| off-argmax rate: `recorded action ≠ argmax(πₖ)` | `prev_model` | exploration (FR-015) |
| `mean log πₖ(a)` | `prev_model` | movement (FR-016) |
| `KL(πₖ ‖ πₖ₊₁)` per pick | both | movement (FR-016) |

Gradient norm comes free from the per-group clip's return value (pre-clip),
averaged over the round's steps; the policy-loss term is averaged over the steps.

**Rationale**: πₖ is exactly the sampling policy, so entropy/perplexity and the
off-argmax rate are *exact* — not blurred across a pass whose weights are moving.
The same pass yields the post-update KL, which no pre-update measurement can.
Cost is one forward-only pass over ~1.8 k picks with two models — small next to
generation, and fully batched (Principle VIII). Both models' PACK sets come from
the same reconstructed state, so the argmax comparison is well-defined.

**Alternative rejected**: instrumenting `AgentPickService` to accumulate stats at
pick time (also exact and cheaper) — it spreads training diagnostics into a
shared inference module, and cannot produce the post-update KL.

### D10 — Anchor margin: a sliding window of per-draft, per-label deck scores

**Decision**: A `deque(maxlen=--anchor-window)` of per-draft
`{label: [deck_score, …]}` maps, appended as each record arrives. Each round the
loop prints
`margin = mean(all learner scores in window) − mean(all anchor scores in window)`
plus the raw per-label means (learner, anchor, every Forge label) and the window's
draft count. Failed builds (`deck_score is None`) are excluded from every mean.
Best margin + its round index are tracked for the final summary (FR-019).

**Rationale**: pure arithmetic over scores already in hand (FR-017 forbids
requiring a pause or a second command). A ~100-draft window over a ~30 % anchor
share gives ~240 anchor decks, so the windowed mean is precise. The same figure
is recomputable post-hoc via `analyze-generated-decks --agent <each>` (FR-020).

### D11 — Checkpointing: `latest.pt` every round, snapshots on a cadence, no guard

**Decision**: `DraftAgentStore.save_checkpoint` to `models/draft/agent/latest.pt`
every round and to `models/draft/agent/{timestamp}.pt` every `--snapshot-every`
rounds (and once at run end/interrupt). `epoch` carries the round index;
`best_val_loss` is written as `inf` (no held-out metric exists — FR-026);
`critic_mean`/`critic_std` are carried through from the base checkpoint
unchanged; `train_config` is the resolved run config; `rl_metadata` is

```python
{"generation": base_generation + 1, "base_checkpoint": str(learner_path),
 "algorithm": "online-grpo", "lr": lr, "rollout_temperature": T,
 "drafts_per_round": n}
```

`generation` is read from the base checkpoint's `rl_metadata["generation"]`
(defaulting to 1 for a gen-1 base) + 1, exactly as the gen-2 trainer does.

**Rationale**: FR-026/FR-027/FR-028 verbatim; no store change is needed because
`rl_metadata` is already free-form and optional.

**Note (accepted, clarified 2026-08-05)**: the shared `latest.pt` means any tool
defaulting to it picks up the in-progress gen-3 mid-run.

### D12 — Seeding and reproducibility

**Decision**: `--seed` (default 42) seeds the torch/numpy init, the per-round
batch shuffling, and the pick-sampling RNG of every model service. Forge-side
randomness — booster contents, per-draft set choice, the Forge AI, the
`forge-r30`/`forge-r100` rolls, and the per-seat mix draw — is **not** seeded, so
a run is reproducible only up to the rollouts it is given. This is stated in the
startup echo and in quickstart.md rather than enforced.

**Rationale**: the closest achievable analogue of the codebase's hardcoded
`RANDOM_SEED = 42` convention. Claiming full reproducibility would be false
advertising for a loop whose data comes from an unseeded JVM.

### D13 — No `--resume`; the run is defined by `--learner`'s warm start

**Decision**: gen-3 has no `--resume` and no resume-precedence flag resolution
(`cli_resume` is not used). Continuing a run means pointing `--learner` at the
last checkpoint written and starting a new run — which restarts the LR warmup
and resets the optimizer moments.

**Rationale**: Simplicity First. FR-025 requires optimiser/LR continuity *across
rounds within a run*, which is in-memory and free; it says nothing about
continuity across process restarts, and gen-3 has no per-round training state
worth persisting (no epoch counter, no best-val, no plateau position). Adding
`--resume` would mean persisting and re-validating optimizer state for a loop
that has no early-stop semantics to resume into.

### D14 — One `ConvertedCardLocator` for the whole run; per-round embedding table

**Decision**: A single `ConvertedCardLocator` instance is shared by the deck
labeler, every pick service, and the trainer's loader, so each card's `.npz` is
decompressed at most once per run (the locator memoizes by name). The per-round
example table is built only over the cards appearing in *that* round (a few
hundred rows), from locator cache hits.

Sharing is not free today: `build_labeler` constructs its own locator internally
(`generate_draft_data.py:347`), as does `GenerateDraftDataSupervisor._build_registry`
(`:527`). So `build_labeler` takes an optional `locator` parameter —
`build_labeler(config, *, locator=None)`, falling back to constructing one when
absent so existing callers are unchanged — and the online trainer passes its own
instance to the labeler and to `AgentRegistry.build(locator=…)` (which already
accepts one). Without that parameter the "one locator" claim is not achievable
and every card is decompressed twice.

**Rationale**: satisfies Principle VIII's load-once/caching item without carrying
a monotonically growing table (and its per-round re-`stack`) across a run that is
meant to run for hours.

### D15 — Warmup is expressed in optimizer steps, not a fraction

**Decision**: `--warmup-steps` (default 200) — the LR ramps linearly over the
first N optimizer steps of the run, then stays constant. The scheduler is still
`train_draft_agent._make_scheduler`'s `LambdaLR`, constructed **once** at run
start with no plateau controller, and stepped once per minibatch step for the
whole run (never rebuilt per round).

**Rationale**: the sibling trainers' `--warmup-frac` is a fraction of a *known*
total step count (epochs × steps/epoch). An online run has no total — it runs
until the operator stops it — so a fraction is undefined. Steps is the only unit
that expresses "a single warmup at the start of the run, then constant" (FR-025)
without inventing a fake horizon. At ~56 steps per 10-draft round the default is
≈3.5 rounds of ramp.

**Alternative rejected**: deriving the horizon from `--max-rounds` — it is
optional, so the warmup length would silently change with an unrelated flag.

## Performance Review (Principle VIII)

- **I/O batching & caching** — one shared memoizing `ConvertedCardLocator` for
  labeler + pick services + trainer, which requires the `build_labeler(...,
  locator=None)` parameter (D14); per-round embedding table built from cache
  hits; the corpus handle is opened once (append mode) and appended per record.
  **Addressed.**
- **GPU placement** — learner model, `prev_model`, frozen pick-service models,
  scorer, and (when used) picker all move to CUDA when available; batches are
  collated onto the device (`_collate` pattern). **Addressed.**
- **GPU batching** — the round's pass is length-bucketed minibatches; the
  diagnostics pass is a single batched `no_grad` sweep (D9); the deck labeler
  already batches a whole pod through one picker/greedy + one scorer forward.
  Per-pick host↔device sync exists only in `AgentPickService._select` (one
  readout per pick), which is inherent to the strictly-synchronous pick protocol
  and unchanged from live-play. Round-level scalars are read once per round, not
  per step. **Addressed.**
- **Streaming & load-once** — drafts are consumed as a stream (never a
  materialised corpus); models/locator/labeler/worker are constructed once at
  startup and reused for the whole run. **Addressed.**
- No optimization beyond this checklist is proposed (Principle II); the obvious
  candidate — parallel Forge workers — is explicitly unnecessary here (drafts
  play no games; spec Assumptions).
