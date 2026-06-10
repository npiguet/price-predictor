# Research: Draft agent — live Forge integration

Phase 0 output for `specs/019-draft-live-play/`. Resolves the design decisions
the spec leaves open and grounds them in the existing codebase (Principle VII).
The normative authority on the external protocol is
[`specs/2026-06-04-draft-agent-live-play.md`](../2026-06-04-draft-agent-live-play.md);
this file records *how* that maps onto the current code.

## Codebase Survey

### Overlapping domain vocabulary

| Existing concept | Where | Decision |
|---|---|---|
| `DraftAgentModel` / `DraftAgentConfig` | `draft/domain/draft_agent_model.py` | **Reuse** — load the frozen policy and run its `forward`; only the policy head is consulted (critic ignored per scope). |
| `DraftAgentStore` (`config` carries `embedding_dim`, `packs`, `P`, `critic_mean/std`) | `draft/infrastructure/draft_agent_store.py` | **Reuse** — load checkpoints; `config.packs`/`config.P`/`config.embedding_dim` drive the geometry/width validation (FR-012). |
| `DraftState` / `CardInstance` / `_recency` / `TYPE_*` | `draft/domain/draft_state.py` | **Reuse** — the online tracker emits the same `DraftState`/`CardInstance` types and shares `_recency`; the inference glue tensorizes exactly these. |
| `DraftGeometry` (pod/pack/pick maps, `direction`, wheel offset) | `draft/domain/draft_geometry.py` | **Reuse** — the online tracker uses the same conventions (pod_size, `direction`, wheel = offset ≥ pod) so its output matches `build_state`. |
| `parse_agent_mix` / `format_agent_mix` / `sample_agents` | `draft/application/agent_mix.py` | **Reuse** — mix grammar is unchanged; model labels are just mix labels that also appear in `--agent-checkpoint`. |
| `ConvertedCardLocator.load_embedding` (per-name memo) | `sealed/infrastructure/converted_card_locator.py` | **Reuse** — single source of `.npz` embeddings; un-embeddable card ⇒ `None` ⇒ dropped (matches training). |
| gen-1 supervisor (`run_id`, crash-restart, status loop, JSONL append) | `draft/application/generate_draft_data.py` | **Reuse/extend** — the existing read loop is extended to also answer pick-requests; labeling/scoring/append untouched. |

**New concepts introduced (justified):**

- **`OnlineDraftStateTracker`** (`draft/domain/online_draft_state.py`) — rebuilds
  a seat's typed-token `DraftState` *incrementally from the pick-request stream*
  (each request = the pack currently in hand at `(pack_number, pick_number)`),
  remembering the seat's own prior responses and the contents of every pack it
  has already seen. No existing concept consumes a partial request stream:
  `build_state` and the loader walk both require a *complete* `DraftRecord`. The
  tracker reuses `CardInstance`/`DraftState`/`_recency` and is **locked to
  `build_state` by a gating equivalence test** (SC-003), so it is a thin sibling,
  not a silent duplicate. No rename of the older concepts is warranted (they keep
  the full-record contract).
- **`AgentPickService`** (`draft/application/agent_pick_service.py`) — owns one
  loaded checkpoint + its tracker registry and turns a pick-request into a chosen
  card name (argmax or seeded temperature sample). The offline counterpart
  `sealed`'s `pick_decks` / the gen-1 labelers do *batched* deck inference from a
  finished pool; none do per-pick live selection, so this is genuinely new
  (adjacent prior art, not a duplicate).

### Adjacent prior art

| Sub-problem | Prior art | Decision |
|---|---|---|
| Spawn/restart a Forge worker, run-id, status/ETA, SIGINT | `draft/application/generate_draft_data.py`, `sealed/application/match_outcomes.py` | **Reuse** — extend the loop, keep restart/append semantics. |
| Launch `DraftWorkerMain` with JVM props | `draft/infrastructure/draft_worker_connector.py` (`build_forge_classpath`, `build_jvm_command`) | **Wrap/extend** — add `stdin=PIPE`, a stderr log file, and `-Ddraft.external.agents=…`. |
| Deck build + score per seat (batched, GPU) | gen-1 `_PickerLabeler` / `_GreedyLabeler` + `score_decks` | **Reuse unchanged** — labeling is identical regardless of who piloted a seat. |
| JSONL append + `--resume` count + partial-line tolerance | `draft/infrastructure/draft_record_io.py` | **Reuse unchanged.** |
| `<<DRAFT-EVENT-JSON>>` transcript parse | `generate_draft_data.parse_sentinel_line` | **Reuse** — add sibling parsers for the two new sentinels. |
| Weighted per-seat agent sampling (Java) | `DraftWorkerMain.AgentMix` + `decidePick` | **Reuse/extend** — `decidePick` branches to the request path for external labels. |
| UTF-8 stdout sentinel discipline (Forge chatter → stderr) | `DraftWorkerMain.main` | **Reuse** — the stdin reader follows the same single-FD discipline. |
| Tiny single-example tensorization | gen-1 `_collate` (batched) | **Reimplement minimally** — a one-example builder reusing the same field layout; documented reason: `_collate` is keyed on a shared table + standardization for training batches, heavier than a per-pick path needs. |

### Convention alignment

Mirrors `src/draft/` (itself a sibling of `src/sealed/`): pure logic in `domain`
(no torch/IO), orchestration in `application`, argparse + connectors + stores in
`infrastructure`; lazy torch imports inside `run_*`; one-way dependency on
`sealed`/`price_predictor`. The Java change stays inside the existing
`DraftWorkerMain` alongside the other forge-connector mains. No deviation.

### Third-instance check

Typed-token reconstruction now exists in three places: `build_state` (full-record
oracle), `_Loader._emit_seat` (full-record incremental, vectorized for training
throughput), and the new `OnlineDraftStateTracker` (request-stream incremental).
**Decision: do not force a shared-core extraction in this feature.** The three
differ on input contract (complete record vs. partial stream) and on hot-path
shape (the training walk is numpy-vectorized for millions of examples; the live
tracker handles one pick at a time). Extracting now would couple a latency-
insensitive live path to a throughput-tuned training path. Instead: the tracker
*reuses* the shared value types and the `_recency` rule, and the gating test pins
it to `build_state`. **Non-blocking follow-up** (for `tasks.md`): once the live
tracker exists, re-evaluate factoring a common `wheel-diff / pack-flush /
recency` kernel used by all three — alongside the gen-1 `train_common` follow-up.

## Technical decisions

### D1 — Pick protocol: two new sentinel lines + an abort response

**Decision.** Add to the worker↔supervisor pipe (UTF-8, one flushed line each):

- `<<DRAFT-PICK-REQUEST>>{…}` (worker stdout) — pack-in-hand + routing scalars
  (design §4.1).
- `<<DRAFT-PICK-RESPONSE>>{…}` (worker stdin) — either `"pick":"<card>"` **or**
  `"abort":true`. The abort form is how the **supervisor** tells the worker to
  drop the in-flight draft when a Python-side fault prevents a genuine pick
  (policy error / all-un-embeddable actions). Strict synchrony holds: every
  request gets exactly one response, pick-or-abort.

**Rationale.** The design note (§4.3) requires supervisor-detected faults to
abandon the draft, but the only documented supervisor→worker line is the
response; an `abort` field keeps the protocol to a single response per request
(no extra message type, no out-of-band channel) and preserves the "≤ 1
outstanding request" invariant the single-threaded reader relies on.
**Alternative rejected:** kill+restart the worker on every Python-side fault —
heavier (loses the live JVM and its Forge init) and conflates the deterministic-
fault case with genuine crashes.

**Worker-detected faults** (mismatched/garbled `<<DRAFT-PICK-RESPONSE>>`, stdin
EOF/vanished peer): the worker abandons the current draft (emits no transcript)
and, where it still can, loops to the next draft; on EOF it exits and the
supervisor restarts it (the existing crash path). The worker emits a
`<<DRAFT-ABANDONED>>{"draft_id":…,"reason":…}` notice before continuing so the
supervisor can log + count the fault; an EOF abandonment is observed by the
supervisor as worker exit.

### D2 — `draft_id` allocated at draft start (Java)

**Decision.** Move `UUID.randomUUID()` from transcript-emit time to the top of
`generateDraft`, thread it through the pick loop, and reuse it in the transcript.
**Rationale.** Pick-requests must carry the same `draft_id` the transcript later
reports (design §4.1) so the supervisor can key trackers per draft and validate
responses. No behavioral change to Forge-only drafts.

### D3 — Online state equivalence is the load-bearing correctness property

**Decision.** The tracker maintains, per `(draft_id, seat)`: the seat's `pool`
(its own prior responses, in order), a `last_seen` map, a `passed` map, and a
record of each pack's contents as last seen, exactly mirroring the offline walk.
Wheel resolution uses the same trigger as `build_state` — at pick `i` of pack
`p`, offset `= i−1`; when `offset ≥ pod_size` the same physical booster has
wheeled back, and the cards it held `pod_size` picks ago that are gone now were
taken by others (FR-019a). Pack-end flush and `_recency` follow `build_state`
verbatim.

**Why it reconstructs from requests alone.** A booster returns to a seat exactly
`pod_size` picks later; the request at `(p, i−pod_size)` already showed the seat
that booster's then-contents, and the seat's own response removed one card — so
"contents `pod_size` ago" minus "contents now" minus "what I took" = taken by
others, identical to the offline `boosters[k].picks[off−pod+1:off]` diff. The
request stream is therefore sufficient; the pick-request stays minimal (design
§4.1).

**Verification.** Gating test (SC-003): take a finished synthetic `DraftRecord`,
replay each model-seat's pick-requests through the tracker, and assert the
emitted `DraftState` (typed-token multiset + `(packs_ago, pick_ago)` per
instance + `pack_actions` + `target` once revealed) equals `build_state` at every
`(seat, pack, pick)`. This is the single new correctness requirement (design §3).

### D4 — Inference: embed `DraftState`, mask to PACK, argmax/sample

**Decision.** For a request: build the `DraftState` via the tracker, drop
un-embeddable cards from every token group (matching training), and if **no**
PACK card is embeddable raise a pick fault (design §4.3 / spec edge case). Build
one-example tensors (`card_emb`, `type_idx`, `packs_ago`, `pick_ago`,
`card_mask`, `pack_number`, `pick_number`) on the model device, forward once,
mask logits to PACK positions, then:

- `--pick-mode argmax`: take the highest-logit PACK card.
- `--pick-mode sample`: sample from `softmax(logits / temperature)` using a
  per-service `random`/torch generator seeded from `--seed` (FR-005, SC-007).

Map the chosen PACK index back to its card name (a member of the held pack) and
return it. **Reproducibility:** with `--seed`, the supervisor seeds one RNG and
all sampled picks become deterministic given identical inputs; Forge-side
randomness is not seeded here (design §5).

### D5 — Geometry & label validation, fail fast (FR-011, FR-012)

**Decision.** At startup: (a) every mix label must be a Forge built-in
(`forge-full`/`forge-r30`/`forge-r100`) **or** bound via `--agent-checkpoint`,
else exit nonzero with a clear message (FR-011); (b) each bound checkpoint's
`config.packs` must equal the live `PACKS` (3) and its `config.P` must be ≥ the
live pack size; the pod size is fixed at 8 on both sides. With `--set` the live
pack size is known at startup and checked then; for random-set runs (`P` is the
max training pack size, ≥ all sealed-legal sets) the per-request `pick_number ≤
P` guard catches any overflow as a pick fault. **Rationale.** Mirrors the
fail-fast width checks already in `train_draft_agent._check_dims`; converts a
silent malformed-pick risk into a startup error.

### D6 — CLI surface (extend, don't add a subcommand)

**Decision.** Add to `generate-draft-data`: `--agent-checkpoint LABEL=PATH`
(repeatable; bare `PATH` ⇒ label `draft-agent`), `--pick-mode {argmax,sample}`
(default `argmax`), `--temperature` (default 1.0), `--seed`, and
`--max-consecutive-faults` (default 5, FR-016). With no `--agent-checkpoint` the
command path is byte-for-byte the gen-1 behavior (SC-004) — the worker gets an
empty external-agent set and never emits a pick-request. **Rationale.** Design §5
mandates an extension, not a new subcommand; the gen-1 flags keep their meaning.

### D7 — Single-threaded supervisor, no worker threads

**Decision.** Keep the supervisor's single read loop over worker stdout. Each
line is one of: pick-request (compute + write response/abort), event-json
(label + append, reset trackers, reset fault counter), abandoned (log + count
fault), or noise (skip). No threading is needed because of strict synchrony
(design §4). The K-consecutive fault counter increments on each
abandoned/abort-fault draft and resets on any completed draft; reaching the
threshold raises a fatal error → nonzero exit (FR-016). **Rationale.** Matches
the design's "single-threaded read loop sees a clean stream" and avoids
concurrency the invariant makes unnecessary.

### D8 — stderr to a log file (FR-015)

**Decision.** The connector pipes worker stderr to a per-run log file under the
output directory (e.g. `output/draft/worker-<run_id>.log`) instead of
`DEVNULL`, so the "diagnostic log" of FR-015 exists. **Rationale.** FR-013/FR-015
call for an observable diagnostic log; today the connector discards stderr.

## Open items

None blocking. The two follow-ups (shared reconstruction-core extraction; gen-1
`train_common`) are recorded as non-blocking `tasks.md` entries, not gates.
