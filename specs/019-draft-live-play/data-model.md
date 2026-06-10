# Data Model: Draft agent — live Forge integration

Phase 1 output. Entities below are either **reused unchanged** from gen-1 (018)
or **new** to this feature. No persisted schema changes: `drafts.jsonl` and the
agent checkpoint format are identical to gen-1 (FR-009, FR-015).

## 1. Reused entities (unchanged)

| Entity | Source | Role here |
|---|---|---|
| `DraftRecord` / `Seat` / `Booster` | `draft/domain/draft_geometry.py` | Output record per completed draft; `Seat.agent` now may carry a model label. |
| `DraftGeometry` | `draft/domain/draft_geometry.py` | Pod/pack/pick conventions shared by the online tracker. |
| `DraftState` / `CardInstance` | `draft/domain/draft_state.py` | Typed-token state the online tracker emits and the policy consumes. |
| `DraftAgentModel` / `DraftAgentConfig` | `draft/domain/draft_agent_model.py` | The frozen policy; `config.{embedding_dim,packs,P}` drive validation. |
| `LoadedDraftAgentCheckpoint` | `draft/infrastructure/draft_agent_store.py` | Loaded weights + config + standardization stats. |

## 2. New runtime entities (not persisted)

### 2.1 PickRequest

The worker→supervisor message for one external-seat pick (design §4.1). Parsed
from `<<DRAFT-PICK-REQUEST>>` JSON.

| Field | Type | Notes |
|---|---|---|
| `draft_id` | str (uuid) | Allocated by the worker at draft start; stable across the draft. |
| `seat` | int | 0-based pod seat. |
| `agent` | str | The seat's mix label (a model label). |
| `pod_size` | int | Live pod size (8). |
| `pack_number` | int | 1-based. |
| `pick_number` | int | 1-based. |
| `set_code` | str | Informational (logging only). |
| `pack` | list[str] | Card names remaining in the held pack, in pick (offset) order. |

**Validation.** `pod_size == 8`; `1 ≤ pack_number ≤ packs`; `1 ≤ pick_number`;
`pack` non-empty. A request whose `pick_number` exceeds the checkpoint's `P`, or
whose legal actions are entirely un-embeddable, is a **pick fault** (§3).

### 2.2 PickResponse

The supervisor→worker message (design §4.2), serialized to
`<<DRAFT-PICK-RESPONSE>>` JSON. Exactly one of `pick` / `abort` is meaningful.

| Field | Type | Notes |
|---|---|---|
| `draft_id` / `seat` / `pack_number` / `pick_number` | echo of the request | Worker validates these match the outstanding request. |
| `pick` | str \| absent | The chosen card; MUST be a name in the request's `pack`. |
| `abort` | bool (default false) | `true` ⇒ supervisor-initiated draft abandonment (no `pick`). |

### 2.3 AbandonedNotice

Worker→supervisor `<<DRAFT-ABANDONED>>{"draft_id":…,"reason":…}` emitted when the
worker self-detects a fault (response mismatch). Lets the supervisor log + count
the fault. EOF abandonment is instead observed as worker exit.

### 2.4 OnlineDraftStateTracker

Pure-domain object, one per `(draft_id, seat)` for the run's model seats.

State held: the seat's `pool` (own picks, ordered), `last_seen: name→(pack,pick)`,
`passed: name→(pack,pick)`, and the per-pack contents-as-last-seen needed for
wheel diffs. Method `observe(request) → DraftState`: advances the walk with the
new pack-in-hand and returns the typed-token state for `(pack_number,
pick_number)`. After the supervisor selects a card it is appended to `pool`
(via a follow-up `commit(card)` or folded into the next `observe`). Lifecycle:
created lazily on a seat's first request; discarded when the draft completes or
is abandoned.

**Invariant (gating, SC-003):** for every `(seat, pack, pick)`, the emitted
`DraftState` equals `draft_state.build_state(record, geometry, seat, pack, pick)`
for the finished record — same typed-token multiset, same per-instance
`(packs_ago, pick_ago)`, same `pack_actions`.

### 2.5 AgentPickService

Application service wrapping one loaded checkpoint. Holds the `DraftAgentModel`
(on device), the `ConvertedCardLocator`, the tracker registry for its label, the
pick mode/temperature, and the seeded RNG. `pick(request) → str`: tracker →
embed (drop un-embeddable) → forward → mask to PACK → argmax/sample → card name.
Raises a `PickFault` when no genuine pick is possible.

### 2.6 AgentRegistry

Maps each model label → its `AgentPickService`; validates labels (FR-011) and
checkpoint geometry (FR-012) at startup; exposes the set of external labels
forwarded to the worker. Built once.

## 3. Fault model (SC-002, FR-006/FR-007/FR-016)

A **pick fault** is any condition preventing a model seat's genuine pick:
Python-side policy/tracker error; a request whose legal actions are entirely
un-embeddable; a malformed/mismatched response; or stdin EOF / vanished peer.
Dropping *individual* un-embeddable cards from the action set is **normal**, not
a fault (matches offline training).

On a pick fault the in-flight draft is **abandoned**: no record is written, the
error is logged prominently, and the run continues toward `--n-drafts` without
counting it. No Forge/first-card/random substitute is ever recorded. The
**consecutive** fault counter increments per abandoned draft and **resets on any
completed draft**; reaching `--max-consecutive-faults` (default 5) aborts the run
with a nonzero exit (FR-016). A recovered worker JVM crash is *not* a pick fault
and does not increment the counter.

## 4. Configuration entity additions

`GenerateDraftDataConfig` gains (all optional; defaults preserve gen-1 behavior):

| Field | Type | Default | Requirement |
|---|---|---|---|
| `agent_checkpoints` | dict[str, Path] | `{}` | FR-001, FR-010 |
| `pick_mode` | `"argmax"` \| `"sample"` | `"argmax"` | FR-005 |
| `temperature` | float | `1.0` | FR-005 |
| `seed` | int \| None | `None` | FR-005, SC-007 |
| `max_consecutive_faults` | int | `5` | FR-016 |

When `agent_checkpoints` is empty the supervisor builds no registry, the worker
receives an empty external-agent set, and the path is identical to gen-1 (SC-004).
