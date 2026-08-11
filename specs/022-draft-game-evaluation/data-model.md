# Data Model: Draft agent game-played evaluation

**Feature**: `022-draft-game-evaluation` | **Date**: 2026-08-09

Three of the four entities already exist. Only `Pairing` is new, and it is transient — it
never reaches disk in its own right.

## Entities

### DraftRecord (existing, read-only)

One pod's draft. Read from `output/draft/drafts.jsonl`; schema fixed by
[`2026-05-28-draft-agent.md`](../2026-05-28-draft-agent.md) § 4 and parsed by
`src/draft/infrastructure/draft_record_io.py`.

| Field used here | Type | Use |
|---|---|---|
| `draft_id` | str | Identifies the record; not written to output |
| `run_id` | str | Filtered on by `--run-id` |
| `seats` | list | The population a pairing is drawn from |
| `boosters[].set_code` | str | All equal within a record; becomes the row's `set_code` |
| `boosters[].picks` | list | Read only for diverted seats, to reconstruct the drafted pool |

Not read: `timestamp`. Draft state is never reconstructed; the pick lists are walked only
by `DraftGeometry.drafted_pool`, and only when the native fraction diverts a seat.

**Validation**: parse-level only — the record must parse, and a trailing partial line is
ignored. No semantic record validation, per the root spec.

### Seat (existing, read-only)

One player's position within a record.

| Field used here | Type | Use |
|---|---|---|
| `agent` | str | Becomes `method_A` or `method_B`, unless the seat is diverted |
| `deck` | list[str] | 40 Forge canonical card names including basics; unused for a diverted seat |

Not read: `deck_score`. The scores were dropped from the output when § 6.2 was removed from
the root spec, so nothing in this feature reads them.

**Diversion.** A seat labelled `forge-full` is diverted with probability
`--forge-native-fraction`, decided once at corpus load and fixed for the run. A diverted
seat contributes its drafted pool — from `DraftGeometry.drafted_pool` — instead of its
recorded deck, and reports under the label `forge-native`. Every other seat is undiverted.
Both variants can occupy the same pod, so they can be drawn against each other.

### Pairing (new, transient)

Two distinct seats of one record, selected to play. It exists only inside a worker, between
the draw and the row it produces, and is never serialised.

| Field | Type | Notes |
|---|---|---|
| `set_code` | str | From the record |
| `label_a`, `label_b` | str | The two seats' labels, `forge-native` where diverted |
| `kind_a`, `kind_b` | `deck` \| `pool` | How the cards beside each label are to be read |
| `cards_a`, `cards_b` | list[str] | A recorded deck, or a drafted pool for a diverted seat |

**Invariants**:

- Both seats come from the same record.
- The two seats are distinct.
- `label_a != label_b` unless `--include-mirrors` is given.

**Lifecycle**: drawn by a worker → played → becomes a `MatchResult` row, or is abandoned on
failure and replaced by another draw. A pairing may be drawn more than once in a run, by the
same worker or by different ones; draws are independent.

### MatchResult (existing, output)

One completed match. Modelled by `forge-connector/.../MatchResult.java` and serialised by
`MatchResultWriter` in the sealed match-outcome format, specified in
[`2026-03-28-sealed-deck-picker.md`](../2026-03-28-sealed-deck-picker.md) § Phase 0 Step 4.

Bindings this feature chooses:

| Field | Value |
|---|---|
| `runId` | UUID of the `play-draft-games` invocation |
| `setCode` | The pairing's set |
| `methodA`, `methodB` | The two seats' labels, `forge-native` where diverted |
| `deckA`, `deckB` | The decks as played — Forge's built deck for a diverted seat, not its pool |

Every other field takes its documented meaning. `MatchResult`'s constructor validates that
`games` and `play` are equal-length and drawn from `{A, B}`, and that the duration is
non-negative — the only validation on the write path, and it already exists.

## Relationships

```text
DraftRecord 1 ── * Seat
DraftRecord 1 ── * Pairing        (both seats from this record)
Pairing     1 ── 0..1 MatchResult (0 when the match is skipped)
```

The `Pairing → MatchResult` arrow is where identity is lost: the output row records the two
agent labels and the two decks, but not which record or seats they came from. That is
accepted — the root spec dropped the richer record — and it is why a tally can group by
label but not cluster by pod.

## State transitions

A pairing has no persisted state. A run has one counter:

```text
played = rows the output file gained since the run began
```

measured by counting output lines, as `MatchOutcomeSupervisor._report_status` already does,
rather than by workers reporting back. It drives the status line, the stopping condition for
`--n-pairings`, and the exit summary.

There is no skipped counter. Workers draw their own pairings, so a match lost to a Forge
exception or to a mid-match recycle was never reserved: the worker draws another and the
loss costs time, not data. A run whose workers are failing therefore shows as a flat match
count and a low live-worker count on the status line rather than as a skip total.
