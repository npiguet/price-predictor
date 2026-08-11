# Research: Draft agent game-played evaluation

**Feature**: `022-draft-game-evaluation` | **Date**: 2026-08-09
**Spec**: [spec.md](spec.md) | **Root spec**: [../2026-08-09-draft-game-evaluation.md](../2026-08-09-draft-game-evaluation.md)

## Codebase Survey

### Overlapping domain vocabulary

| Existing concept | Where | Decision |
|---|---|---|
| `MatchResult` — one played match with run id, set, two method tags, two decks, `games`, `play`, duration | `forge-connector/.../MatchResult.java` | **Reuse unchanged.** It is exactly this feature's output record; the agent labels go in `methodA`/`methodB`. |
| `GamePlayer.PlayedMatch` / `GameOutcome` — per-game winner, play-first, duration | `forge-connector/.../GamePlayer.java` | **Reuse unchanged.** Already returns every field a match-outcome row needs. |
| `DraftRecord` / seat / booster geometry | `src/draft/infrastructure/draft_record_io.py`, `src/draft/domain/draft_geometry.py` | **Reuse.** The corpus reader and the record shape are the sampling input; no new draft-side entity. |
| `RoundRobinOutcome` / `RoundRobinResults` — win-rate aggregation | `src/sealed/domain/round_robin_results.py` | **Do not reuse.** It indexes an N×N grid positionally and reports one fixed pair of names. This feature tallies by label with `scripts/analyze_winrates.py` instead. |
| "method tag" (`forge-best`, `random`, a `build-decks --label`) | `specs/2026-03-28-sealed-deck-picker.md` § Phase 0 Step 4 | **Extend by convention, not by code.** An agent label is a new member of the same open set of tags; nothing enumerates them. |

No parallel concept is introduced, so no rename is proposed.

### Adjacent prior art

| Prior art | Where | Decision |
|---|---|---|
| `MatchResultWriter` — appends one 10-field row per call, documented as safe for concurrent workers | `forge-connector/.../MatchResultWriter.java` | **Reuse unchanged.** It already opens/writes/closes per row for exactly this concurrency. |
| `ValidationWorkerMain` + `ValidationMatchPlayer` — reads explicit `deckA;deckB` pairings, plays best-of-K | `forge-connector/.../Validation*.java` | **Do not extend; write a sibling.** See D1. |
| `MatchOutcomeSupervisor` — N workers, monitor threads, restart on exit, 60s status line, recycle oldest, signal handling | `src/sealed/application/match_outcomes.py` | **Extract and reuse.** See D5 and the third-instance check. |
| `EvaluationConnector` — spawns workers over shard files, polls, restarts | `src/sealed/infrastructure/evaluation_connector.py` | **Do not reuse.** Its poll loop assumes a bounded, pre-enumerated work list; this feature's workers are autonomous. |
| `GeneratedDecksIndex` — loads a Python-written flat deck file, indexes it, serves random decks with mirror exclusion | `forge-connector/.../GeneratedDecksIndex.java` | **Mirror directly.** It is the precedent for the seat table (D3): same file-in, sample-autonomously shape. |
| `DeckBuilder.buildStandard` — `new SealedDeckBuilder(pool).buildDeck()` | `forge-connector/.../DeckBuilder.java` | **Reuse unchanged.** Package-visible, and the new worker shares its package. |
| `DraftGeometry.drafted_pool` — reconstructs a seat's drafted pool from a record | `src/draft/domain/draft_geometry.py` | **Reuse unchanged.** Keeps the booster arithmetic in one language. |
| `build_jvm_command`, `build_forge_classpath`, `kill_process_tree` | `src/price_predictor/infrastructure/forge_jvm.py` | **Reuse unchanged.** Already the shared JVM entry point for every worker in the repo. |
| `scripts/analyze_winrates.py` — tallies match-outcome rows by method label | `scripts/` | **Reuse unchanged, and do not modify.** It is the whole reporting story (spec § 7). |
| `DraftWorkerConnector` | `src/draft/infrastructure/draft_worker_connector.py` | **Mirror its shape** for the new connector; it is the draft package's existing Forge bridge. |

### Convention alignment

The sibling to mirror is the sealed match-generation path, because this feature is the
same shape: a long-running Python supervisor over Forge JVM workers that append rows to a
delimited corpus.

- Use case in `src/draft/application/`, mirroring `src/sealed/application/match_outcomes.py`.
- Sampling logic in `src/draft/domain/`, importing nothing outside the domain layer.
- Worker bridge in `src/draft/infrastructure/`, mirroring `draft_worker_connector.py`.
- Java worker main beside its peers in `forge-connector/`, one `*Main` plus one testable
  player class, as `ValidationWorkerMain`/`ValidationMatchPlayer` are split.
- Dependency direction stays `draft → sealed → price_predictor`; nothing new points back.

No deviations.

### Third-instance check

Supervising a pool of Forge JVM workers, restarting them when they die, is implemented
**twice** today:

1. `src/sealed/application/match_outcomes.py` — monitor thread per worker, restart on exit,
   60-second status line, recycle the oldest worker, terminate all on SIGINT/SIGTERM.
2. `src/sealed/infrastructure/evaluation_connector.py` — spawn over shard files, poll for
   progress, restart crashed or stalled workers.

This feature would be the third. Principle VII therefore requires extracting the shared
abstraction rather than adding a third copy, and Principle II's three-use-case bar for a
new abstraction is met exactly.

**Proposal**: extract the supervisor loop from `match_outcomes.py` into
`src/price_predictor/infrastructure/forge_jvm.py`, which already hosts
`build_jvm_command`, `build_forge_classpath` and `kill_process_tree`, as a
`ForgeWorkerPool` taking a command factory, a worker count, a status-line callback and a
recycle interval. `match-outcomes` and this feature both consume it.

`EvaluationConnector` is **left alone** for now: its loop is bounded and its stall
detection has no counterpart here, so folding it in would be a speculative generalisation
(Principle II). This unifies two of the three instances and is recorded as a follow-up
task, not silently skipped.

## Decisions

### D1 — A sibling Java worker, not an extension of `ValidationWorkerMain`

**Decision**: add `DraftGameWorkerMain`, `SeatTableIndex` and `DraftGamePlayer` to
`forge-connector`, reusing `GamePlayer` and `MatchResultWriter` unchanged.

**Rationale**: `ValidationMatchPlayer` collapses a played match to `winsA;winsB`
(`ValidationMatchPlayer.appendOutcome`). That two-number format is consumed *positionally*
by `aggregate_results` in `src/sealed/domain/round_robin_results.py`, which recovers the
pairing from row order and requires a perfect-square row count. Changing what that worker
writes would break `evaluate-scorer`. The new player writes a full `MatchResult` instead,
which is a different output contract for a different consumer.

**Alternatives considered**:

- *Extend `ValidationMatchPlayer` with a mode flag* — rejected: two output formats behind a
  flag in one class, with a working consumer depending on the old one.
- *Have Python join worker outcomes back to pairings* — rejected: the outcome file carries
  no pairing identity and no `games`/`play`/duration, so Python would have to reconstruct
  what the JVM already knows. `GamePlayer` returns all of it.

### D2 — Workers write the output rows directly

**Decision**: each JVM appends its own `MatchResult` rows to the single output file via
`MatchResultWriter`. Python never parses a game result.

**Rationale**: `MatchResultWriter` opens, writes one line and closes per row precisely so
"concurrent workers can all append without corruption" (its own docstring), and this is
how `MatchWorkerMain` already feeds `match-outcomes.txt` from 12 JVMs. Reusing it means no
new serialisation, no per-worker merge step, and no risk of the Python side reformatting a
row the Java side already knows how to format.

**Alternatives considered**: per-worker files merged at the end — rejected: adds a merge
step and breaks the "rows are appended as each match completes" requirement (FR-011).

### D3 — A flat seat table is the entire Java/Python interface

**Decision**: Python projects the corpus once into a seat table —
`draft_id;set_code;label;kind;cards`, one seat per line — and each worker loads it and
samples from it independently. See [contracts/seat-table.md](contracts/seat-table.md).

**Rationale**: this is the smallest interface that keeps both the JSON parsing and the
booster geometry on the side that already has them. It has direct precedent:
`GeneratedDecksIndex` loads a Python-written flat deck file, indexes it, and serves random
decks to `MatchGenerator` with mirror exclusion — the same shape of problem. Because
diversion is applied when the table is written, the worker needs no native-fraction
parameter; a diverted seat simply arrives labelled `forge-native` with `kind = pool`.

**Alternatives considered**:

- *Workers read `drafts.jsonl` directly* — rejected on the geometry, not on JSON. Adding a
  JSON dependency to `forge-connector` is cheap and consistent with its pom, which already
  declares `minlog` and assembles a `jar-with-dependencies`; the module is not stdlib-only,
  and the CLAUDE.md line saying so needs correcting either way. The real cost is porting
  `draft_geometry.py`'s pack-direction and seat-of-pick arithmetic to Java, giving a second
  implementation whose divergence would silently build native decks from the wrong cards.
- *Per-pairing shard files written by the supervisor* — rejected: a live protocol rewritten
  throughout the run, plus a shard-refill loop, to buy control Python does not need.

### D4 — Workers are autonomous; the supervisor only counts

**Decision**: each worker loops — sample a pod, sample two seats, play, append — until it is
killed. The supervisor writes the table, spawns, restarts, recycles, and stops the run when
the output file has gained `--n-pairings` rows since it began.

**Rationale**: this makes the command structurally identical to `match-outcomes`, whose
`MatchWorkerMain` is likewise autonomous and whose supervisor likewise measures progress by
counting output lines. Since the spec already requires the progress and recycling behaviour
to match that command, sharing its shape as well as its code is the consistent choice.

**Consequence**: the supervisor never learns how many pairings were drawn or discarded, so
"skipped" is no longer a reportable quantity — see D6.

### D5 — Recycling and status reporting come from the extracted pool

**Decision**: the 60-second status line, the oldest-worker recycle, and the worker
lifecycle lines are the extracted `ForgeWorkerPool`'s behaviour, not reimplemented.

**Rationale**: the spec requires them to be identical to `match-outcomes`
(FR-023, FR-024). Sharing the code is the only way that stays true as either evolves.
Recycling is a liveness mechanism: Forge workers sometimes hang in near-infinite loops, and
because this feature *assigns* pairings, a hung worker would otherwise hold its pairing
forever instead of failing it.

### D6 — Matches played is counted; skipped is not reported

**Decision**: matches played is the number of rows the output file gained since the run
began. Nothing counts failures.

**Rationale**: `MatchOutcomeSupervisor._report_status` already counts output lines rather
than tracking an internal counter, so the status line's "matches completed" keeps its
existing meaning. With autonomous workers there is no assigned pairing to lose: a worker
whose match fails immediately draws another, so a failure costs time rather than data and
there is nothing to reconcile.

**Consequence**: the earlier "pairings drawn minus matches played" skip count is gone, and
with it a per-worker reporting channel that would otherwise have been needed. A run whose
workers are failing shows up as a flat match count and a low live-worker count on the status
line, which is what SC-006 now asks for.

### D7 — Draft games must not enter scorer training data

**Decision**: the new player does not write `cards-played.txt`, and the output file is
`output/draft/draft-games.txt`, never `output/sealed/match-outcomes.txt`.

**Rationale**: `GamePlayer` collects per-game card play through `PlayedCardCollector`, but
writing it is the caller's job via a separate `CardsPlayedWriter` — so not writing it is
simply not calling it. Keeping the corpora separate is stated in the spec's Assumptions:
`train-scorer` and `train-encoder` read the sealed files, and draft-agent matches are not
sealed-pool matches.

### D8 — `--best-of` needs no new plumbing

**Decision**: pass `-Dbest.of` as `EvaluationConnector._build_worker_command` already does.

**Rationale**: `ValidationWorkerMain` reads `-Dbest.of` and hands it to
`new GamePlayer(bestOf)`, which sets `GameRules.setGamesPerMatch`. The new main does the
same. Forge ends the match at a majority, so `games` has the length the format expects.

### D9 — Forge-native decks: Python reconstructs the pool, Java builds it

**Decision**: for a diverted seat, Python computes the drafted pool with
`DraftGeometry.drafted_pool` and writes it into the seat table with `kind = pool`. The worker
resolves the names and calls `DeckBuilder.buildStandard`, which is
`new SealedDeckBuilder(pool).buildDeck()` — Forge's own builder.

**Rationale**: both halves already exist and neither needs changing.
`drafted_pool(record, seat)` walks the booster geometry and returns the seat's full pool;
its docstring already describes rebuilding a pool for exactly this purpose.
`DeckBuilder.buildStandard` is package-visible, and the new worker lives in the same
package, so no visibility change is needed. Keeping reconstruction in Python means the JVM
still never reads `drafts.jsonl` and stays a pure "play what you are given" process.

**Alternatives considered**:

- *Reconstruct the pool in Java* — rejected: the worker would need the record, the booster
  geometry and the seat index, duplicating `draft_geometry.py` in a second language for no
  gain.
- *Use `DeckBuilder.buildDeck`* — rejected: it rolls a weighted method and would return
  `forge-3sub`, `forge-8sub` or `random` most of the time. `buildStandard` is the
  unconditional Forge-best path this feature wants.

**Consequence for the seat table**: a seat now offers either a finished deck or a pool, so
its line carries a `kind` marker. See [contracts/seat-table.md](contracts/seat-table.md).

### D10 — Diversion is decided once per seat, at corpus load

**Decision**: when the seat table is written, each `forge-full` seat is independently
diverted with probability `--forge-native-fraction`, and that choice is fixed for the run
because it is baked into the table the workers read.

**Rationale**: a seat then represents one deck for the whole run, so the two labels denote
two stable populations. Re-rolling per pairing would let the same seat appear under both
labels, which weakens the comparison for no benefit and makes a pairing's identity depend
on when it was drawn.

**Consequence**: `forge-full` and `forge-native` seats coexist in a pod, so they can be
drawn against each other. That is the sharpest form of the comparison — same pod, same
drafted cards on the reference side, two builders — and it falls out of the design rather
than needing a special case.

## Resolved unknowns

The two dependencies the root spec left open (§ 9 of its earlier draft) are now answered:

- **Can the caller read back which side played first?** Yes. `GamePlayer.GameOutcome`
  carries `playFirst` per game; only `ValidationMatchPlayer` throws it away. The spec's
  earlier requirement to *control* play-first was withdrawn, so reading it back is enough.
- **Can results be joined to their input pairing?** Not with the existing worker, and the
  new design removes the need: the worker writes the finished row itself (D2), so the join
  never happens in Python.
