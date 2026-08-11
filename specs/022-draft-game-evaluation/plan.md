# Implementation Plan: Draft agent game-played evaluation

**Branch**: `022-draft-game-evaluation` | **Date**: 2026-08-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/022-draft-game-evaluation/spec.md`

## Summary

`play-draft-games` samples pairs of decks drafted in the same pod, plays a best-of-N Forge
match between each pair, and appends the results to a file in the sealed match-outcome
format so `scripts/analyze_winrates.py` reports per-agent win rates over them unchanged.

The technical approach is almost entirely reuse. `GamePlayer` already returns per-game
winner, play-first and duration; `MatchResult` already models a match-outcome row;
`MatchResultWriter` already appends those rows safely from concurrent JVMs. The only Java
addition is a worker main that joins them — `ValidationMatchPlayer` discards all of it into
`winsA;winsB` and cannot be changed without breaking `evaluate-scorer`. On the Python side
the work is the corpus projection, a worker supervisor, and a CLI subcommand.

Python's whole contribution to a running match is one file. It projects the corpus once into
a flat seat table — one line per seat, carrying the pod id, set, label, and either a deck or
a drafted pool — and each worker then loads it and samples pods and pairs on its own, exactly
as `GeneratedDecksIndex` already serves random decks to `MatchWorkerMain`. The supervisor
spawns, restarts, recycles, and counts output rows; it issues no work.

`--forge-native-fraction` diverts a share of `forge-full` seats to Forge's own sealed deck
builder, working from the seat's drafted pool and reporting under the label `forge-native`.
Both halves exist already: `DraftGeometry.drafted_pool` reconstructs the pool in Python, and
`DeckBuilder.buildStandard` wraps Forge's `SealedDeckBuilder` in the same Java package as
the new worker. Diversion is applied when the table is written, so it reaches the worker as
nothing more than a seat's label and `kind`.

## Technical Context

**Language/Version**: Python 3.14 (`src/draft`), Java 17 (`forge-connector`)
**Primary Dependencies**: MTG Forge (`forge-game`, `forge-core`) via the sibling `../forge`
checkout; no new third-party dependency on either side
**Storage**: append-only delimited text — `output/draft/draft-games.txt` (output),
`output/draft/drafts.jsonl` (input), one seat table in the system temporary directory,
deleted when the run ends
**Testing**: pytest for Python, JUnit for `forge-connector`; the projection, the seat-table
line format, and the worker's pod/pair sampling are pure and belong in the fast suites
**Target Platform**: Windows 11 desktop, CPU-bound Forge JVMs
**Project Type**: CLI subcommand plus a Java worker main
**Performance Goals**: none set. Throughput is Forge's, not ours; 12 concurrent JVMs is the
default carried over from `match-outcomes`
**Constraints**: each worker JVM runs with `-Xmx1200m` as the existing validation worker
does, so 12 workers must fit alongside the desktop's other memory use. No GPU is involved
**Scale/Scope**: the yardstick corpus is a few hundred pods of 8 seats, offering candidate
pairs in the low thousands; it is append-only and grows between runs, so nothing in the
design may assume a fixed size. An unbounded run is expected to be interrupted after hours

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Note |
|---|---|---|
| I. Fast automated tests | PASS | The corpus projection and seat-table formatting are unit-tested in Python; pod grouping, pair sampling and mirror rejection in Java, where they now live. Anything booting Forge is excluded from the fast suites, matching how the sealed worker tests are split. |
| II. Simplicity first | PASS | One new Java main, one use case, one connector, one CLI subcommand. The single new abstraction (`ForgeWorkerPool`) is an extraction with three existing call sites, not a speculative one. |
| III. Data integrity | PASS | Rows are written by the existing `MatchResult`, whose constructor validates `games`/`play` length and alphabet. The corpus is read-only. The output file is new, so no schema migration arises. |
| IV. DDD & separation | PASS | Sampling in `draft/domain` (no I/O), orchestration in `draft/application`, JVM bridge in `draft/infrastructure`. Dependencies point inward; `draft → sealed → price_predictor` is unchanged. |
| V. Forge interoperability | N/A | This feature adds a CLI worker, not the `PricePredictorClient` stub. It changes nothing Forge consumes. |
| VI. Documentation | PASS, with obligations | `src/draft/CLAUDE.md` gains the subcommand; the root `CLAUDE.md` corpus-format list gains `draft-games.txt`; the README gains the workflow. Tracked as tasks, in the same change. |
| VII. Codebase-aware planning | PASS | Survey below. |
| VIII. Performance-conscious | PASS | Review below. |

### Codebase Survey (Principle VII — required)

Recorded in [research.md § Codebase Survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 4 concepts reused (`MatchResult`, `GamePlayer.PlayedMatch`,
  the draft record reader, the method-tag convention), 1 explicitly not reused
  (`RoundRobinResults`), 0 parallel concepts introduced, 0 renames proposed.
- **Adjacent prior art**: 5 reused unchanged, 1 mirrored for shape, 2 rejected with reasons
  recorded (`ValidationMatchPlayer`, `EvaluationConnector`).
- **Convention alignment**: mirrors the sealed match-generation path; 0 deviations.
- **Third-instance check**: FAILED as written, resolved by extraction. Supervising Forge
  workers exists twice (`match_outcomes.py`, `evaluation_connector.py`); this would be the
  third. `ForgeWorkerPool` is extracted into
  `src/price_predictor/infrastructure/forge_jvm.py` and shared with `match-outcomes`.

Follow-up tasks this survey surfaced, to appear in `tasks.md`:

1. Extract `ForgeWorkerPool` and migrate `MatchOutcomeSupervisor` onto it, with the sealed
   command's behaviour unchanged.
2. Record in `research.md` — done — that `EvaluationConnector` stays a separate
   implementation, so a future reader does not mistake it for an oversight.
3. Correct the `_kill_oldest_worker` docstring in `src/sealed/application/match_outcomes.py`,
   which attributes recycling to JVM slowdown rather than to workers hanging in
   near-infinite loops.

### Performance Review (Principle VIII — required when applicable)

The feature moves data (a JSONL corpus in, a delimited corpus out) and runs no model
compute.

- **I/O batching & caching**: addressed. The corpus is read once, at startup, to write the
  seat table; it is never reopened. Each worker loads that table once and samples from
  memory thereafter, so no pairing costs a read. The output file is appended one row per
  match by the JVM that played it.
- **GPU placement**: N/A — no model is loaded. Deck scores are not read and the scorer is
  not invoked.
- **GPU batching**: N/A — same reason.
- **Streaming & load-once**: addressed. The projection streams: each record is parsed,
  emitted as seat lines, and dropped, so Python never holds the corpus. Each worker does
  hold the whole seat table, and there are `--workers` of them — at a few thousand seats of
  40-odd card names that is single-digit megabytes per JVM, well inside the existing
  `-Xmx1200m`. The status line's match count is read by counting output lines, as
  `match-outcomes` does, which is a per-interval scan rather than a per-match one.

No optimization beyond this checklist is planned; none has been measured as needed.

## Project Structure

### Documentation (this feature)

```text
specs/022-draft-game-evaluation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── cli.md           # play-draft-games CLI contract
│   └── seat-table.md    # the Python → worker seat file, and worker behaviour
├── checklists/
│   └── requirements.md  # from /speckit.specify
└── tasks.md             # /speckit.tasks output — NOT created here
```

### Source Code (repository root)

```text
src/draft/
├── domain/
│   ├── seat_table.py               # NEW — corpus → seat rows, diversion, line formatting
│   └── draft_geometry.py           # REUSED — drafted_pool() reconstructs a diverted seat's pool
├── application/
│   └── play_draft_games.py         # NEW — project the table, supervise workers, summarise
└── infrastructure/
    ├── cli.py                      # EDIT — add the play-draft-games subcommand
    └── draft_game_connector.py     # NEW — builds the worker command

src/price_predictor/infrastructure/
└── forge_jvm.py                    # EDIT — extract ForgeWorkerPool here

src/sealed/application/
└── match_outcomes.py               # EDIT — migrate onto ForgeWorkerPool; fix the docstring

forge-connector/src/main/java/com/pricepredictor/connector/
├── DraftGameWorkerMain.java        # NEW — property parsing, loop until killed
├── SeatTableIndex.java             # NEW — load and group by pod, sample a non-mirror pair
└── DraftGamePlayer.java            # NEW — testable body: build sides, play, write a row

tests/unit/draft/
├── domain/test_seat_table.py       # NEW — diversion rate, label override, line formatting
└── application/test_play_draft_games.py  # NEW — stopping condition, summary

forge-connector/src/test/java/com/pricepredictor/connector/
├── SeatTableIndexTest.java         # NEW — parsing, pod grouping, mirror rejection
└── DraftGamePlayerTest.java        # NEW — deck vs pool sides, row construction
```

**Structure Decision**: the existing three-layer `src/draft` package, extended in place.
Sampling is pure domain logic, the supervisor is an application use case, and the JVM
bridge is infrastructure — the same split `generate_draft_data.py` and
`draft_worker_connector.py` already use. The Java side follows the
`ValidationWorkerMain`/`ValidationMatchPlayer` split so the logic is unit-testable without
booting Forge.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| New abstraction `ForgeWorkerPool`, touching a working sealed command | Principle VII's third-instance rule: worker supervision already exists twice, and this feature is the third | Copying the supervisor a third time was the simpler local change, but leaves three copies of the recycle/restart/status logic that the spec requires to stay identical |
| A second Forge worker main alongside `ValidationWorkerMain` | Its output format is consumed positionally by `evaluate-scorer` via `aggregate_results` | Extending the existing worker behind a mode flag would put two output contracts in one class and risk a working evaluation path |
