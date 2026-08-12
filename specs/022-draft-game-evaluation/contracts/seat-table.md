# Contract: seat table

**Feature**: `022-draft-game-evaluation`

The whole interface between Python and the Forge workers. Python projects the corpus into
this file once at startup; each worker loads it and samples on its own thereafter. Nothing
is written to it while workers are running, and nothing outside this feature reads it.

**Location and lifetime**: a single file in the system temporary directory, created by the
supervisor before the workers start and deleted when the run ends, including on interrupt.
It is not an output of the feature and nothing may depend on it surviving.

This mirrors how `GeneratedDecksIndex` already feeds `MatchWorkerMain`: a flat,
Python-written file that the JVM indexes in memory and samples from autonomously.

## Worker invocation

```text
java -Dseats.file=<seat table> -Doutput.file=<output path> -Drun.id=<uuid>
     -Dbest.of=<N> -Dinclude.mirrors=<true|false> -Xmx1200m
     -cp <forge classpath>
     com.pricepredictor.connector.DraftGameWorkerMain
```

Built with `build_jvm_command` and `build_forge_classpath`, as every other worker is.
`-Xmx1200m` matches `EvaluationConnector._build_worker_command`.

Missing `seats.file`, `output.file` or `run.id`: message to stderr, exit 2. Absent or
unparseable `best.of`: warn and use 3, as `ValidationWorkerMain` does. Absent
`include.mirrors`: false.

There is no native-fraction property. Diversion is decided in Python when the table is
written, and reaches the worker as the seat's `label` and `kind`.

## File format

One seat per line, five semicolon-separated fields:

```text
draft_id;set_code;label;kind;cards
```

| Field | Meaning |
|---|---|
| `draft_id` | Groups seats into pods; opaque to the worker, compared only for equality |
| `set_code` | The set that pod drafted; becomes the row's `set_code` |
| `label` | Becomes `method_A` or `method_B`; already `forge-native` for a diverted seat |
| `kind` | `deck` or `pool` — how to read the cards field |
| `cards` | Pipe-separated Forge canonical card names, duplicates repeated |

`kind = deck` means a finished 40-card deck, played as given. `kind = pool` means a seat's
drafted pool **in pick order**, from which the worker builds a deck with Forge's own draft
builder before playing. Only diverted seats carry `pool`, and they always carry the label
`forge-native`. Pick order is load-bearing for a `pool` row: the worker replays it to recover
the colours the seat committed to while drafting.

Card names may contain spaces, commas and apostrophes; they never contain `;` or `|`.

Seats of one pod are written on consecutive lines, but a worker must group by `draft_id`
rather than rely on adjacency.

## Worker behaviour

Load the table once, group seats by `draft_id`, then loop until killed:

1. Choose a pod uniformly at random.
2. Choose two distinct seats of that pod uniformly at random.
3. If their labels are equal and `include.mirrors` is false, discard and return to step 1.
   A pod whose seats all share one label is skipped, so this terminates.
4. Resolve each side's cards through `FModel.getMagicDb()`. A `deck` side becomes that deck;
   a `pool` side is passed to `DeckBuilder.buildDrafted`, which replays the picks into a
   `DeckColors` and calls `new BoosterDeckBuilder(pool, colors).buildDeck()` — the builder and
   the colour source Forge itself uses for a drafted pool. That method is package-visible and
   this worker shares its package, so nothing needs widening.
5. Play one best-of-`best.of` match with `GamePlayer`.
6. Build a `MatchResult` — `timestamp` now, `runId` from `run.id`, `setCode` and the two
   labels from the seat lines, `games`/`play`/duration from the played match, and the two
   decks **as played**, so a diverted seat records Forge's built deck rather than its pool.
7. Append it to `output.file` with `MatchResultWriter`.

On an exception for one pairing: log to stderr and return to step 1. No row is written, and
nothing is lost — an autonomous worker simply draws another pairing.

The worker keeps no state across restarts and never terminates on its own.

## Concurrency

Every worker appends to the same `output.file` through `MatchResultWriter`, which opens,
writes one line and closes per row for exactly this reason. This is how `MatchWorkerMain`
already feeds `match-outcomes.txt` from 12 concurrent JVMs.

## Lifecycle

The supervisor owns worker lifetime and the stopping condition:

- It writes the seat table, then starts the workers.
- A worker that exits is restarted; the longest-running worker is recycled on the status
  interval, as in `match-outcomes`.
- It counts rows added to `output.file` since the run began. When that reaches
  `--n-pairings`, or when interrupted, it terminates the workers.
- Because workers draw their own pairings, the supervisor never learns how many were drawn
  or discarded. It knows only how many matches were recorded.
