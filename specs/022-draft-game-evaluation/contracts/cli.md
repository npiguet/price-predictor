# Contract: `play-draft-games` CLI

**Feature**: `022-draft-game-evaluation`

## Invocation

```text
python -m draft play-draft-games
    [--drafts-path PATH] [--run-id ID]...
    [--output-path PATH]
    [--n-pairings N]
    [--best-of N]
    [--include-mirrors]
    [--forge-native-fraction F]
    [--workers N]
```

## Options

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--drafts-path` | path | `output/draft/drafts.jsonl` | Corpus to sample from |
| `--run-id` | str, repeatable | none | Restrict to records with a matching `run_id`; absent means the whole corpus |
| `--output-path` | path | `output/draft/draft-games.txt` | Append target, sealed match-outcome format |
| `--n-pairings` | int > 0 | none | How many matches this run records before stopping; absent means play until interrupted |
| `--best-of` | odd int > 0 | 3 | Games per match |
| `--include-mirrors` | flag | off | Keep pairings whose two seats share an agent label |
| `--forge-native-fraction` | float in [0, 1] | 0 | Share of `forge-full` seats whose deck Forge rebuilds from their drafted pool; those seats are labelled `forge-native` |
| `--workers` | int > 0 | 12 | Concurrent Forge worker processes |

There is no seed option. Pair sampling happens inside the workers, each of which seeds
itself, and a run is not reproducible.

## Startup validation

Runs before any game is played. On failure: a message naming the offending option or file,
and exit 2.

- `--drafts-path` exists and parses.
- At least one pairing survives the `--run-id` and mirror filters.
- `--best-of` is a positive odd integer.
- `--n-pairings`, if given, is a positive integer.
- `--workers` is a positive integer.
- `--forge-native-fraction` is between 0 and 1 inclusive.

## Startup echo

Emitted after validation, before the first worker starts:

- the corpus path and the number of records in scope;
- the `run_id`s selected, or that the whole corpus is in scope;
- the pairing target, or that the run is unbounded;
- `--best-of`, `--workers`, and whether mirrors are included;
- when the native fraction is non-zero, how many `forge-full` seats were diverted to
  `forge-native`, out of how many;
- the output path.

## Progress reporting

Identical in shape to the sealed `match-outcomes` supervisor, every 60 seconds:

```text
[{elapsed:.0f}s] {n} matches completed | {rate:.1f} matches/min | {alive}/{workers} workers alive
```

Plus its worker-lifecycle lines:

```text
Starting {workers} workers...
Worker {i} exited (code {rc}), restarting...
Recycled longest-running worker (PID {pid}, age {age:.0f}s)
Shutting down, terminating {workers} workers...
```

## Exit summary

Printed when the run ends, including when it ends by interrupt:

```text
=== play-draft-games ===
matches played   {played}
elapsed          {elapsed}
output           {output_path}
```

`played` is the number of rows the output file gained since the run began. Failed matches
are not counted: workers draw their own pairings, so a failure costs time rather than data.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Requested matches recorded, or clean interrupt |
| 2 | Validation failure, missing file, or bad flag |
| 130 | Interrupt before the first match completed |

## Output

Appends to `--output-path` in the sealed match-outcome format
([`2026-03-28-sealed-deck-picker.md`](../../2026-03-28-sealed-deck-picker.md) § Phase 0
Step 4), one row per match, written by the worker JVM that played it. `method_A` and
`method_B` carry the two seats' agent labels.

The command writes no other corpus. In particular it does not write `cards-played.txt`, and
it never appends to `output/sealed/match-outcomes.txt`.

## Downstream

```text
python scripts/analyze_winrates.py output/draft/draft-games.txt
```

Runs unchanged. This feature must not modify that script.
