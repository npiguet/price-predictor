# CLI Contract: `sealed match-outcomes`

## Command

```
python -m sealed match-outcomes [--workers N]
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--workers` | int | 12 | Number of parallel Java worker processes to spawn |

## Behavior

1. Creates output directory `./output/sealed/` if it does not exist
2. Spawns `N` Java worker subprocesses, each running `MatchWorkerMain`
3. Workers run indefinitely, each independently generating match outcomes and appending to `./output/sealed/match-outcomes.txt`
4. Supervisor prints a status line to stdout every 60 seconds
5. On Ctrl+C (SIGINT) or SIGTERM, supervisor terminates all workers and exits cleanly
6. If a worker crashes, supervisor restarts it automatically

## Output

### Terminal (stdout)

```
Starting 12 workers...
Worker 0 started (PID 12345)
Worker 1 started (PID 12346)
...
[60s] 47 matches completed | 47.0 matches/min | 12/12 workers alive
[120s] 98 matches completed | 49.0 matches/min | 12/12 workers alive
Worker 3 exited (code 1), restarting...
Worker 3 started (PID 12400)
...
Shutting down, terminating 12 workers...
Done.
```

### File Output (`./output/sealed/match-outcomes.txt`)

One line per match outcome, appended atomically:

```
card1|card2|...|card40;card1|card2|...|card40;wins_a;wins_b
```

**Fields** (semicolon-separated):
1. **deck_a**: Pipe-separated card names (40 cards, Forge canonical names, duplicates repeat)
2. **deck_b**: Same format as deck_a
3. **wins_a**: Integer 0-2, games won by deck A
4. **wins_b**: Integer 0-2, games won by deck B

**Invariants**:
- `wins_a + wins_b` is 2 or 3
- Each deck contains exactly 40 cards
- File is append-only; existing data is never overwritten

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown (user interrupted) |
| 2 | Configuration error (e.g. Java not found, JAR not built) |

## Java Worker Subprocess

Each worker is invoked as:

```
java -Xmx1200m -cp <classpath> com.pricepredictor.connector.MatchWorkerMain
```

The classpath includes:
- `forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`
- `forge-game-2.0.10-SNAPSHOT.jar`
- `forge-core-2.0.10-SNAPSHOT.jar`
- `forge-gui-2.0.10-SNAPSHOT.jar`
- `forge-ai-2.0.10-SNAPSHOT.jar`
- `forge-gui/target/dependency/*`

Workers write directly to the output file (not via stdout). Worker stdout/stderr is inherited by the supervisor for diagnostic output.
