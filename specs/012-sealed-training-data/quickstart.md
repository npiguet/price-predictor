# Quickstart: Sealed Training Data Generation

## Prerequisites

1. **Java 17+** on PATH
2. **Forge** built at `../forge/` (relative to project root):
   ```bash
   cd ../forge
   mvn install -DskipTests
   ```
3. **forge-connector** built:
   ```bash
   cd forge-connector
   mvn package -DskipTests
   ```

## Generate Match Outcomes

```bash
python -m sealed match-outcomes
```

This starts 12 worker processes that continuously generate sealed match outcomes. Output is appended to `./output/sealed/match-outcomes.txt`.

### Options

```bash
# Use fewer workers (e.g. on a machine with limited RAM)
python -m sealed match-outcomes --workers 4
```

### Stop

Press **Ctrl+C** to stop all workers and exit cleanly.

### Verify Output

Check that the file contains well-formed records:

```bash
head -5 output/sealed/match-outcomes.txt
```

Each line should have 4 semicolon-separated fields: two pipe-separated deck lists and two integer win counts.

## Architecture

```
python -m sealed match-outcomes
        │
        ├── Worker 0 (java MatchWorkerMain) ──┐
        ├── Worker 1 (java MatchWorkerMain) ──┤
        ├── ...                               ├──► output/sealed/match-outcomes.txt
        └── Worker 11 (java MatchWorkerMain) ─┘
```

Each worker independently:
1. Picks a random sealed-legal set
2. Generates 2 booster pools (6 boosters each)
3. Builds a deck from each pool (4 construction methods, weighted random)
4. Plays a best-of-3 match via Forge AI
5. Appends one result line to the output file
6. Repeats indefinitely

The Python supervisor monitors workers and restarts any that crash.
