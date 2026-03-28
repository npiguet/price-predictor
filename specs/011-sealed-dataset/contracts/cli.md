# CLI Contract: python -m sealed

## Module Entry Point

```
python -m sealed <command> [options]
```

Both commands are independent. Neither requires the other to have run first.

---

## encode-cards

Encodes all card scripts in a folder to `.npz` embedding files. Skips cards that already have an embedding file.

```
python -m sealed encode-cards
    [--encoder-path PATH]
    [--vocab-path PATH]
    [--cards-path PATH]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--encoder-path` | `models/price-predictor/transformer/latest.pt` | Path to the pretrained transformer model `.pt` file |
| `--vocab-path` | `models/price-predictor/transformer/vocab.txt` | Path to the tokenizer vocabulary file |
| `--cards-path` | `output/cardsfolder/` | Directory containing `.txt` card script files (searched recursively) |

### Behavior

1. Validates that `--encoder-path` and `--vocab-path` exist; exits with error if not.
2. Scans `--cards-path` recursively for `.txt` files.
3. For each `.txt` file without a corresponding `.npz`:
   - Strips the `name:` line from the card text.
   - Tokenizes and encodes using the loaded model.
   - Writes the embedding atomically (`{stem}.tmp.npz` → `{stem}.npz`).
4. Reports progress to stdout every 100 cards.
5. Prints a summary on completion: cards processed, cards skipped.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | All cards processed successfully (or already encoded) |
| `1` | One or more cards failed to encode (errors logged; processing continues) |
| `2` | Fatal error (missing encoder/vocab, unreadable cards-path) |

### Example Output

```
Encoding cards in output/cardsfolder/
Progress: 100/32000 encoded (31900 skipped)
Progress: 200/32000 encoded (31800 skipped)
...
Done: 32000 processed, 0 skipped, 0 errors
```

---

## generate-pools

Generates sealed pools using Forge's booster generation logic and writes them to a flat text file.

```
python -m sealed generate-pools
    [--set SET_CODE]
    [--size N]
    [--pools-path PATH]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--set` | `RVR` | MTG set code to generate boosters from (e.g. `RVR`, `MH3`, `BLB`) |
| `--size` | `10000` | Number of sealed pools to generate |
| `--pools-path` | `output/sealed/pools/{set}/` | Output directory; `pools.txt` is written here |

### Behavior

1. Resolves `--pools-path` (substituting `{set}` placeholder with the set code).
2. Creates the output directory if it does not exist.
3. Invokes the forge-connector JAR (`PoolMain`) as a subprocess.
4. Streams progress output from the subprocess to stdout.
5. Overwrites any existing `pools.txt` at the target path.

### Output Format

`pools.txt` — one pool per line, card names separated by semicolons:

```
Ponder;Lightning Bolt;Counterspell;Dark Ritual;...
Giant Growth;Serra Angel;Wrath of God;...
```

- No basic land names appear in any line.
- Duplicate card names within a line are valid.
- Each line contains 84–90 card names (varies by set booster size).

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | All pools generated and written successfully |
| `2` | Fatal error (invalid set code, JAR not found, Java not on PATH) |

### Example Output

```
Generating 10000 RVR sealed pools...
Generated 1000/10000 pools
Generated 2000/10000 pools
...
Done: 10000 pools written to output/sealed/pools/RVR/pools.txt
```

---

## Java subprocess: PoolMain

The `generate-pools` command invokes the forge-connector JAR with `PoolMain` as the entry class.

```
java -cp <classpath> com.pricepredictor.connector.PoolMain
    --set <SET_CODE>
    --size <N>
    --pools-path <PATH>
```

**Stdout protocol**: One progress line per 1000 pools: `Generated N/TOTAL pools`. Final line: `Done`.

**Exit codes**: `0` = success, `1` = error (message on stderr).
