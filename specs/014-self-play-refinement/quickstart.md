# Quickstart: Self-Play Refinement

**Feature**: 014-self-play-refinement

## Prerequisites

- Python venv activated with `sealed` package installed (`pip install -e ".[dev]"`)
- `forge-connector` JAR built (`cd forge-connector && mvn package -DskipTests`)
- Sibling Forge checkout at `../forge` built with `mvn install -DskipTests`
- Card embeddings generated (`python -m sealed encode-cards`)
- A trained scorer checkpoint (e.g. `models/sealed/scorer/best_*.pt` or `latest.pt`)
- `resources/AllPrintings.json` present (needed for eligible set enumeration)

## Self-Play Refinement Loop

### Step 1: Generate pools from random sets

```bash
python -m sealed generate-pools --size 10000
```

Produces `output/sealed/pools/pools.txt` with ~10,000 pools from random sealed-legal sets. Each line: `SET_CODE;Card1|Card2|...|CardN`.

To generate from a specific set instead:

```bash
python -m sealed generate-pools --set MH3 --size 10000
```

### Step 2: Build scorer decks

```bash
python -m sealed build-decks \
    --pools-path output/sealed/pools/pools.txt \
    --checkpoint models/sealed/scorer/latest.pt
```

Produces `output/sealed/generated-decks.txt`. Each line: `SET_CODE;Card1|Card2|...|Card40` — a complete 40-card deck (23 nonland + 17 basic lands).

### Step 3: Generate self-play match data

```bash
python -m sealed match-outcomes \
    --generated-decks-path output/sealed/generated-decks.txt
```

Runs indefinitely, appending match outcomes to `output/sealed/match-outcomes.txt`. Press Ctrl-C to stop.

Each match: scorer-built deck A vs opponent deck B (methods 1-5 with weights 4:3:2:1:4). Same-set pairing enforced.

### Step 4: Retrain the scorer

```bash
python -m sealed train-scorer \
    --outcomes-path output/sealed/match-outcomes.txt \
    --epochs 20
```

Trains on the combined corpus (Phase 0 + self-play data).

### Step 5: Evaluate

```bash
python -m sealed evaluate-scorer \
    --checkpoint models/sealed/scorer/best_*.pt
```

Evaluates on a randomly selected set by default. Use `--set RVR` to evaluate on a specific set.

### Step 6: Repeat

Go back to Step 1 with the new scorer checkpoint. Each iteration should produce a scorer that builds better decks, creating harder training data for the next iteration.

## Phase 0 (unchanged)

All existing workflows continue to work without changes:

```bash
# Phase 0 match generation (no --generated-decks-path)
python -m sealed match-outcomes

# Fixed-set pool generation
python -m sealed generate-pools --set RVR --size 10000
```

## CLI Reference

### generate-pools

| Argument | Default | Description |
|---|---|---|
| `--set` | *(random)* | Set code. When omitted, each pool uses a random sealed-legal set. |
| `--size` | 10000 | Number of pools to generate. |
| `--pools-path` | `output/sealed/pools/pools.txt` (no set) or `output/sealed/pools/{set}/pools.txt` (with set) | Output directory. |

### build-decks

| Argument | Default | Description |
|---|---|---|
| `--pools-path` | *(required)* | Input pools file (with set code prefixes). |
| `--checkpoint` | `models/sealed/scorer/latest.pt` | Scorer model checkpoint. |
| `--cards-path` | `output/cardsfolder/` | Directory with `.npz` card embeddings. |
| `--output` | `output/sealed/generated-decks.txt` | Output generated-decks file. |

### match-outcomes

| Argument | Default | Description |
|---|---|---|
| `--workers` | 12 | Number of parallel Java workers. |
| `--generated-decks-path` | *(none)* | Path to generated-decks file. Enables self-play mode. |

### evaluate-scorer

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Scorer checkpoint to evaluate. |
| `--set` | *(random)* | Set code. When omitted, a random sealed-legal set is used. |
| `--pools` | 12 | Number of evaluation pools. |
| `--best-of` | 3 | Games per match. |
| `--workers` | 4 | Parallel Java workers. |
