# Data Model: Sealed Dataset Preparation

## Entities

### CardEmbedding

An embedding file produced by `encode-cards` for a single card script.

| Field | Type | Description |
|---|---|---|
| `embedding` | `float32[2 × d_model]` | Pooled card representation: `cat([max_pool, mean_pool])` over transformer encoder outputs. Dimension is `2 × d_model` of the loaded model (e.g. 256 for d_model=128, 512 for d_model=256). |

**Storage**: One `.npz` file per card script. Located in the same directory as the source `.txt` card script, with identical filename stem (e.g. `Lightning-Bolt.txt` → `Lightning-Bolt.npz`).

**Invariants**:
- The `name:` line is stripped from the card text before encoding — the embedding captures game characteristics, not card identity.
- Files are written atomically (temp → rename). Existence of the file implies it is complete and valid.
- Content is deterministic given the same encoder model and vocabulary.

---

### SealedPool

One line in the pools text file, representing the non-land cards opened from 6 boosters of a single MTG set.

| Field | Type | Description |
|---|---|---|
| card names | `List<String>` | Card names as returned by Forge's booster generator. Duplicates allowed. Basic lands excluded. |

**Storage**: All pools for a given set are stored in a single file at `output/sealed/pools/{set-code}/pools.txt`, one pool per line. Card names within a line are separated by semicolons.

**Invariants**:
- No basic land names appear in any pool line.
- Card names match the names used in the corresponding `.npz` embedding files (same Forge data source).
- Duplicate card names within a single pool are valid (e.g. opening two copies of the same common).
- File is overwritten (not appended) on each `generate-pools` run.

---

### PoolDataset

The complete `pools.txt` file for a given set.

| Field | Type | Description |
|---|---|---|
| set code | `String` | The MTG set code used to generate all pools (e.g. `RVR`). |
| pool count | `int` | Number of lines (= number of pools). |
| path | `Path` | Absolute path to the `pools.txt` file. |

---

## File Layout

```text
output/
├── cardsfolder/               # Card script source files (from forge-connector)
│   ├── a/
│   │   ├── air_elemental.txt  # Source card script
│   │   ├── air_elemental.npz  # Embedding (produced by encode-cards)
│   │   └── ...
│   └── ...
└── sealed/
    └── pools/
        └── RVR/
            └── pools.txt      # 10,000 pool lines (produced by generate-pools)
```

## State Transitions

```
encode-cards run:
  .txt exists, no .npz  →  encode  →  .npz written (atomic)
  .txt exists, .npz exists  →  skip  →  no change
  run interrupted mid-write  →  .tmp.npz cleaned  →  re-run re-encodes

generate-pools run:
  pools.txt absent  →  generate  →  pools.txt written
  pools.txt present  →  overwrite  →  pools.txt replaced with fresh data
```
