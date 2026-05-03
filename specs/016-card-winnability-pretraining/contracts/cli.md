# CLI Contracts

Two new sealed subcommands and two default-value flips in existing
sealed subcommands. Each contract below is the minimum set of
flags, error conditions, and exit codes that this feature ships.

## `python -m sealed build-vocab` (new)

Build the sealed-side tokenizer vocabulary from the converted card
corpus. Thin wrapper around
`price_predictor.application.build_vocabulary.build_vocabulary` with
sealed defaults (Decision D-1, FR-007–FR-009).

### Flags

| Flag             | Type | Default                                | Notes                                   |
|------------------|------|----------------------------------------|-----------------------------------------|
| `--cards-folder` | path | `output/cardsfolder/`                  | Searched recursively for `*.txt`.       |
| `--vocab-path`   | path | `models/sealed/encoder/vocab.txt`      | Output. Parent directory auto-created.  |
| `--target-size`  | int  | `5000`                                 | Approximate vocab size (Decision D-1).  |

### Behavior

1. Resolve `--cards-folder`. If empty (no `*.txt` files), exit code
   `1` with message `"Cards folder is empty: <path>. Run python -m
   price_predictor convert first."`.
2. Call `build_vocabulary(cards_path, freq_threshold=2,
   printings_path=resources/AllPrintings.json if present else None)`.
3. Truncate the resulting vocab dict to `--target-size` entries by
   frequency (special tokens always preserved).
4. Call `tokenizer_store.save_vocabulary(vocab, vocab_path)`.
5. Print one summary line: `"Wrote N tokens to <vocab-path> (corpus
   coverage: P%)"`.

### Exit codes

| Code | Condition                                                   |
|------|-------------------------------------------------------------|
| `0`  | Vocabulary written successfully.                            |
| `1`  | Cards folder missing or empty.                              |
| `2`  | `--target-size` smaller than the seed-token count.          |

### Side effects

- Writes `--vocab-path`. Does NOT touch
  `models/price-predictor/transformer/vocab.txt` (FR-008).

---

## `python -m sealed train-encoder` (new)

Train a sealed encoder (token + card encoder + regression head) on
per-card winnability targets. Saves only the encoder weights
(FR-014, FR-016, FR-017, FR-020).

### Flags

| Flag                    | Default                              | FR     |
|-------------------------|--------------------------------------|--------|
| `--cards-played-path`   | `output/sealed/cards-played.txt`     | FR-021 |
| `--cards-folder`        | `output/cardsfolder/`                | FR-021 |
| `--vocab-path`          | `models/sealed/encoder/vocab.txt`    | FR-021 |
| `--model-output`        | `models/sealed/encoder/`             | FR-021 |
| `--batch-size`          | `64`                                 | FR-021 |
| `--epochs`              | `100`                                | FR-021 |
| `--lr`                  | `1e-4`                               | FR-021 |
| `--patience`            | `20`                                 | FR-021 |
| `--dropout`             | `0.1`                                | FR-021 |
| `--n-layers`            | `6`                                  | FR-021 |
| `--n-heads`             | `4`                                  | FR-021 |
| `--n-pool-queries`      | `4`                                  | FR-021 |
| `--shrinkage-k`         | `20`                                 | FR-021 |

Hardcoded constants (not flags): `d_model = 256`, `ff_dim = 1024`,
`val_fraction = 0.2`, `random_seed = 42`, `loss = MSE`,
`stratification = winnability quartiles`. `max_seq_len` is computed
from the corpus at start (rounded up to a multiple of 8).

### Behavior

1. **Pre-flight checks** (in this order; first failure wins):
   - `--vocab-path` missing → exit `2`, message names
     `python -m sealed build-vocab`.
   - `--cards-played-path` missing or zero-length → exit `3`,
     message names `python -m sealed match-outcomes`.
   - `--cards-folder` empty → exit `4`, message names
     `python -m price_predictor convert`.
2. **Aggregation pass**: stream `cards-played.txt`, count
   `wins_when_played[c]` and `wins_when_in_deck[c]` for the winning
   side of every game, exclude cards with `wins_when_in_deck == 0`,
   apply Bayesian shrinkage with `--shrinkage-k`.
3. **Corpus consistency check** (FR-023d): for every card name in
   the label map, verify that `output/cardsfolder/<filename>.txt`
   exists (using `card_name_corrections` for known typos). If any
   are missing, exit `5` with a message naming up to 20 missing
   cards plus a total count, pointing the user at
   `python -m price_predictor convert`.
4. Write `output/sealed/cards-win-rates.txt` (FR-013a).
5. Build the train/val split (stratified on winnability quartile,
   card-level disjoint, seed=42, val_fraction=0.2).
6. Construct a fresh `SealedEncoderModel(SealedEncoderConfig(...))`,
   move to GPU if available.
7. Train with AdamW + LambdaLR (linear warmup, then constant) +
   MSE loss. Snapshot the best-by-val-loss checkpoint via the
   `_BestCheckpoint` pattern. Stop early after `--patience`
   consecutive no-improvement epochs.
8. After training: write
   `models/sealed/encoder/{ISO_TIMESTAMP}.pt` with only
   `model_state_dict` (regression head filtered out) +
   `config`. Copy that file's contents to
   `models/sealed/encoder/latest.pt`.
9. Print a final summary: `"Best epoch: E, val_loss: V. Saved
   <path>"`.

### Stdout

Per-epoch progress lines via `_log()` (timestamp prefix), matching
`train_scorer.py` style:

```
[2026-05-03 14:22:01] Epoch 1/100  train_loss=0.0481  val_loss=0.0413  best=*
[2026-05-03 14:22:18] Epoch 2/100  train_loss=0.0395  val_loss=0.0398  best=*
...
[2026-05-03 14:35:02] Early stop after 22 epochs without improvement.
[2026-05-03 14:35:02] Best epoch: 21, val_loss: 0.0312. Saved models/sealed/encoder/2026-05-03T14-22-01.pt
```

### Exit codes

| Code | Condition                                                         |
|------|-------------------------------------------------------------------|
| `0`  | Training completed; best checkpoint saved.                        |
| `2`  | Vocabulary file missing.                                          |
| `3`  | `cards-played.txt` missing or empty.                              |
| `4`  | Cards folder empty.                                               |
| `5`  | Corpus references cards absent from `output/cardsfolder/`.        |
| `6`  | Misconfiguration (`d_model % n_pool_queries != 0`, etc.).         |

### Side effects

- Writes `models/sealed/encoder/{timestamp}.pt`,
  `models/sealed/encoder/latest.pt`, and
  `output/sealed/cards-win-rates.txt`.
- Does NOT touch `match-outcomes.txt` or `cards-played.txt`.

---

## `python -m sealed encode-cards` (default change only)

The behavior is unchanged. This feature flips the default of
`--encoder-checkpoint` (FR-025).

### Default change

| Flag                    | Old default                                            | New default                            |
|-------------------------|--------------------------------------------------------|----------------------------------------|
| `--encoder-checkpoint`  | `models/price-predictor/transformer/latest.pt`         | `models/sealed/encoder/latest.pt`      |

### New error condition (FR-026)

When the resolved default checkpoint does not exist *and* the user
did not pass `--encoder-checkpoint` explicitly, exit code `2` with
message:

```
Sealed encoder not found at models/sealed/encoder/latest.pt.
Run python -m sealed train-encoder, or pass --encoder-checkpoint <path> explicitly.
```

Existing path with explicit `--encoder-checkpoint` continues to
work without modification (FR-027).

### Vocab path coupling

The default `--vocab-path` for `encode-cards` becomes
`models/sealed/encoder/vocab.txt` (matches the default encoder).
Old default (`models/price-predictor/transformer/vocab.txt`)
remains valid via explicit `--vocab-path`.

---

## `python -m sealed train-scorer` (default change only)

Identical to `encode-cards`: flip the `--encoder-checkpoint`
default and add the same missing-file error (FR-024, FR-026).

### Default change

| Flag                    | Old default                                            | New default                            |
|-------------------------|--------------------------------------------------------|----------------------------------------|
| `--encoder-checkpoint`  | `models/price-predictor/transformer/latest.pt`         | `models/sealed/encoder/latest.pt`      |

Phase A and Phase B both pick up the new default. Phase B's
`--scorer-checkpoint <phaseA>.pt` continues to override only the
scorer state-dict; the encoder-checkpoint default still resolves
to the new path unless explicitly overridden.

### New error condition

Same shape as `encode-cards`: exit code `2` if the resolved default
file is missing and no explicit `--encoder-checkpoint` was passed.

### Resume semantics

When `--resume <phaseB>.pt` is passed, the resume code path
already pulls the encoder weights from the resumed checkpoint
(per `train_scorer.py`'s existing logic). That path is unchanged.
The default flip only affects fresh kickoffs.
