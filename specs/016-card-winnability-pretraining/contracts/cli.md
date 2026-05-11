# CLI Contracts

Two sealed subcommands exercised by this feature (`build-vocab` and
`train-encoder`), plus default-value flips on two existing subcommands
(`encode-cards`, `train-scorer`) that were already shipped in the v1
iteration of spec 016 and are documented here for completeness.

## `python -m sealed build-vocab`

Build the sealed-side tokenizer vocabulary from the converted card
corpus. Thin wrapper around
`price_predictor.application.build_vocabulary.build_vocabulary` with
sealed defaults (Decision D-1, FR-007–FR-009a).

### Flags

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--cards-folder` | path | `output/cardsfolder/` | Searched recursively for `*.txt`. |
| `--vocab-path` | path | `models/sealed/encoder/vocab.txt` | Output. Parent directory auto-created. |
| `--target-size` | int | `5000` | Approximate vocab size (Decision D-1). Seeded specials, including `[MASK]`, are always preserved. |

### Behavior

1. Resolve `--cards-folder`. If empty (no `*.txt` files), exit code `1`
   with message `"Cards folder is empty: <path>. Run python -m
   price_predictor convert first."`.
2. Call `build_vocabulary(cards_path, freq_threshold=2, printings_path=
   resources/AllPrintings.json if present else None)`. The seeded specials
   include `[PAD]`, `[UNK]`, `cardname`, **`[MASK]`** (FR-009a).
3. Truncate the resulting vocab dict to `--target-size` entries by
   frequency. Seeded specials and domain tokens are always preserved.
4. Call `tokenizer_store.save_vocabulary(vocab, vocab_path)`.
5. Print one summary line: `"Wrote N tokens to <vocab-path> (corpus
   coverage: P%)"`.

### Exit codes

| Code | Condition |
|---|---|
| `0` | Vocabulary written successfully. |
| `1` | Cards folder missing or empty. |
| `2` | `--target-size` smaller than the seed-token count (now includes `[MASK]`). |

### Side effects

- Writes `--vocab-path`. Does NOT touch
  `models/price-predictor/transformer/vocab.txt` (FR-008).
- The output vocab MUST contain a `[MASK]` token; downstream `train-encoder`
  rejects vocabularies without it (FR-023a).

---

## `python -m sealed train-encoder`

Train a sealed encoder + five regression heads + an MLM head on per-card
winnability labels and an auxiliary mask-prediction objective. Saves only
the encoder weights (FR-014, FR-016, FR-017, FR-020).

### Flags

| Flag | Default | FR |
|---|---|---|
| `--cards-played-path` | `output/sealed/cards-played.txt` | FR-021 |
| `--cards-folder` | `output/cardsfolder/` | FR-021 |
| `--vocab-path` | `models/sealed/encoder/vocab.txt` | FR-021 |
| `--model-output` | `models/sealed/encoder/` | FR-021 |
| `--batch-size` | `64` | FR-021 |
| `--epochs` | `100` | FR-021 |
| `--lr` | `1e-4` | FR-021 |
| `--patience` | `20` | FR-021 |
| `--dropout` | `0.1` | FR-021 |
| `--n-layers` | `6` | FR-021 |
| `--n-heads` | `4` | FR-021 |
| `--n-pool-queries` | `4` | FR-021 |
| `--shrinkage-k` | `20` | FR-021 |
| `--mlm-weight` | `0.1` | FR-021 |
| `--mlm-mask-prob` | `0.15` | FR-021 |

**Hardcoded constants** (not flags), per FR-022 + Clarifications 2026-05-10:

- `d_model = 256`, `ff_dim = 1024`
- `val_fraction = 0.2`, `random_seed = 42`
- `stratification = score_play quartiles` with FR-018 fallback chain
- `loss formula = L_reg + (--mlm-weight) · L_mlm` per FR-017
- Per-head per-batch sum-to-1 weight normalization on `L_reg`
- `optimizer = AdamW with per-parameter-group max-norm 1.0 gradient clipping`
- `lr schedule = linear warmup over the first 5% of (--epochs × batches_per_epoch) steps, then constant`
- `max_seq_len = corpus longest tokenized card, rounded up to multiple of 8`

### Behavior

1. **Pre-flight checks** (in this order; first failure wins):
   - `--vocab-path` missing → exit `2`, message names `python -m sealed build-vocab`.
   - `--vocab-path` present but lacking `[MASK]` token → exit `2`, message names `python -m sealed build-vocab` (FR-023a).
   - `--cards-played-path` missing or zero-length → exit `3`, message names `python -m sealed match-outcomes`.
   - `--cards-folder` empty → exit `4`, message names `python -m price_predictor convert`.
   - Architectural mismatch (`d_model % --n-pool-queries != 0` or `d_model % --n-heads != 0`) → exit `6` (raised inside `SealedEncoderConfig.__post_init__`).

2. **Aggregation pass 1** (FR-010a): stream `cards-played.txt`, count for
   every card observed in either side's deck:
   - 4 primary counters (`wins_when_played`, `wins_when_in_deck`,
     `losses_when_played`, `losses_when_in_deck`)
   - 4 `@play` subset counters (same four, restricted to games where the
     card's owner was the starter)

3. **Missing-card check** (FR-023d): for every card name observed in
   pass 1, verify the card has a `.txt` under `--cards-folder` (using
   `card_name_corrections` and the front-face / `rebalanced/`
   fallbacks). For any miss, log a warning naming up to 20 missing
   cards plus a total count, pointing the user at `python -m
   price_predictor convert`, then drop those cards from the counter
   dict (and therefore from the label map, the split, and the
   dataset). This does **not** abort the run — training proceeds with
   the remaining cards.

4. **Aggregation pass 2** (FR-010b): stream `cards-played.txt` again. For
   every card, on first encounter resolve its color identity from its
   `mana cost:` line via the regex helper of Decision D-16. For each
   game, compute each side's deck-color set as the union over its deck
   contents. For every (card, color-X-in-deck) pairing, increment four
   per-color counters: `wins_when_played_with_X`,
   `wins_when_in_deck_with_X`, `losses_when_played_with_X`,
   `losses_when_in_deck_with_X`.

5. **Build label map** (FR-011, FR-012): compute 9 raw + 9 shrunk labels
   per card. Cells whose slice denominator is zero are stored as
   present-but-empty (the head's loss contribution will be masked out at
   training time). Cards with `wins_when_in_deck + losses_when_in_deck
   == 0` are excluded entirely.

6. **Write `output/sealed/cards-win-rates.txt`** (FR-013a, Decision D-15):
   one header row + one row per included card, 23 columns each, sorted by
   `shrunk_score_play` descending. Empty cells are written as the empty
   string in both raw and shrunk columns.

7. **Build the train/val split** (FR-018, Decision D-4): card-level
   disjoint, stratified on `score_play` quartile with the fallback chain;
   `val_fraction = 0.2`, `random_seed = 42`.

8. **Construct a fresh `SealedEncoderModel(SealedEncoderConfig(...))`**
   with the new `regression_heads` ModuleDict and the new `mlm_head`,
   move to GPU if available.

9. **Train with**:
   - `AdamW` optimizer + `LambdaLR` (linear warmup over the first 5% of
     scheduled steps, constant after).
   - Per-parameter-group max-norm 1.0 gradient clipping between
     `loss.backward()` and `optimizer.step()` (FR-022).
   - Per-card MLM mask draw at every batch step (Decision D-10):
     non-special, non-pad positions selected with probability
     `--mlm-mask-prob`; selected positions overwritten with `[MASK]`
     before the masked sequence enters the token encoder.
   - Per-batch per-head sum-to-1 weighted MSE on the 9 regression heads
     (Decisions D-11), summed with the `(1/5)` factor on the color-lift
     block per FR-017.
   - MLM cross-entropy at masked positions only; reduced by
     `mask.sum().clamp(min=1)`.
   - Full batch loss = `L_reg + (--mlm-weight) * L_mlm`.

10. **Validate at end of every epoch**: re-evaluate the full loss on the
    val set; update `_BestCheckpoint` on improvement; bump
    `epochs_since_best` otherwise. Stop early after `--patience`
    consecutive no-improvement epochs (FR-019, Clarification 2026-05-10).

11. **Save**: write `models/sealed/encoder/{ISO_TIMESTAMP}.pt` with
    `model_state_dict` (filtered to encoder children — heads and MLM
    head dropped per FR-020) + `config`. Copy that file's contents to
    `models/sealed/encoder/latest.pt`.

12. **Print final summary**: `"Best epoch: E, val_loss: V. Saved <path>"`.

### Stdout

Two log lines per epoch via `_log()` — the loss decomposition plus a
human-readable diagnostics line:

```
[2026-05-10 14:22:01] Epoch 1/100  train_loss=0.9194 (reg=0.1910, mlm=7.2841 ppl=1456.1)  val_loss=0.6419 (reg=0.0483, mlm=5.9355 ppl=378.5 acc=4.7%)  best=*
[2026-05-10 14:22:01]   val corr (pred vs target): score_play=+0.18 score_draw=+0.11 played_rate=+0.42 cast_lift=+0.06 | color_lift W=+0.03 U=-0.01 B=+0.09 R=+0.04 G=+0.02
...
[2026-05-10 14:35:02] Early stop after 21 epochs without improvement.
[2026-05-10 14:35:02] Best epoch: 24, val_loss: 0.3812. Saved models/sealed/encoder/2026-05-10T14-22-01.pt
```

- `mlm` is cross-entropy in nats; `ppl = exp(mlm)` is the more legible
  perplexity; `acc` is val-set top-1 masked-token accuracy.
- The `val corr` line is the per-head Pearson correlation between the
  encoder's predictions and the (shrunk) targets, over the cells each
  card actually contributes to — `~0` means the head is predicting the
  mean, `~0.3+` means it's tracking the per-card signal. A head with
  fewer than two non-empty val cells (or zero target variance) shows
  `--`.
- The `(reg=…, mlm=…)` breakdown and the diagnostics are informational;
  the `best=*` flag is still based on the `val_loss` column
  (`reg + (--mlm-weight)·mlm`) per FR-019.

### Exit codes

| Code | Condition |
|---|---|
| `0` | Training completed; best checkpoint saved. |
| `2` | Vocabulary missing or lacking `[MASK]`. |
| `3` | `cards-played.txt` missing or empty. |
| `4` | Cards folder empty. |
| `6` | Architectural misconfiguration (`d_model % n_pool_queries != 0`, etc.). |
| `130` | KeyboardInterrupt (Ctrl-C). |

(There is no longer an exit code for "corpus references missing cards" —
those are reported via a log warning and dropped, per FR-023d.)

### Side effects

- Writes `models/sealed/encoder/{timestamp}.pt`,
  `models/sealed/encoder/latest.pt`, and
  `output/sealed/cards-win-rates.txt`.
- Does NOT touch `match-outcomes.txt` or `cards-played.txt`.

---

## `python -m sealed encode-cards` (default change only — already shipped in v1)

The behavior is unchanged. The v1 iteration of spec 016 already flipped
the default of `--encoder-checkpoint` (FR-025); documented here for
completeness.

### Default

| Flag | Old default | Current default |
|---|---|---|
| `--encoder-checkpoint` | `models/price-predictor/transformer/latest.pt` | `models/sealed/encoder/latest.pt` |
| `--vocab-path` | `models/price-predictor/transformer/vocab.txt` | `models/sealed/encoder/vocab.txt` |

### Missing-file error (FR-026)

If the resolved default checkpoint does not exist *and* the user did not
pass `--encoder-checkpoint` explicitly, exit code `2` with message:

```
Sealed encoder not found at models/sealed/encoder/latest.pt.
Run python -m sealed train-encoder, or pass --encoder-checkpoint <path> explicitly.
```

Existing path with explicit `--encoder-checkpoint` continues to work
without modification (FR-027).

---

## `python -m sealed train-scorer` (default change only — already shipped in v1)

Identical to `encode-cards`: the v1 iteration of spec 016 flipped the
`--encoder-checkpoint` default (FR-024) and added the same missing-file
error (FR-026). Phase A and Phase B both pick up the new default. Phase
B's `--scorer-checkpoint <phaseA>.pt` continues to override only the
scorer state-dict; the encoder-checkpoint default still resolves to the
new path unless explicitly overridden. Resume semantics are unchanged.
