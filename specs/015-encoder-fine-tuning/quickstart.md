# Quickstart: Encoder Fine-Tuning (Phase B)

**Feature**: 015-encoder-fine-tuning

This walk-through runs Phase B end-to-end on top of an existing Phase A
checkpoint. It assumes the standard repo layout, that
`output/cardsfolder/` is fully populated with `.txt` and `.npz` files,
and that `models/price-predictor/transformer/latest.pt` and `vocab.txt`
exist.

## Prerequisites

```bash
.venv\Scripts\activate    # Windows
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
```

A Phase A scorer checkpoint must exist before Phase B starts. If you don't
have one yet:

```bash
python -m sealed train-scorer
# → models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt
```

(The Phase A run uses `--embedding-lr 0` by default and stops via
`--patience` once `val_acc` plateaus; expect 10–20 epochs.)

## Step 1 — Phase B kickoff

```bash
python -m sealed train-scorer \
    --scorer-checkpoint models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --embedding-lr 1e-7
```

What this does:

- Loads scorer weights from the Phase A checkpoint (`--scorer-checkpoint`); discards the Phase A optimizer state, epoch, and `best_val_accuracy`.
- Loads encoder weights from the price-predictor's `latest.pt` (the same file that produced the existing `.npz` cache).
- Constructs a single `AdamW` optimizer with two parameter groups: scorer at `--lr` (default `1e-5`), encoder at `--embedding-lr` (`1e-7`).
- Trains until validation accuracy plateaus per `--patience` (default 5) or `--epochs` (default 100), whichever fires first.

End-of-epoch logging adds two new lines for Phase B:

```text
Epoch N: train_loss=...  train_acc=...  val_loss=...  val_acc=...  embedding_drift=0.012345
  scores: winner=...±...  loser=...±...
  grad_norms: scorer=...  encoder=...
```

If `embedding_drift` exceeds 1.0 within the first three epochs, abort
(Ctrl-C) and restart with a lower `--embedding-lr` (try `1e-8`).

To continue an interrupted Phase B run, re-invoke with `--resume`
instead of `--scorer-checkpoint`:

```bash
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt \
    --embedding-lr 1e-7
```

The resumed checkpoint already contains encoder weights; passing
`--encoder-checkpoint` explicitly here is a hard error.

## Step 2 — Refresh the `.npz` cache

```bash
python -m sealed encode-cards \
    --scorer-checkpoint models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt \
    --clean
```

What this does:

- Extracts encoder weights from the Phase B scorer checkpoint's `encoder_state_dict`.
- Loads the price-predictor's `TransformerConfig` and tokenizer (the encoder *architecture* is unchanged from Phase A; only the weights moved).
- Deletes every existing `.npz` under `output/cardsfolder/` (`--clean`).
- Re-encodes every `.txt`. Cards never seen during Phase B are re-encoded too — they inherit whatever generalization the encoder picked up.

Expect this to take 10–30 minutes on CPU for a fully populated cache;
GPU is significantly faster but not required.

If you forget `--scorer-checkpoint` and just run
`python -m sealed encode-cards --clean`, the default
`--encoder-checkpoint` (price-predictor `latest.pt`) is used and you'll
re-cache against the **pre-fine-tuned** encoder, undoing Phase B.

## Step 3 — Evaluate Phase B

Phase B writes to a distinct filename (`best_phaseB_l<...>_emblr<...>.pt`)
that does not collide with the Phase A `best_*.pt` in the same
`--checkpoint-dir` (FR-009a). Both files exist side by side after a
Phase B run.

```bash
python -m sealed evaluate-scorer \
    --checkpoint models/sealed/scorer/best_phaseB_l6_h4_s4_ff1088_mlp256_lr1e-05_emblr1e-07.pt \
    --set BLB
```

Then evaluate Phase A against the same pool set, switching back to the
Phase A `.npz` cache first:

```bash
# Restore the Phase A .npz cache (saved before Step 2's --clean), then:
python -m sealed evaluate-scorer \
    --checkpoint models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt \
    --set BLB
```

Compare the two win rates. If Phase B wins (or matches with reasonable
confidence), it ships. If Phase B regresses, revert to Phase A by
restoring the Phase A `.npz` cache and using the Phase A checkpoint —
no code changes needed (SC-003).

## Common Errors and Their Fixes

| Error | Cause | Fix |
|---|---|---|
| `--resume <path> is a Phase A checkpoint but --embedding-lr 1e-7 requests Phase B` | Trying to start Phase B with `--resume` instead of `--scorer-checkpoint`. | Use `--scorer-checkpoint <phaseA>.pt`. |
| `--embedding-lr 1e-7 requires either --scorer-checkpoint or --resume` | Phase B from a randomly-initialized scorer. | Add `--scorer-checkpoint <phaseA>.pt`. |
| `architecture flag --n-layers conflicts with --scorer-checkpoint` | Architecture flag explicitly passed alongside `--scorer-checkpoint`. | Omit the architecture flag — it's inherited. |
| `--encoder-checkpoint conflicts with --resume on a Phase B checkpoint` | Explicit `--encoder-checkpoint` on a Phase B `--resume`. | Omit `--encoder-checkpoint`. |
| `<path> is a Phase A scorer checkpoint and contains no encoder weights` | `encode-cards --scorer-checkpoint` pointed at a Phase A checkpoint. | Use `--encoder-checkpoint` for non-Phase-B sources. |
| `--encoder-checkpoint and --scorer-checkpoint are mutually exclusive` | Both flags explicitly passed to `encode-cards`. | Choose one source. |

## What Happens When Things Are Working

- **Step 0 of Phase B**: a `_log` line records the size of the reference batch (number of unique cards in batch 0). This is the set of cards used for `embedding_drift` for the rest of the run.
- **Each epoch**: `train_loss` and `val_loss` decrease (slower than Phase A); `val_acc` may improve, plateau, or regress; `embedding_drift` grows monotonically and stays well below 1.0; encoder gradient norm sits near 1.0 (the clip threshold) for early epochs and tapers off.
- **`best_*.pt`** is overwritten when `val_acc` hits a new peak; otherwise `latest.pt` is the only file updated.
- **Early stop**: training ends when `--patience` epochs have passed without a new peak.

## Further Reading

- Spec: `specs/015-encoder-fine-tuning/spec.md`
- Plan + research: `specs/015-encoder-fine-tuning/plan.md` and `research.md`
- Original brief: `specs/2026-04-27-encoder-fine-tuning.md`
- Related: `specs/2026-03-28-sealed-deck-picker.md` (the embedding-schedule context that motivates Phase B).
