# CLI Contract: `python -m sealed train-scorer`

**Feature**: 015-encoder-fine-tuning

This file is the canonical, normative contract for the `train-scorer`
subcommand after this feature ships. It supersedes the matching section
of any older spec where the two disagree.

## Synopsis

```text
python -m sealed train-scorer [options]
```

Two valid invocation shapes:

1. **Phase A** — encoder frozen (consumes `.npz` cache), scorer trained from scratch or resumed.
2. **Phase B** — encoder fine-tuned alongside the scorer; bootstrapped from a Phase A checkpoint or resumed from a Phase B checkpoint.

The current run's phase is determined by `--embedding-lr`:
- `--embedding-lr 0` (default) → Phase A.
- `--embedding-lr <nonzero>` → Phase B.

## Flag Reference

### Existing flags (unchanged behavior)

| Flag | Default | Notes |
|---|---|---|
| `--outcomes-path` | `output/sealed/match-outcomes.txt` | Match-outcomes corpus. |
| `--cards-path` | `output/cardsfolder/` | Directory of `.npz` (and, in Phase B, `.txt`) card files. |
| `--checkpoint-dir` | `models/sealed/scorer/` | Where `latest.pt` and `best_*.pt` are written. |
| `--epochs` | `100` | Hard upper bound. |
| `--batch-size` | `64` | Training batch size. |
| `--lr` | `1e-5` | Scorer learning rate. (Default updated from `1e-3` per `specs/2026-04-27-encoder-fine-tuning.md` original brief.) |
| `--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`, `--mlp-hidden`, `--dropout` | (existing) | Scorer architecture. **Forbidden when `--scorer-checkpoint` is supplied** — architecture is inherited from the loaded checkpoint's stored config. |
| `--val-fraction` | `0.2` | Validation split fraction. |
| `--random-seed` | `42` | Seed for the deterministic train/val split (FR-011a). |

### New flags

| Flag | Default | Behavior |
|---|---|---|
| `--embedding-lr` | `0` | Encoder parameter group's learning rate. `0` = Phase A; non-zero = Phase B. (FR-001) |
| `--scorer-checkpoint <path>` | (none) | Bootstrap scorer weights from a Phase A checkpoint to start a fresh Phase B run. Loads `model_state_dict` only; ignores `optimizer_state_dict`, `epoch`, `best_val_accuracy`, and (if present) `encoder_state_dict`. Inherits architecture from the loaded checkpoint's `config`. (FR-003a) |
| `--encoder-checkpoint <path>` | `models/price-predictor/transformer/latest.pt` | Source of encoder weights when starting a fresh Phase B run via `--scorer-checkpoint`. Has no effect on Phase A runs (encoder not in graph). Forbidden if explicitly passed on a Phase B `--resume`. (FR-003) |
| `--patience <int>` | `5` | Early-stop training after this many consecutive epochs without a new peak `val_acc`. (FR-011) |
| `--encoder-chunk-size <int>` | `128` | Phase B only: chunk the encoder forward pass over each step's unique cards into pieces of this size, with gradient checkpointing per chunk so peak activation memory is bounded by one chunk. |
| `--max-grad-norm <float>` | `100.0` | Per-parameter-group L2 norm cap applied between backward and optimizer step. Loose by default so it acts as a NaN-spike guard rather than an effective LR throttle; the per-epoch report shows mean and max pre-clip norms so a too-low setting is visible. |

### Removed flags

| Flag | Replacement |
|---|---|
| `--unfreeze-embeddings` | Subsumed by `--embedding-lr`. (FR-002) |
| `--val-interval` | Validation now runs once per epoch unconditionally. (Decision §6) |

## Mutual Exclusivity & Phase Rules

| Rule | Result on violation | Source |
|---|---|---|
| `--resume` and `--scorer-checkpoint` cannot both be set. | Reject with: `"--resume and --scorer-checkpoint are mutually exclusive: --resume continues an existing run; --scorer-checkpoint bootstraps a fresh Phase B run from a Phase A checkpoint."` | FR-003a |
| When `--scorer-checkpoint` is supplied, no architecture flag (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`, `--mlp-hidden`) may be set on the CLI. | Reject with: `"architecture flag <flag-name> conflicts with --scorer-checkpoint; architecture is inherited from the checkpoint's stored config. Omit the flag."` | FR-003a |
| Phase B (`--embedding-lr` non-zero) requires either `--scorer-checkpoint` or `--resume`. | Reject with: `"--embedding-lr <value> requires either --scorer-checkpoint <phaseA>.pt (fresh Phase B kickoff) or --resume <phaseB>.pt (continuing Phase B). Phase B against a randomly-initialized scorer is not supported."` | FR-004a |
| Cross-phase `--resume`: resumed checkpoint's phase must match the current run's phase. | Reject with: `"--resume <path> is a Phase {A,B} checkpoint but --embedding-lr {<nonzero>,0} requests Phase {B,A}. Use --scorer-checkpoint to start a fresh Phase B run from a Phase A checkpoint."` | FR-004 |
| `--encoder-checkpoint` explicitly passed alongside Phase B `--resume`. | Reject with: `"--encoder-checkpoint conflicts with --resume on a Phase B checkpoint, which already carries fine-tuned encoder weights. Omit --encoder-checkpoint."` (Default value never triggers.) | FR-004 |

Detection note: argparse cannot natively distinguish "user passed
`--encoder-checkpoint` with the same value as the default" from "user
omitted the flag". Implementation: register
`--encoder-checkpoint default=None` and resolve to the
`models/price-predictor/transformer/latest.pt` literal *after* the
above checks.

## Help Text Requirements

Every flag listed above MUST appear in `train-scorer --help` with:

- A one-line purpose statement.
- The default value (or `(no default)` for `--scorer-checkpoint`).
- Mutual-exclusivity / phase semantics, where they apply, in plain English.

## Output

The subcommand writes:

| Path | Phase A | Phase B |
|---|---|---|
| `<checkpoint-dir>/latest.pt` | overwritten each epoch | — |
| `<checkpoint-dir>/latest_phaseB.pt` | — | overwritten each epoch |
| `<checkpoint-dir>/best_l<n>_h<n>_s<n>_ff<d>_mlp<d>_lr<f>.pt` | overwritten when `val_acc` improves | — |
| `<checkpoint-dir>/best_phaseB_l<n>_h<n>_s<n>_ff<d>_mlp<d>_lr<f>_emblr<f>.pt` | — | overwritten when `val_acc` improves |

Phase B uses distinct filenames (the `phaseB` prefix on `best_*.pt` and the
`_phaseB` suffix on `latest`) so the Phase A checkpoints used to bootstrap
the run survive intact in the same directory. A regression in Phase B can
be reverted by simply switching back to the Phase A files (SC-003).

Both files contain the keys defined in `data-model.md#phase-a-scorer-checkpoint`:
`model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_accuracy`,
`config` (architecture), `train_config` (full flag dict), and — Phase B
only — `encoder_state_dict`.

## End-of-Epoch Logging Contract

Each epoch boundary the subcommand prints:

- `train_loss`, `train_acc`, `val_loss`, `val_acc`
- `embedding_drift` — Phase B only (FR-012)
- `grad_norms`: per parameter group, formatted as `<name>=mean(<f>)/max(<f>)` where the values are the **pre-clip** L2 norms aggregated across the epoch's batches. Mean tells you the typical step's gradient magnitude; max tells you whether the step's clipping bound (`--max-grad-norm`) is being hit on any batch.
  - Phase A: a single `scorer=mean(...)/max(...)` entry.
  - Phase B: two entries, `scorer=mean(...)/max(...)` and `encoder=mean(...)/max(...)` (single combined L2 norm across each parameter group, FR-012).

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Run completed (training loop reached `--epochs` or `--patience` early-stop). |
| `2` | Validation failure (mutually-exclusive flags, file-not-found, phase mismatch, etc.). |
| `130` | Ctrl-C interrupt. |

## Backwards-Compatibility

- `--unfreeze-embeddings` is removed: invocations that pass it MUST fail with an `unrecognized arguments` error. (Old shell scripts must be updated; per spec § Assumptions, Phase A will be retrained from scratch after this feature ships.)
- `--val-interval` is removed: same handling.
- Pre-feature Phase A checkpoints carry `Adam` optimizer state. Resuming them after the AdamW switch is **not supported** — user is expected to retrain Phase A.

## Examples

```bash
# Phase A from scratch
python -m sealed train-scorer

# Resume Phase A
python -m sealed train-scorer --resume models/sealed/scorer/latest.pt

# Phase B kickoff
python -m sealed train-scorer \
    --scorer-checkpoint models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-5.pt \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --embedding-lr 1e-7

# Resume Phase B (after interruption)
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-5.pt \
    --embedding-lr 1e-7
```
