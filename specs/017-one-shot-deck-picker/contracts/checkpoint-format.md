# Checkpoint Format: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Spec**: [spec.md](../spec.md)

The picker persists training-run state as PyTorch `.pt` files in
`models/sealed/picker/`. Schema mirrors `LoadedScorerCheckpoint`
(`src/sealed/infrastructure/scorer_store.py:20`) with the metric renamed
to reflect val-reward semantics.

## File layout

```
models/sealed/picker/
├── latest.pt                 # overwritten each epoch (resume point)
└── best_{timestamp}.pt       # one per run; overwritten when a new val-reward best is reached
```

`{timestamp}` is fixed at training-run start, so each run produces exactly
two files: a `latest.pt` (the resume point, refreshed every epoch) and a
`best_{timestamp}.pt` (the best-by-validation-reward checkpoint of that run,
overwritten in place as the run finds new bests). The run-stamped name means
concurrent or sequential runs never clobber each other's best checkpoint.
No per-epoch snapshot files are written.

Both files use the same payload schema; the selector (best vs. latest) is
the filename, and the contents are whatever was last saved into that file.

## Payload schema

Stored via `torch_checkpoint.save_checkpoint`
(`src/price_predictor/infrastructure/torch_checkpoint.py:14`), which adds the
`config` key by `asdict`-ing the supplied dataclass. The resulting `dict[str,
Any]` payload, as produced by `PickerStore.save_checkpoint`:

| Key | Type | Required | Description |
|---|---|---|---|
| `model_state_dict` | `dict[str, Tensor]` | Yes | Picker weights only. Includes the input projection (when present), the SAB stack, the per-card head, and the aux head. No scorer / encoder / auditor weights are embedded (FR-038). |
| `optimizer_state_dict` | `dict[str, Any]` | Yes | AdamW state — momentum / variance buffers and per-group hyperparameter snapshot. Restored by `--resume`; discarded by `--picker-checkpoint` (FR-022, FR-023). |
| `epoch` | int | Yes | Number of completed epochs at save time (0-indexed). The next epoch a resumed run would start is `epoch + 1`. |
| `best_val_reward` | float | Yes | The highest mean validation reward observed in this run, including the current epoch if it set a new best. `-inf` (or a very negative sentinel) on a fresh run's epoch 0 if the metric has not yet been computed; in normal saves this is always a real value. |
| `config` | dict (serialized `PickerConfig`) | Yes | Architecture record: `embedding_dim`, `d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`. Used by `load_checkpoint` to reconstruct the model. Architecture is the source-of-truth for resume / bootstrap (FR-022, FR-023). |
| `train_config` | dict | Yes | JSON-friendly serialization of `TrainPickerConfig` (all fields, with `Path` values stringified). Used by resume precedence — when a resumed run omits a resumable flag, the value falls back to this dict before falling back to the dataclass default. Mirrors the scorer's `train_config` payload field (`scorer_store.py:36`). |

## Resume precedence

When `--resume <checkpoint>` is set, `cli.run_train_picker` reads
`train_config` from the checkpoint and applies the same three-tier
precedence used by `train-scorer`:

1. **Explicit CLI flag**: if the user passed `--batch-size 32`, use 32.
2. **Resumed `train_config`**: if the CLI omitted the flag, fall back to
   the value recorded in the saved `train_config`.
3. **Dataclass default**: if neither source has a value (e.g., the
   checkpoint was saved by an earlier version that lacked the field), fall
   back to the `TrainPickerConfig` field default.

Architecture fields (`d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`)
bypass this rule — they are inherited from the checkpoint's `config` field
(not `train_config`), and explicit CLI architecture flags are rejected
upstream (FR-022, FR-023). The two configs serve different roles:

- `config` is the architecture record. It defines model shape. Immutable
  across resumes.
- `train_config` is the training-run hyperparameter record. It defines how
  training proceeds. CLI flags can override individual fields.

## Width validation

At training-run startup, `train-picker` validates the picker checkpoint's
`config.embedding_dim` against the loaded `.npz` cache's row width (read
from the first cache file by `ConvertedCardLocator.load_embedding(...).shape[-1]`).
On mismatch, raises a `ValueError` with the same shape as `_check_scorer_width`
(`train_scorer.py:430`):

```
--resume <path> expects {N}-wide card embeddings, but the .npz cache under
<cards-path> is {M}-wide. Re-run `sealed encode-cards` with the encoder
this picker was trained on, or start a fresh picker (drop --resume).
```

(FR-034.)

Inference path (`pick-decks`) performs the same check against the picker
checkpoint loaded via `--picker-checkpoint` and the `.npz` cache (FR-035).

## Backwards / forwards compatibility

Not applicable in v1 — this is the first picker checkpoint format. Future
schema changes should follow the same evolution rule the scorer uses: a
checkpoint without a new field is loaded with the field's dataclass default,
and a checkpoint with an unrecognized field raises in
`PickerConfig(**raw_config)` (which is the desired fail-fast behavior).

## Cross-references

- `ScorerStore.save_checkpoint` (`src/sealed/infrastructure/scorer_store.py:42`) — pattern source.
- `train_scorer.py:_build_train_config` (`src/sealed/application/train_scorer.py:955`) — pattern source for `train_config` flattening.
- `train_scorer.py:_check_scorer_width` (`src/sealed/application/train_scorer.py:430`) — pattern source for the width-mismatch error message.
