# Artifact Contract: Sealed Scorer Checkpoint (`.pt`)

**Feature**: 015-encoder-fine-tuning

Contract for the sealed scorer checkpoint files written by `train-scorer`
and read by `train-scorer --resume`, `train-scorer --scorer-checkpoint`,
`evaluate-scorer`, `build-decks`, and (Phase B only) `encode-cards
--scorer-checkpoint`.

## File Format

A single `torch.save`-serialized Python `dict` produced via
`price_predictor.infrastructure.torch_checkpoint.save_checkpoint`. Top-level
keys:

| Key | Type | Phase A | Phase B | Source |
|---|---|---|---|---|
| `model_state_dict` | `dict[str, Tensor]` | required | required | Scorer (`SetTransformerScorer`). |
| `optimizer_state_dict` | `dict` | required | required | `AdamW`. Phase A: single param group. Phase B: two groups (scorer, encoder). |
| `epoch` | `int` | required | required | Last completed epoch (0-indexed in `start_epoch + epoch_idx`). |
| `best_val_accuracy` | `float` | required | required | Peak validation accuracy seen so far. |
| `config` | `dict` | required | required | `asdict(ScorerConfig)` — architecture only (fields: `d_model`, `n_layers`, `n_heads`, `n_seeds`, `d_ff`, `mlp_hidden`, `dropout`). |
| `train_config` | `dict[str, Any]` | required | required | All `train-scorer` CLI flag values (FR-009). |
| `encoder_state_dict` | `dict[str, Tensor]` | absent | required | `CardPriceTransformerModel.state_dict()` — token embedding table + position embedding + encoder stack + output dropout. (FR-009) |
| `encoder_config` | `dict[str, Any]` | absent | required | `asdict(TransformerConfig)` for the encoder that produced `encoder_state_dict`. Required so resumes and `encode-cards --scorer-checkpoint` can reconstruct a `CardPriceTransformerModel` whose architecture matches the saved weights, without relying on the price-predictor `latest.pt` being stable. (FR-009) |

The presence of the `encoder_state_dict` key is the **authoritative
phase indicator** (FR-004). Code paths that need to know the phase MUST
test for the key, not look at flag values. `encoder_config` MUST be
present whenever `encoder_state_dict` is present.

## `train_config` Schema

The `train_config` dict is a flat mapping of every `train-scorer` CLI
flag's resolved value (after argparse defaults and CLI overrides
applied). Keys mirror the `TrainScorerConfig` field names (snake_case):

```python
{
  "outcomes_path": "output/sealed/match-outcomes.txt",
  "cards_path": "output/cardsfolder/",
  "checkpoint_dir": "models/sealed/scorer/",
  "epochs": 100,
  "batch_size": 64,
  "lr": 1e-5,
  "embedding_lr": 1e-7,           # 0.0 in Phase A
  "n_layers": 6,
  "n_heads": 4,
  "n_seeds": 4,
  "d_ff": 1088,
  "mlp_hidden": 256,
  "dropout": 0.2,
  "patience": 5,
  "val_fraction": 0.2,
  "random_seed": 42,
  "scorer_checkpoint": "models/sealed/scorer/best_phaseA.pt",  # or None
  "encoder_checkpoint": "models/price-predictor/transformer/latest.pt",
  "resume": null,                   # path or null
}
```

`Path`-typed values are serialized as strings. `None` becomes JSON `null`
(or, in the Python dict, plain `None`).

`train_config` is **advisory** — it lets a reader see how the run was
launched. It is *not* re-loaded as the architecture source; that's still
`config`.

## Resume Precedence (FR-010)

When `train-scorer --resume <path> [other flags]` runs, each training
flag's effective value for the current run is resolved by this priority
order:

1. **Explicit CLI argument** wins. If the user passed the flag on the
   command line, that value is used.
2. Otherwise, **the resumed checkpoint's `train_config`** wins. The user's
   prior choice survives the resume so that
   `train-scorer --resume <ckpt>` with no other flags continues training
   exactly as it was interrupted.
3. Otherwise, **the argparse / dataclass default** wins. (Fallback for
   pre-feature checkpoints with no `train_config` field, and for any
   future flag added after the resumed checkpoint was saved.)

The current run's resolved values become the new `train_config` written
to subsequent checkpoints.

Implementation detail: argparse cannot natively distinguish "user passed
the flag with the default value" from "user omitted the flag". The
implementation MUST register every resumable training flag with
`default=None` (sentinel) and apply the precedence above in
`run_train_scorer` *after* loading the resumed checkpoint, falling back
to the dataclass default when `train_config` is absent or doesn't carry
the field. This is the same late-resolve pattern used for
`--encoder-checkpoint`'s mutual-exclusivity carve-out.

Architecture flags (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`,
`--mlp-hidden`, `--dropout`) are NOT subject to this precedence:
architecture is loaded directly from the resumed checkpoint's `config`
(scorer) and `encoder_config` (encoder) fields, and any explicit CLI
architecture flag on `--resume` MUST be rejected (a mismatched
architecture would fail `load_state_dict`).

Architecture flags (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`,
`--mlp-hidden`, `--dropout`) MUST NOT be changed on resume; the
checkpoint's `config` is loaded directly into the scorer constructor and
mismatched architecture would fail `load_state_dict`. The CLI rejects
explicit architecture flags on `--resume` for the same reason it rejects
them on `--scorer-checkpoint`.

## Phase A → Phase B Bootstrap

When `train-scorer --scorer-checkpoint <phaseA>.pt [...]` runs:

| Field | Behavior |
|---|---|
| `model_state_dict` | Loaded into the scorer. |
| `config` | Loaded; the scorer is constructed with this architecture (CLI architecture flags rejected). |
| `optimizer_state_dict` | **Ignored**. New optimizer is built from scratch. |
| `epoch` | **Ignored**. New run starts at epoch 0. |
| `best_val_accuracy` | **Ignored**. New run resets to `-1.0`. |
| `train_config` | **Ignored** (the new run defines its own). |
| `encoder_state_dict` (if present, i.e. user passed a Phase B checkpoint) | **Ignored**. Encoder weights for the fresh run come from `--encoder-checkpoint` per FR-003. |

## Phase B → `encode-cards`

When `encode-cards --scorer-checkpoint <phaseB>.pt` runs:

| Field | Behavior |
|---|---|
| `encoder_state_dict` | Loaded into a fresh `CardPriceTransformerModel`. **Required** — absence = Phase A checkpoint = error per FR-014. |
| `encoder_config` | Used to construct the `CardPriceTransformerModel` so its architecture matches the saved weights. **Required** alongside `encoder_state_dict`. |
| All other fields | Ignored. |

The `CardPriceTransformerModel` is constructed with the
`TransformerConfig` reconstructed from the checkpoint's `encoder_config`
field, and its weights are populated from `encoder_state_dict`. This
makes the Phase B checkpoint self-contained: `encode-cards` no longer
depends on the price-predictor `latest.pt` being unchanged since the
Phase B run produced this checkpoint. Symmetric with `train-scorer
--resume <phaseB>`, which loads the encoder the same way.

## Backwards-Compatibility

| Pre-feature checkpoint state | Reader behavior |
|---|---|
| Lacks `encoder_state_dict`, lacks `encoder_config`, lacks `train_config` | Treated as Phase A. `train_config` is reconstructed from the resumed CLI invocation (or left empty). |
| Carries `Adam` optimizer state | **Not supported.** Spec § Assumptions: Phase A is retrained from scratch after this feature ships. |

There is no checkpoint-format version field. The presence/absence of
named keys is the schema. Future format changes that need a version
field can add one without breaking this contract — readers that don't
know the field will continue to read the existing keys.
