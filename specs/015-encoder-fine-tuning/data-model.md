# Data Model: Encoder Fine-Tuning (Phase B) for Sealed Scorer

**Feature**: 015-encoder-fine-tuning
**Date**: 2026-04-29

This document captures the entities and on-disk artifacts that this
feature introduces or changes. Domain entities (Python dataclasses /
nn.Modules) live alongside their files; on-disk artifacts have a
versioned format. Every entity is mapped back to one or more spec
requirements.

## Domain Entities

### `TrainScorerConfig` (extended)

Location: `src/sealed/application/train_scorer.py`

Existing dataclass extended with the Phase B flag set. The fields below
are the **canonical CLI surface** for `train-scorer` after this feature
ships.

| Field | Type | Default | Source | Notes |
|---|---|---|---|---|
| `outcomes_path` | `Path` | `output/sealed/match-outcomes.txt` | (existing) | Unchanged. |
| `cards_path` | `Path` | `output/cardsfolder/` | (existing) | Unchanged. Now also used to locate `.txt` files for tokenization in Phase B. |
| `checkpoint_dir` | `Path` | `models/sealed/scorer/` | (existing) | Unchanged. |
| `resume` | `Path \| None` | `None` | (existing) | Phase-locked: see invariants below. |
| `scorer_checkpoint` | `Path \| None` | `None` | **NEW** (FR-003a) | Bootstrap scorer weights from a Phase A checkpoint. Mutually exclusive with `resume`. Must be present whenever `embedding_lr > 0` and `resume is None`. |
| `encoder_checkpoint` | `Path` | `models/price-predictor/transformer/latest.pt` | **NEW** (FR-003) | Source of encoder weights when starting a fresh Phase B run. The default value alone never triggers the FR-004 conflict; only an explicit pass on a Phase B `--resume` does. |
| `epochs` | `int` | `100` | (existing) | Hard upper bound; `--patience` early-stops first. |
| `batch_size` | `int` | `64` | (existing) | Unchanged. |
| `lr` | `float` | `1e-3` → `1e-5` | (existing, default updated per `specs/2026-04-27-encoder-fine-tuning.md` original brief) | Scorer learning rate. |
| `embedding_lr` | `float` | `1e-5` → `0` | (existing, **semantics + default change**, FR-001) | `0` = Phase A (encoder out of training graph). Non-zero = Phase B (encoder in training graph). |
| `n_layers`, `n_heads`, `n_seeds`, `d_ff`, `mlp_hidden`, `dropout` | (existing) | unchanged defaults | (existing) | Architecture flags. **Forbidden when `scorer_checkpoint` is supplied** (FR-003a) — architecture is inherited from the loaded checkpoint's stored config. |
| `patience` | `int` | `5` | **NEW** (FR-011) | Stop after this many consecutive epochs without a new peak `val_acc`. |
| `val_fraction` | `float` | `0.2` | (existing) | Unchanged. |
| `random_seed` | `int` | `42` | (existing) | Drives the deterministic train/val split (FR-011a). |
| `unfreeze_embeddings` | — | — | **REMOVED** (FR-002) | Subsumed by `embedding_lr`. |
| `val_interval` | — | — | **REMOVED** (Decision §6) | Validation is now fixed at once per epoch. |

**Derived "phase" property**:

```python
@property
def phase(self) -> Literal["A", "B"]:
    return "A" if self.embedding_lr == 0 else "B"
```

**Invariants** (validated in `run_train_scorer` before construction):

1. `embedding_lr >= 0` (FR-001).
2. If `phase == "B"`, exactly one of `resume` or `scorer_checkpoint` MUST be set (FR-004a).
3. `resume` and `scorer_checkpoint` are mutually exclusive (FR-003a).
4. If `scorer_checkpoint` is set, no architecture flag (`n_layers`, `n_heads`, `n_seeds`, `d_ff`, `mlp_hidden`) may be passed on the CLI (FR-003a). Detection: compare incoming flag value against the dataclass default; reject if any architecture flag was explicitly set by the user on the same invocation.
5. Cross-phase resume MUST be rejected (FR-004): the resumed checkpoint's phase (presence of `encoder.state_dict`) must match `self.phase`.
6. On Phase B resume, an explicit `--encoder-checkpoint` MUST be rejected (FR-004); the default value never triggers.

### `EncodeCardsConfig` (extended)

Location: `src/sealed/application/encode_cards.py`

The `EncodeCardsConfig` dataclass itself stays minimal (it's the use-case
config, not the CLI assembly). The CLI flags below are wired in
`infrastructure/cli.py::_build_encode_cards_parser`:

| Flag | Type | Default | Source | Notes |
|---|---|---|---|---|
| `--encoder-checkpoint` | `Path` | `models/price-predictor/transformer/latest.pt` | (existing, **renamed** from `--encoder-path`) | Decision §7. |
| `--scorer-checkpoint` | `Path \| None` | `None` | **NEW** (FR-013) | Mutually exclusive with an *explicitly passed* `--encoder-checkpoint` (FR-013 carve-out). |
| `--vocab-path` | `Path` | `models/price-predictor/transformer/vocab.txt` | (existing) | Unchanged. |
| `--cards-path` | `Path` | `output/cardsfolder/` | (existing) | Unchanged. |
| `--clean` | `bool` | `False` | (existing) | Unchanged behavior (idempotent skip when not passed). |

**Invariants** (validated in `run_encode_cards`):

1. `encoder_checkpoint` and `scorer_checkpoint` are mutually exclusive when both were explicitly passed (FR-013). Use the `--encoder-checkpoint default=None` + late-resolve pattern to distinguish "user passed" from "default applied".
2. If `scorer_checkpoint` is supplied and the loaded checkpoint contains no `encoder_state_dict`, reject with a clear error pointing the user at `--encoder-checkpoint` (FR-014).

### `TrainingMetrics` (unchanged; one field re-purposed)

Location: `src/sealed/application/train_scorer.py:71`

| Field | Type | Notes |
|---|---|---|
| `train_losses` | `list[float]` | Existing; appended each epoch. |
| `val_losses` | `list[float]` | Existing; appended each end-of-epoch validation. |
| `val_accuracies` | `list[float]` | Existing; drives `--patience` early stopping. |
| `embedding_drifts` | `list[float]` | Existing field, **re-purposed** (FR-012). One drift value per Phase B epoch (mean L2 distance of post-encoder vectors on the reference batch from their step-0 values). Empty in Phase A. |

### `EpochStats` (extended)

Location: `src/sealed/application/train_scorer.py:79`

| Field | Type | Notes |
|---|---|---|
| `loss` | `float` | Existing. |
| `accuracy` | `float` | Existing. |
| `grad_norms` | `dict[str, float]` | **Replaced** by two named keys: `"scorer"` and (Phase B only) `"encoder"`. The encoder norm is the single combined L2 over the entire encoder parameter group (FR-012 clarification). |

### `ResumeState` (extended)

Location: `src/sealed/application/train_scorer.py:96`

| Field | Type | Notes |
|---|---|---|
| `model` | `SetTransformerScorer` | Existing; constructed from the resumed/bootstrap checkpoint's `ScorerConfig`. |
| `start_epoch` | `int` | Existing. |
| `best_val_accuracy` | `float` | Existing. |
| `optimizer_state` | `dict \| None` | Existing. **Cleared** when the bootstrap path (`scorer_checkpoint`) is used (FR-003a). |
| `encoder_state_dict` | `dict \| None` | **NEW**. Populated when (a) `--resume <phaseB>.pt` (loaded from the resumed checkpoint), or (b) `--scorer-checkpoint <phaseA>.pt` with `embedding_lr > 0` (loaded from the price-predictor `--encoder-checkpoint` file). `None` in Phase A. |
| `phase` | `Literal["A", "B"]` | **NEW**. Determined by presence of `encoder_state_dict` in the *resumed* checkpoint, used solely for the FR-004 phase-lock check against `config.phase`. |

### `_TrainingContext` (extended)

Location: `src/sealed/application/train_scorer.py:111`

| Field | Type | Notes |
|---|---|---|
| `model` | `SetTransformerScorer` | Existing. |
| `optimizer` | `torch.optim.AdamW` | Existing field; **class change** (FR-005a). |
| `embedding_table` | `EmbeddingTable` | Existing. **Role change**: in Phase B its rows are overwritten each batch from encoder output (Decision §8). |
| `train_loader`, `val_loader` | `DataLoader` | Existing. |
| `latest_path`, `best_path`, `start_epoch`, `best_val_accuracy`, `device` | (existing) | Unchanged. |
| `encoder` | `CardPriceTransformerModel \| None` | **NEW**. The trainable encoder, `.to(device)`. `None` in Phase A. |
| `tokenizer` | `MtgTokenizer \| None` | **NEW**. Loaded from `vocab.txt`. `None` in Phase A. |
| `card_token_cache` | `dict[int, tuple[Tensor, Tensor]] \| None` | **NEW**. Maps embedding-table row → (input_ids, attention_mask). Built lazily as new cards appear in batches; persists for the run's lifetime (read-only after first encounter). `None` in Phase A. |
| `reference_batch` | `ReferenceBatch \| None` | **NEW**. See entity below. Captured at step 0 of Phase B; `None` until then and `None` in Phase A. |
| `train_config` | `dict[str, Any]` | **NEW**. The flat dict of training-flag values used to produce this run; persisted in checkpoints (FR-009, Decision §1). |

### `ReferenceBatch` (new)

Location: `src/sealed/application/train_scorer.py` (private dataclass)

| Field | Type | Notes |
|---|---|---|
| `card_indices` | `LongTensor (num_unique,)` | Embedding-table row indices for the unique cards seen in step 0 of Phase B. |
| `input_ids` | `LongTensor (num_unique, max_seq_len)` | Tokenized card text, deduped. |
| `attention_mask` | `LongTensor (num_unique, max_seq_len)` | Mask aligned with `input_ids`. |
| `step0_text_vectors` | `FloatTensor (num_unique, 2 * d_model)` | Encoder text-slice output captured during step 0's forward pass, *before* `optimizer.step()`. Detached, on the training device. |

State transitions:

- *Phase A*: never constructed.
- *Phase B step 0*: constructed from the unique card rows in the first
  training batch. Encoder output captured by `.detach().clone()` after
  the forward pass and before `optimizer.step()`.
- *Subsequent epochs*: `step0_text_vectors` is read-only; the reference
  forward pass each epoch goes through `model.eval()` and computes the
  drift metric.

### `LoadedScorerCheckpoint` (extended)

Location: `src/sealed/infrastructure/scorer_store.py:19`

| Field | Type | Notes |
|---|---|---|
| `model_state_dict` | `dict[str, Any]` | Existing. |
| `optimizer_state_dict` | `dict[str, Any]` | Existing. |
| `epoch` | `int` | Existing. |
| `best_val_accuracy` | `float` | Existing. |
| `config` | `ScorerConfig` | Existing — architecture only. |
| `encoder_state_dict` | `dict[str, Any] \| None` | **NEW** (FR-009). `None` for Phase A checkpoints; populated for Phase B. Used by Phase B `--resume`, by Phase A → Phase B bootstrap (when *not* present, telling caller to load encoder from `--encoder-checkpoint` instead), and by `encode-cards --scorer-checkpoint`. |
| `encoder_config` | `dict[str, Any] \| None` | **NEW** (FR-009). `asdict(TransformerConfig)` for the encoder that produced `encoder_state_dict`. `None` for Phase A checkpoints; populated for Phase B. Required so `train-scorer --resume <phaseB>` can construct a `CardPriceTransformerModel` whose architecture matches the saved weights without depending on the price-predictor `latest.pt` being stable across the run. |
| `train_config` | `dict[str, Any] \| None` | **NEW**. The full training-flag dict (FR-009). `None` for legacy checkpoints; populated going forward. |

### `EmbeddingTable` (modified)

Location: `src/sealed/infrastructure/match_data_loader.py:78`

| API | Status | Notes |
|---|---|---|
| `__init__(vectors, name_to_idx)` | (existing) | Unchanged. |
| `forward(indices)` | (existing) | Unchanged. |
| `deterministic_feature_stats(indices)` | (existing) | Unchanged. |
| `freeze()` / `unfreeze()` / `is_frozen()` | **REMOVED** (FR-002 ripple) | The encoder is the parameter group; the table is no longer a fine-tuning target. The `_build_optimizer` dispatch on `is_frozen()` is replaced by `config.phase`. |
| New: `set_text_vectors(indices, text_vectors)` | **NEW** | Writes the leading `2 * d_model` columns of the rows at `indices` to `text_vectors`, leaving the trailing deterministic-feature slice untouched. The write goes through a non-leaf tensor so autograd flows back through `text_vectors` to the encoder. |

### `CardEncoder` (extended)

Location: `src/sealed/domain/card_encoder.py`

| Method | Status | Notes |
|---|---|---|
| `encode(converted: ConvertedCardText) -> np.ndarray` | (existing) | Unchanged: single-card, `@torch.no_grad()`, includes deterministic-feature concat. Used by `encode-cards`. |
| New: `encode_batch_text(input_ids, attention_mask, *, with_grad: bool) -> Tensor` | **NEW** | Batched, returns the `(B, 2 * d_model)` text-vector slice only (no deterministic concat). `with_grad=True` runs without `torch.no_grad`; this is the call site Phase B training uses. |

The single-card path stays via `@torch.no_grad()` to keep `encode-cards`
inference cheap; the batched path is the Phase B hot loop.

## On-Disk Artifacts

### Phase A scorer checkpoint (`models/sealed/scorer/best_*.pt`)

A `torch.save`-serialized dict produced via the shared
`price_predictor.infrastructure.torch_checkpoint.save_checkpoint`
helper. Keys:

| Key | Phase A | Phase B | Notes |
|---|---|---|---|
| `model_state_dict` | ✓ | ✓ | Scorer (`SetTransformerScorer`). |
| `optimizer_state_dict` | ✓ | ✓ | AdamW (FR-005a). Phase A: single group; Phase B: two groups. |
| `encoder_state_dict` | — | ✓ | Trainable encoder (`CardPriceTransformerModel`). FR-009. |
| `epoch` | ✓ | ✓ | Last completed epoch. |
| `best_val_accuracy` | ✓ | ✓ | Peak val accuracy seen so far (used for `--patience`). |
| `config` | ✓ | ✓ | `asdict(ScorerConfig)` — architecture. |
| `train_config` | ✓ | ✓ | All training flags (FR-009). New for any checkpoint produced after this feature ships. |

The presence of `encoder_state_dict` is the **authoritative phase
indicator** (FR-004). No checkpoint version field; the keyset itself
is the schema.

### `.npz` card embedding (`output/cardsfolder/<letter>/<sanitized_name>.npz`)

Format unchanged: `float32` array of shape `(2 * d_model + FEATURE_COUNT,)`
under key `"embedding"`. The file is rewritten by `encode-cards` after
Phase B; downstream consumers read it identically (FR-015).

### `match-outcomes.txt`, `pools.txt`, `generated-decks.txt`

Unchanged by this feature.

## State Transitions

### Phase A → Phase B kickoff

```
[best_phaseA.pt] + [price-predictor/latest.pt]
        │
        ▼
train-scorer --scorer-checkpoint best_phaseA.pt
             --encoder-checkpoint price-predictor/latest.pt
             --embedding-lr 1e-7
        │
        ▼ build _TrainingContext:
        │  - scorer ← load_state_dict(best_phaseA.model_state_dict)
        │  - encoder ← CardPriceTransformerModel(load latest.pt config)
        │              .load_state_dict(latest.pt state_dict)
        │  - optimizer ← AdamW([scorer @ lr=1e-5, encoder @ lr=1e-7])
        │  - epoch counter ← 0
        │  - best_val_accuracy ← -1.0   (NOT inherited from Phase A — FR-003a)
        │  - reference_batch ← None     (captured at step 0)
        ▼
training loop ...
```

### Phase B `--resume`

```
[mid_phaseB.pt]
        │
        ▼
train-scorer --resume mid_phaseB.pt --embedding-lr 1e-7
        │
        ├── reject if mid_phaseB.encoder_state_dict is None    (FR-004 phase-lock)
        ├── reject if user explicitly passed --encoder-checkpoint  (FR-004 carve-out)
        ▼ build _TrainingContext:
        │  - scorer ← load_state_dict(mid_phaseB.model_state_dict)
        │  - encoder ← CardPriceTransformerModel(mid_phaseB.encoder_config)
        │              .load_state_dict(mid_phaseB.encoder_state_dict)
        │  - optimizer ← AdamW(...) ; load_state_dict(mid_phaseB.optimizer_state_dict)
        │  - epoch counter ← mid_phaseB.epoch + 1
        │  - best_val_accuracy ← mid_phaseB.best_val_accuracy
        ▼
training loop continues ...
```

### `encode-cards --scorer-checkpoint`

```
[best_phaseB.pt]
        │
        ▼
encode-cards --scorer-checkpoint best_phaseB.pt --clean
        │
        ├── reject if best_phaseB.encoder_state_dict is None    (FR-014)
        ├── reject if user also explicitly passed --encoder-checkpoint  (FR-013)
        ▼
        encoder ← CardPriceTransformerModel(best_phaseB.encoder_config)
                  .load_state_dict(best_phaseB.encoder_state_dict)
        │
        ▼ for each .txt under output/cardsfolder/:
        │    .npz ← encoder.encode(converted) (deterministic features concat)
        ▼
[refreshed .npz cache]
```

## Validation Rules Cross-Reference

| Validation rule | Source | Enforcement layer |
|---|---|---|
| `embedding_lr >= 0` | FR-001 | CLI parser (argparse `type=float`) + `run_train_scorer` |
| `--resume` and `--scorer-checkpoint` mutually exclusive | FR-003a | `run_train_scorer` |
| Architecture flags forbidden with `--scorer-checkpoint` | FR-003a | `run_train_scorer` |
| Phase-locked resume | FR-004 | `_resume_or_build_model` (raises) |
| Explicit `--encoder-checkpoint` forbidden on Phase B resume | FR-004 | `run_train_scorer` |
| Phase B requires `--scorer-checkpoint` xor `--resume` | FR-004a | `run_train_scorer` |
| AdamW used in both phases | FR-005a | `_build_optimizer` |
| Per-group max-norm 1.0 clipping | FR-008 | `_train_one_epoch` |
| `encoder_state_dict` saved in Phase B checkpoints, omitted in Phase A | FR-009 | `ScorerStore.save_checkpoint` |
| `encoder_config` saved in Phase B checkpoints, omitted in Phase A | FR-009 | `ScorerStore.save_checkpoint` |
| `train_config` saved in every checkpoint | FR-009 | `ScorerStore.save_checkpoint` |
| Resume precedence: explicit CLI > resumed `train_config` > argparse/dataclass default | FR-010 | `run_train_scorer` (sentinel-default + late-resolve per `contracts/checkpoint-format.md §Resume Precedence`) |
| Validation once per epoch; `--patience` drives early stop | FR-011 | `TrainScorerUseCase.execute` |
| Deterministic train/val split | FR-011a | `_load_dataset` (already correct) |
| `embedding_drift` + encoder grad norm logged each epoch | FR-012 | `_print_epoch_report` |
| `encode-cards` flag mutual exclusivity (carve-out for default) | FR-013 | `run_encode_cards` |
| Phase A checkpoint to `--scorer-checkpoint` rejected | FR-014 | `run_encode_cards` |
| All cards re-encoded (no skip-by-corpus) | FR-015 | (existing `EncodeCardsUseCase` is already corpus-agnostic) |
| Help text for every new/changed flag | FR-016 | argparse registration in `cli.py` |

## Naming Decisions

- `--encoder-checkpoint` (not `--encoder-path`) on **both** subcommands. The existing `encode-cards --encoder-path` is renamed.
- `--scorer-checkpoint` on **both** subcommands. New on each.
- `train_config` is the dict serialization key for the full training-flag set; `config` keeps its existing meaning (architecture-only `ScorerConfig`).
- `encoder_state_dict` (snake_case) is the checkpoint key. It mirrors the existing `model_state_dict` / `optimizer_state_dict` naming.
