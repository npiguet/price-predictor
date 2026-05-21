# Data Model: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Date**: 2026-05-20 | **Spec**: [spec.md](spec.md)

This document enumerates the entities, configs, runtime tensors, and on-disk
artifacts the picker feature introduces or consumes. Each entry lists fields,
relationships, validation rules (FR cross-references), and lifecycle.

## 1. In-memory entities

### 1.1 `PickerConfig` (new, `src/sealed/domain/picker_model.py`)

Architecture record for one picker instance. Frozen for a given run; loaded
from the checkpoint on resume / bootstrap.

| Field | Type | Default | Notes |
|---|---|---|---|
| `embedding_dim` | int | _(required)_ | Width of the `.npz` cache rows the picker consumes. Set at construction time from the loaded scorer's `ScorerConfig.d_model` (FR-002). |
| `d_model` | int | `embedding_dim` | Picker internal width. When equal to `embedding_dim`, no input projection is inserted; otherwise a single `Linear(embedding_dim, d_model)` is inserted ahead of the first SAB (FR-002, § 1 step 2). |
| `n_layers` | int | `4` | Number of SAB layers (FR-021). |
| `n_heads` | int | `8` | Attention heads per layer. `d_model % n_heads == 0` enforced in `__post_init__` (FR-033). |
| `d_ff` | int | `4 * d_model` | Feed-forward dim, computed when unset. |
| `dropout` | float | `0.0` | Dropout in SAB layers. |

**Validation rules**:

- `d_model % n_heads == 0` — raised in `__post_init__` with a clear divisibility error (FR-033).
- `d_model > 0` and `n_layers > 0` — sanity checks.
- `embedding_dim > 0` — implicit (validated upstream when the scorer is loaded).

**Lifecycle**: Created once per training run. Persisted into the checkpoint
payload's `config` field (FR-038). Re-instantiated by `PickerStore.load_checkpoint`
on resume / inference.

### 1.2 `PickerModel` (new, `src/sealed/domain/picker_model.py`)

`nn.Module` subclass. Holds the optional input projection, the SAB stack,
the per-card head, and the auxiliary pool-quality head.

| Submodule | Type | Shape / role |
|---|---|---|
| `input_projection` | `nn.Linear(embedding_dim, d_model)` \| `nn.Identity` | Inserted iff `d_model != embedding_dim` (FR-002). |
| `sab_layers` | `nn.ModuleList[SAB]` | `n_layers` blocks, imported from `sealed.domain.scorer_model` (research §D1). |
| `per_card_head` | `nn.Linear(d_model, 1)` | Applied per-token; produces one logit per pool card (FR-004). |
| `aux_head` | `nn.Linear(d_model, 1)` | Applied to masked mean-pool of token outputs; produces one scalar per pool (FR-005, § 1.2). |

**Forward signature**:
`forward(pool_cards: (B, max_N, embedding_dim), pool_mask: (B, max_N)) -> (logits: (B, max_N), aux_pred: (B,))`.

The `pool_mask` boolean indicates real-card positions (True) vs. padding
(False), following `SetTransformerScorer`'s convention.

**Lifecycle**: Constructed from `PickerConfig`. Moved to the training device.
State-dict-only persistence (FR-037).

### 1.3 `TrainPickerConfig` (new, `src/sealed/application/train_picker.py`)

Dataclass capturing every CLI knob for `train-picker`. Mirror of
`TrainScorerConfig` (`src/sealed/application/train_scorer.py:40`).

| Field | Type | Default | FR ref | Resumable |
|---|---|---|---|---|
| `pools_path` | `Path` | _(required)_ | FR-010, FR-021 | Yes |
| `scorer_checkpoint` | `Path` | `models/sealed/scorer/latest.pt` | FR-009, FR-036 | Yes |
| `auditor_scorer_checkpoint` | `Path \| None` | `None` | FR-021, FR-030 | Yes |
| `cards_path` | `Path` | `output/cardsfolder/` | FR-001, FR-021 | Yes |
| `checkpoint_dir` | `Path` | `models/sealed/picker/` | FR-037 | Yes |
| `resume` | `Path \| None` | `None` | FR-022 | n/a |
| `picker_checkpoint` | `Path \| None` | `None` | FR-023 | n/a |
| `d_model` | int \| None | `None` (= embedding_dim) | FR-002 | Forbidden on resume/bootstrap (FR-022, FR-023) |
| `n_layers` | int | `4` | FR-021 | Forbidden on resume/bootstrap |
| `n_heads` | int | `8` | FR-021 | Forbidden on resume/bootstrap |
| `d_ff` | int \| None | `None` (= 4 * d_model) | FR-021 | Forbidden on resume/bootstrap |
| `dropout` | float | `0.0` | FR-021 | Forbidden on resume/bootstrap |
| `aux_weight` | float | `0.1` | FR-021, § 3.4, § 1.2 | Yes |
| `batch_size` | int | `16` | FR-021 | Yes |
| `n_samples` | int | `64` | FR-011, FR-021 | Yes |
| `temperature` | float | `1.0` | FR-021, § 3.2 | Yes |
| `entropy_coef` | float | `0.01` | FR-016, FR-021 | Yes |
| `entropy_decay_after` | int | `5` | FR-016, FR-021 | Yes |
| `lr` | float | `3e-4` | FR-017, FR-021 | Yes |
| `max_grad_norm` | float | `1.0` | FR-017, FR-021 | Yes |
| `epochs` | int | `100` | FR-021 | Yes |
| `val_fraction` | float | `0.2` | FR-018, FR-021 | Yes |
| `patience` | int | `10` | FR-020, FR-021 | Yes |
| `kl_coef` | float | `0.0` | FR-025, FR-021 | Yes |

**Constants** (not flags):

- `random_seed = 42` — hardcoded, no CLI flag (Clarify Q5, FR-018). Used for
  weight init, pool shuffle, deck sampling, train/val split.

**Mutual exclusivity rules** (enforced in `cli.run_train_picker`):

- `resume` and `picker_checkpoint` mutually exclusive (FR-024).
- Architecture flags (`n_layers`, `n_heads`, `d_ff`, `dropout`, `d_model`)
  forbidden alongside either `resume` or `picker_checkpoint` (FR-022, FR-023).
- `kl_coef != 0` requires `picker_checkpoint` to be set (FR-025).
- `scorer_checkpoint` must exist at startup or be omitted (default path must
  exist if no flag passed) (FR-036).

### 1.4 `PickDecksConfig` (new, `src/sealed/application/pick_decks.py`)

Dataclass for the `pick-decks` CLI.

| Field | Type | Default | FR ref |
|---|---|---|---|
| `pools_path` | `Path` | _(required)_ | FR-026 |
| `picker_checkpoint` | `Path` | `models/sealed/picker/latest.pt` | FR-026 |
| `cards_path` | `Path` | `output/cardsfolder/` | FR-026 |
| `label` | `str` | _(required)_ | FR-027 |
| `output` | `Path` | `output/sealed/generated-decks.txt` | FR-026 |
| `resume` | `bool` | `False` | FR-028 |

**Validation**:

- `label` rejected if it contains `;`, `\|`, or whitespace (reuses
  `_parse_label` from `src/sealed/infrastructure/cli.py:35`).
- Width compatibility between picker checkpoint and `.npz` cache enforced
  at load time (FR-035).

### 1.5 Runtime tensors (training step)

These are not persisted; they exist for the duration of one training step.
Documented because their shapes anchor the training-loop code and the unit
tests.

| Tensor | Shape | Notes |
|---|---|---|
| `pool_embeddings` | `(B, max_N, embedding_dim)` | One sub-batch of `B` pools, padded to the longest. Cached on GPU. |
| `pool_mask` | `(B, max_N)` bool | True = real pool card. |
| `is_land_mask` | `(B, max_N)` bool | True = `is_land_embedding(card)`. Used by the sampler to bucket picks. |
| `logits` | `(B, max_N)` float | Picker's per-card output. |
| `aux_pred` | `(B,)` float | Picker's per-pool aux scalar. |
| `sampled_picks` | `(B*S, max_picks)` long | Pool indices in pick order, per sampled deck. `max_picks` is the per-batch max chosen-count (between 23 and 23+max nonbasic lands in any pool). |
| `picked_mask` | `(B*S, max_picks)` bool | True = real pick (some rows finish in 23 picks; trailing positions masked). |
| `sampled_deck_cards` | `(B*S, max_picks, embedding_dim)` | Same as `sampled_picks` but with embeddings filled in for the scorer forward. |
| `sampled_deck_mask` | `(B*S, max_picks)` bool | Same shape as `picked_mask`; passed to scorer as `key_padding_mask`. |
| `rewards` | `(B, S)` float | Scorer scores after un-flatten. |
| `baselines` | `(B,)` float | `rewards.mean(dim=1)`. |
| `advantages` | `(B, S)` float | `rewards - baselines.unsqueeze(1)`, detached (FR-014); optionally divided by the per-pool reward std (GRPO normalization) when `normalize_advantage` is set. |
| `log_probs` | `(B, S)` float | Plackett-Luce log-probs, differentiable in `logits` (FR-014). |
| `entropy` | `(B,)` float | Per-pool entropy of the picker's softmax over `pool_mask`-valid positions. |

## 2. On-disk artifacts

### 2.1 Picker checkpoint payload (`models/sealed/picker/{*.pt}`)

Written by `PickerStore.save_checkpoint`. Schema (per FR-038, mirrors
`LoadedScorerCheckpoint`):

| Key | Type | Notes |
|---|---|---|
| `model_state_dict` | dict | Picker weights only. No scorer / encoder / auditor weights. |
| `optimizer_state_dict` | dict | AdamW state. |
| `epoch` | int | Current epoch counter (0-indexed). |
| `best_val_reward` | float | Best validation reward seen so far. `-inf` for the initial checkpoint on a fresh run. |
| `config` | `PickerConfig` dict | Architecture record, used by `load_checkpoint` to reconstruct the model (FR-022, FR-023). |
| `train_config` | dict | JSON-friendly serialization of `TrainPickerConfig` (used by resume precedence). |

**Files written** (FR-037):

- `latest.pt` — overwritten each epoch; the resume point.
- `best_{timestamp}.pt` — one per run (`{timestamp}` fixed at run start), overwritten in place whenever val reward exceeds the previous best. No per-epoch snapshot files are written.

### 2.2 Pools file (input; existing format, unchanged)

Read by `parse_pools` (`src/sealed/infrastructure/pool_file_reader.py:32`).
Format: `SET_CODE;Card1|Card2|...|CardN` per line. Existing format from
spec 011; no schema change.

**Used by**: `train-picker` (FR-010), `pick-decks` (FR-026).

### 2.3 Generated-decks file (output; existing format, unchanged)

Written by `pick-decks`, line-by-line, identical schema to `build-decks`
output:

```
LABEL;SET_CODE;Card1|Card2|...|Card40
```

Where `LABEL` is the `--label` argument verbatim (FR-027). One line per
input pool (FR-026, SC-005). 40 cards per line: 23 spells + up to 17
nonbasic lands the picker selected + basic lands from `compute_basic_lands`
(FR-006, FR-007).

**Consumed by**: `match-outcomes --side-a-decks`, `match-outcomes
--side-b-decks` (unchanged behavior; see CLAUDE.md "Generated-decks file
format").

## 3. Entities reused unchanged

The following entities from the existing sealed package are consumed by
this feature without modification (research § "Adjacent prior art"):

| Entity | File | Use |
|---|---|---|
| `SetTransformerScorer` | `src/sealed/domain/scorer_model.py:78` | Loaded frozen, in `.eval()`. Reward source (training scorer); optional second instance (auditor scorer). |
| `ScorerConfig` | `src/sealed/domain/scorer_model.py:14` | Read for `d_model` (= picker's `embedding_dim` default). |
| `SAB` | `src/sealed/domain/scorer_model.py:26` | Imported into `picker_model.py` for the trunk. |
| `LoadedScorerCheckpoint`, `ScorerStore` | `src/sealed/infrastructure/scorer_store.py` | Loads training and auditor scorers. |
| `ConvertedCardLocator` | `src/sealed/infrastructure/converted_card_locator.py:27` | Per-card `.npz` lookup; per-card `.txt` lookup for `compute_basic_lands`. |
| `parse_pools` | `src/sealed/infrastructure/pool_file_reader.py:32` | Reads the input pools file. |
| `compute_basic_lands` | `src/sealed/domain/manabase.py:20` | Fills basic lands to 40 cards (FR-007). |
| `is_land_embedding` | `src/sealed/domain/card_embedding_layout.py:65` | Partitions chosen cards into spells / nonbasic lands during the deterministic walk (FR-008). |
| `FEATURE_COUNT`, `COLOR_FLAGS`, `MANA_VALUE`, `POWER`, `TOUGHNESS` | `src/sealed/domain/card_embedding_layout.py` | Distributional summary aggregations (FR-032; research §D7). |
| `_parse_label` | `src/sealed/infrastructure/cli.py:35` | Validates `pick-decks --label` (FR-027). |
| `_count_complete_lines_and_truncate_partial` | `src/sealed/application/build_decks.py:26` | Implements `pick-decks --resume` append-and-skip (FR-028). |
| `torch_checkpoint.save_checkpoint` / `load_checkpoint` | `src/price_predictor/infrastructure/torch_checkpoint.py:14, 20` | Underlying `.pt` serialization. |

## 4. Entities introduced

| Entity | File | Role |
|---|---|---|
| `PickerConfig` | `src/sealed/domain/picker_model.py` (new) | Architecture record (1.1). |
| `PickerModel` | `src/sealed/domain/picker_model.py` (new) | The policy model (1.2). |
| `TrainPickerConfig` | `src/sealed/application/train_picker.py` (new) | Training-run CLI/config (1.3). |
| `TrainPickerUseCase` | `src/sealed/application/train_picker.py` (new) | Orchestrates one training run; mirrors `TrainScorerUseCase`. |
| `PickDecksConfig` | `src/sealed/application/pick_decks.py` (new) | Inference CLI/config (1.4). |
| `PickDecksUseCase` | `src/sealed/application/pick_decks.py` (new) | One pool → one 40-card deck loop; mirrors `BuildDecksUseCase`. |
| `LoadedPickerCheckpoint` | `src/sealed/infrastructure/picker_store.py` (new) | Typed `.pt` load result, mirrors `LoadedScorerCheckpoint`. |
| `PickerStore` | `src/sealed/infrastructure/picker_store.py` (new) | `.pt` persistence wrapper, mirrors `ScorerStore`. |

No new package boundary; everything sits inside the existing `sealed`
package. No changes to `price_predictor`.

## 5. State transitions

The picker has no in-process state machine (no mode flags, no streaming
state). The lifecycle is:

```
TrainPickerConfig (from CLI)
    │
    ▼
TrainPickerUseCase.execute()
    │
    ├─ load scorer (frozen)
    ├─ load auditor scorer (frozen, optional)
    ├─ load .npz cache (frozen)
    ├─ load pool file
    ├─ build PickerModel (or load via --resume / --picker-checkpoint)
    │
    ▼
for epoch in epochs:
    for batch in train_pools:
        forward → sample → score → loss → backward → step
    val_reward, audit_corr, distrib_summaries = validate(val_pools)
    persist(latest.pt; best_{timestamp}.pt if new best)
    early-stop check
    │
    ▼
done — TrainPickerResult
```

`pick-decks` is even simpler:

```
PickDecksConfig (from CLI)
    │
    ▼
PickDecksUseCase.execute()
    │
    ├─ load picker (frozen)
    ├─ load .npz cache (frozen)
    ├─ load pool file
    │
    ▼
for (set_code, pool_names) in pools:
    forward → deterministic walk → manabase fill → write line
    │
    ▼
done — written count
```

No background threads, no async I/O, no checkpointed sub-state inside an
epoch.
