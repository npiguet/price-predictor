# CLI Contracts: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Spec**: [spec.md](../spec.md)

Two new subcommands are added to the `sealed` CLI (`python -m sealed`).
Existing subcommands are unchanged.

## `sealed train-picker`

Train a one-shot picker from scratch using REINFORCE against a frozen scorer.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--pools-path` | path | _(required)_ | Pre-generated pools file (produced by `sealed generate-pools`). Format: `SET_CODE;Card1\|...\|CardN` per line. (FR-010, FR-021) |
| `--scorer-checkpoint` | path | `models/sealed/scorer/latest.pt` | Frozen scorer used as the reward function. Default must exist or the run fails fast (FR-036). |
| `--auditor-scorer-checkpoint` | path | _(none)_ | When set, enables the FR-030 cross-scorer audit: validation decks are also scored with this checkpoint and the rank correlation between training and auditor scores is logged each epoch. Must have the same `.npz` cache width as the training scorer (validated at startup). |
| `--cards-path` | path | `output/cardsfolder/` | Directory of `.npz` card-embedding files. |
| `--checkpoint-dir` | path | `models/sealed/picker/` | Output directory for `latest.pt` and `best_{timestamp}.pt`. |
| `--resume` | path | _(none)_ | Continue a stopped run from this checkpoint. Loads picker weights, optimizer state, epoch counter, and `best_val_reward`. Architecture flags forbidden when set (FR-022). Mutually exclusive with `--picker-checkpoint` (FR-024). |
| `--picker-checkpoint` | path | _(none)_ | Bootstrap a fresh run from this checkpoint's picker weights only; optimizer state, epoch counter, and validation metadata discarded. Architecture flags forbidden (FR-023). Required when `--kl-coef` is non-zero (FR-025). Mutually exclusive with `--resume`. |
| `--d-model` | int | _(derived = `embedding_dim`)_ | Picker internal width. When unset, defaults to the cache embedding width and no input projection is inserted. When set to a value other than `embedding_dim`, a single `Linear(embedding_dim, d_model)` projection is inserted ahead of the first SAB (FR-002). Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--n-layers` | int | `4` | Number of SAB layers (FR-021). Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--n-heads` | int | `8` | Attention heads per SAB. `d_model` must be divisible by this (FR-033). Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--ff-dim` | int | `4 * d_model` | Feed-forward dim in SAB. Computed from the resolved `d_model` when unset. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--dropout` | float | `0.0` | Dropout in SAB layers. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--aux-weight` | float | `0.1` | Coefficient on the auxiliary pool-quality MSE loss (FR-005, § 1.2, § 3.4). Setting to `0` disables the aux loss while keeping the head parameters in the model. |
| `--normalize-advantage` / `--no-normalize-advantage` | bool | `False` | GRPO-style per-pool advantage normalization (divide the centered reward by the per-pool reward std). Applies only to `--objective reinforce`. Resumable. |
| `--objective` | `reinforce` \| `topk` | `reinforce` | Training objective (FR-039). `reinforce` = advantage-weighted policy gradient with a per-pool baseline; `topk` = reward-ranked / best-of-N (keep the `--topk` highest-reward decks per pool and maximize their log-prob, no baseline). Resumable and **not** architecture-locked — a `reinforce` checkpoint may be resumed/warm-started under `topk`. |
| `--topk` | int | `16` | Decks kept per pool under `--objective topk` (FR-039). Must satisfy `1 ≤ topk < --n-samples`; `topk == --n-samples` fails fast (no selection pressure). Smaller = greedier. Resumable. |
| `--batch-size` | int | `16` | Pools per gradient step. |
| `--n-samples` | int | `64` | Sampled decks per pool per step (FR-011). |
| `--temperature` | float | `1.0` | Softmax temperature for sampling (FR-021, § 3.2). |
| `--entropy-coef` | float | `0.01` | Initial entropy coefficient (FR-016, FR-021). |
| `--entropy-decay-after` | int | `5` | Consecutive monotonically-improving val-reward epochs required before entropy starts decaying from its initial value (FR-016). |
| `--lr` | float | `3e-4` | AdamW learning rate (FR-017). |
| `--max-grad-norm` | float | `1.0` | Per-parameter-group L2-norm cap (FR-017). |
| `--epochs` | int | `100` | Maximum epochs. One epoch = one shuffled pass through the training portion of `--pools-path` (FR-018). |
| `--val-fraction` | float | `0.2` | Validation slice: first this fraction of the pools file. Excluded from training shuffles and reused identically across epochs (FR-018). |
| `--patience` | int | `10` | Early-stop after this many epochs without validation-reward improvement (FR-020). |
| `--kl-coef` | float | `0.0` | KL penalty coefficient against `--picker-checkpoint`'s frozen reference distribution. `0.0` disables (default for REINFORCE-from-random). Non-zero requires `--picker-checkpoint` (FR-025). |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Training completed successfully (early-stop or `--epochs` reached). |
| `2` | Argument / configuration error (mutually-exclusive flags violated, architecture flag passed with `--resume` or `--picker-checkpoint`, scorer checkpoint missing, etc.). All `2` exits print a clear error message to stderr identifying the offending flag. |
| `6` | Architecture validation error (e.g., `n_heads` does not divide `d_model`) (FR-033). |
| `130` | Interrupted by Ctrl-C. |

### Required output

On every epoch, exactly one progress line to stdout containing at minimum:

- `epoch={int}` (0-indexed).
- `policy_loss={float}`, `entropy_loss={float}`, `aux_loss={float}` — loss decomposition (FR-029).
- `val_reward={float}` — mean reward on the validation slice (FR-019, FR-029).
- `audit_corr={float}` — only when `--auditor-scorer-checkpoint` is set (FR-030).
- Distributional summaries from FR-032: `colors_mean`, `creatures_mean`, `type_creature_share`, `cmc_hist` (5 bins).

On run completion: one summary line listing the best epoch number, best val reward, and the path to `best_{timestamp}.pt`.

### Resume precedence

The same precedence rule as `train-scorer` (`src/sealed/infrastructure/cli.py:_RESUMABLE_FLAG_NAMES`):

1. Explicit CLI flag value.
2. Value from the resumed checkpoint's `train_config`.
3. Dataclass default in `TrainPickerConfig`.

Architecture flags (`--d-model`, `--n-layers`, `--n-heads`, `--ff-dim`, `--dropout`) bypass this and are forbidden when either `--resume` or `--picker-checkpoint` is set — architecture is inherited from the checkpoint's stored `PickerConfig`.

### Error conditions

| Condition | Behavior | FR |
|---|---|---|
| `--resume` and `--picker-checkpoint` both set | Exit 2 with "mutually exclusive" message | FR-024 |
| Architecture flag set alongside `--resume` or `--picker-checkpoint` | Exit 2 naming the offending flag | FR-022, FR-023 |
| `--kl-coef != 0` without `--picker-checkpoint` | Exit 2 directing user to pass `--picker-checkpoint` | FR-025 |
| `--scorer-checkpoint` path does not exist (default or explicit) | Exit 2 directing user to train a scorer first or pass an explicit path | FR-036 |
| `--auditor-scorer-checkpoint` width disagrees with `.npz` cache | Exit 2 with width-mismatch error citing both widths and the cache path | FR-035 (analogue) |
| `--n-heads` does not divide resolved `d_model` | Exit 6 with divisibility error before any model construction | FR-033 |
| `.npz` cache width disagrees with checkpoint loaded via `--resume` / `--picker-checkpoint` | Exit 2 at startup with width-mismatch error | FR-034 |
| Pools file does not exist or is unparseable | Exit 2 with file path + parse error | (general) |

## `sealed pick-decks`

The inference counterpart to `sealed build-decks`. Reads a pools file, runs
the picker once per pool, fills basic lands via `compute_basic_lands`, writes
a `generated-decks.txt` file consumable by `match-outcomes`.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--pools-path` | path | _(required)_ | Input pools file. Same format as `build-decks --pools-path`. |
| `--picker-checkpoint` | path | `models/sealed/picker/latest.pt` | Picker weights (FR-026). |
| `--cards-path` | path | `output/cardsfolder/` | Directory of `.npz` card-embedding files (FR-026). |
| `--label` | string | _(required)_ | Generation-method tag written as the first column of every output line (FR-027). Must not contain `;`, `\|`, or whitespace (reuses `_parse_label`). |
| `--output` | path | `output/sealed/generated-decks.txt` | Output deck file. |
| `--resume` | flag | `False` | Append-and-skip semantics: count complete lines already in `--output`, skip that many pools from the front of `--pools-path`, append remaining decks. Without this flag the output file is truncated (FR-028). |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All input pools processed; output file complete. |
| `2` | Argument error or picker-checkpoint / cache width mismatch (FR-035). |

### Output format

Identical to `build-decks` output (CLAUDE.md "Generated-decks file format"):

```
LABEL;SET_CODE;Card1|Card2|...|Card40
```

One line per input pool. Exactly 40 cards: 23 spells (FR-006) + the nonbasic
lands the picker selected before the spell quota filled (FR-006) + basic
lands from `compute_basic_lands` to fill to 40 (FR-007). Pools with fewer
than 23 spells in the embeddable set are silently skipped (matches the
`build-decks` behavior).

### Stdout

Progress lines every ~1% of the input pool count (matches `build-decks`
cadence in `build_decks.py:130`). One final line: `Wrote {N} decks to {output_path}`.

## Subparser registration

Two new helpers added to `src/sealed/infrastructure/cli.py`:

```python
def _build_train_picker_parser(subparsers) -> None: ...
def _build_pick_decks_parser(subparsers) -> None: ...
```

Both registered in `build_parser()` alongside the existing `_build_*_parser`
calls. Two new run dispatchers:

```python
def run_train_picker(args: argparse.Namespace) -> int: ...
def run_pick_decks(args: argparse.Namespace) -> int: ...
```

Same shape as `run_train_scorer` / `run_build_decks` respectively. Architecture-flag rejection
follows the same `_TRAIN_PICKER_ARCHITECTURE_FLAGS` tuple pattern as
`_TRAIN_SCORER_ARCHITECTURE_FLAGS` (`cli.py:384`).
