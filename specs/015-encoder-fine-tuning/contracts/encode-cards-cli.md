# CLI Contract: `python -m sealed encode-cards`

**Feature**: 015-encoder-fine-tuning

This file documents the `encode-cards` subcommand surface after this
feature ships.

## Synopsis

```text
python -m sealed encode-cards [options]
```

Two valid invocation shapes:

1. **From a price-predictor (or Phase A) source** — encoder weights come from a price-predictor `.pt` file via `--encoder-checkpoint`.
2. **From a Phase B scorer source** — encoder weights are extracted from a Phase B scorer checkpoint's `encoder_state_dict` via `--scorer-checkpoint`.

In both cases, the use case re-encodes every `.txt` card under
`--cards-path` (subject to the existing skip-if-`.npz`-exists
idempotence, overridable with `--clean`).

## Flag Reference

| Flag | Default | Behavior |
|---|---|---|
| `--encoder-checkpoint <path>` | `models/price-predictor/transformer/latest.pt` | Encoder weights source. **Renamed** from the previous `--encoder-path`. (Decision §7) |
| `--scorer-checkpoint <path>` | (none) | Extract encoder weights from a Phase B scorer checkpoint. (FR-013, FR-014) |
| `--vocab-path <path>` | `models/price-predictor/transformer/vocab.txt` | Tokenizer vocabulary. Unchanged. |
| `--cards-path <path>` | `output/cardsfolder/` | Directory of `.txt` files to encode (searched recursively). Unchanged. |
| `--clean` | `False` | Delete all existing `.npz` files before encoding. Unchanged. |

## Removed flags

| Flag | Replacement |
|---|---|
| `--encoder-path` | Renamed to `--encoder-checkpoint`. (Decision §7) |

## Mutual Exclusivity

| Rule | Result on violation | Source |
|---|---|---|
| `--encoder-checkpoint` and `--scorer-checkpoint` cannot both be set **when both were explicitly passed**. The default value of `--encoder-checkpoint` does not trigger the rule. | Reject with: `"--encoder-checkpoint and --scorer-checkpoint are mutually exclusive: choose one source for the encoder weights."` | FR-013 |
| `--scorer-checkpoint` pointed at a Phase A checkpoint (no `encoder_state_dict`). | Reject with: `"<path> is a Phase A scorer checkpoint and contains no encoder weights. Use --encoder-checkpoint for non-Phase-B sources."` | FR-014 |

Detection note: same as `train-scorer`. Register
`--encoder-checkpoint default=None` and resolve to the literal default
*after* the conflict check.

## Help Text Requirements

Every flag listed above MUST appear in `encode-cards --help` with a
one-line purpose, default value, and any mutual-exclusivity rule
expressed in plain English (FR-016).

## Output

For every `<letter>/<sanitized_name>.txt` under `--cards-path`, a sibling
`.npz` file is written with the same name. The file format is unchanged:
a single `float32` array of shape `(2 * d_model + FEATURE_COUNT,)` under
key `"embedding"`.

If a `.npz` already exists, it is skipped (default) or replaced (with
`--clean`, which deletes all `.npz` files up front and then re-encodes
every `.txt`).

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All `.txt` files processed without errors. |
| `1` | Run completed but at least one card raised an error (errors collected and printed). |
| `2` | Setup failure (missing files, conflicting flags, Phase A checkpoint passed to `--scorer-checkpoint`, etc.). |

## Backwards-Compatibility

- `--encoder-path` is removed: invocations that pass it MUST fail with an `unrecognized arguments` error.
- `.npz` files produced before this feature remain readable; their format is unchanged.

## Examples

```bash
# Existing pre-Phase-B usage — explicit price-predictor checkpoint
python -m sealed encode-cards \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --clean

# Same, relying on the default
python -m sealed encode-cards --clean

# Refresh the cache from a Phase B scorer checkpoint
python -m sealed encode-cards \
    --scorer-checkpoint models/sealed/scorer/best_l6_h4_s4_ff1088_mlp256_lr1e-5.pt \
    --clean
```
