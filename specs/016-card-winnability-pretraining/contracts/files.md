# File-Format Contracts

Three on-disk artifacts whose schema is fixed by spec 016. Downstream
readers may rely on these guarantees.

## `output/sealed/cards-played.txt` (already in place)

Append-only, line-buffered. Written by the Java match worker
(`CardsPlayedWriter`).

### Line schema (eleven semicolon-separated fields, no trailing `;`)

```
timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter
```

Field-by-field:

1. **`timestamp`** — ISO 8601 UTC, e.g. `2026-05-03T14:22:01Z`.
2. **`run_id`** — UUID string. Identical to the parent match's `run_id`
   in `match-outcomes.txt`.
3. **`set_code`** — uppercase ASCII (e.g. `BLB`, `RVR`). Both decks come
   from this set.
4. **`method_A`** — deck-A build-method tag (`forge-best`, `forge-3sub`,
   `forge-8sub`, `random`, or a `LABEL` from a generated-decks file).
5. **`method_B`** — deck-B build-method tag, same shape.
6. **`cards_played_A`** — `|`-separated **distinct** card names. Empty
   string if no non-basic cards were played by side A in this game.
7. **`cards_played_B`** — same for side B.
8. **`cards_not_played_A`** — `|`-separated **distinct** card names of
   deck A's non-basic cards that did not enter the battlefield or stack.
   Disjoint from `cards_played_A`.
9. **`cards_not_played_B`** — same for side B.
10. **`winner`** — `A` or `B`, matching the corresponding char in the
    parent match's `games` field.
11. **`starter`** — `A` or `B`, matching the corresponding char in the
    parent match's `play` field.

### Invariants

- `cards_played_X ∪ cards_not_played_X == set(distinct non-basic card
  names in deck X)` (FR-004, FR-004a).
- `cards_played_X ∩ cards_not_played_X == ∅`.
- A name appears in `cards_played_X` iff at least one copy entered the
  battlefield or the stack while controlled by side X (FR-003).
- Game lines for one match appear contiguously and in game order (FR-005).
- Concurrent worker writes do not interleave: each write is
  open-write-close.
- Trailing partial line is tolerated by readers (Edge Cases): the final
  non-newline-terminated suffix is silently discarded.

### Joinability

`cards-played.txt` joins to `match-outcomes.txt` by
`(run_id, set_code, method_A, method_B)` plus positional offset within
that group: the i-th game block of a match corresponds to the i-th
`(run_id, set_code, method_A, method_B)` row of `match-outcomes.txt`.

## `output/sealed/cards-win-rates.txt` (schema replaced by this iteration)

Overwritten on each `train-encoder` run. Path is fixed (not
flag-configurable, FR-013a).

### Schema (one header row + N data rows, semicolon-separated, no trailing `;`)

**Header**:

```
card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G
```

**Example data rows**:

```
Lightning Bolt;1840;2310;120;195;0.71429;0.69412;0.59231;0.57814;0.93878;0.93215;0.13520;0.12831;;;;;;;0.04211;0.04102;;
Plains;;;;;;;;;;;;;;;;;;;;;;
```

(`Plains` would not actually appear — basic lands are excluded by
FR-004a — but the empty-cell rendering is illustrative: every cell whose
slice denominator is zero is the empty string in both raw and shrunk
columns for that head.)

### Field types

| Position | Field | Type | Notes |
|---|---|---|---|
| 1 | `card_name` | string | Canonical Forge spelling. |
| 2 | `wins_when_played` | int | Pass 1 primary counter. |
| 3 | `wins_when_in_deck` | int | Pass 1 primary counter. |
| 4 | `losses_when_played` | int | Pass 1 primary counter. |
| 5 | `losses_when_in_deck` | int | Pass 1 primary counter. |
| 6 | `raw_score_play` | float in [-1, +1] or empty | FR-011 raw form. |
| 7 | `shrunk_score_play` | float in [-1, +1] or empty | FR-011 shrunk form with `k = --shrinkage-k`. |
| 8 | `raw_score_draw` | float in [-1, +1] or empty | FR-011. |
| 9 | `shrunk_score_draw` | float in [-1, +1] or empty | FR-011. |
| 10 | `raw_played_rate` | float in [0, 1] or empty | FR-011. |
| 11 | `shrunk_played_rate` | float in [0, 1] or empty | FR-011. |
| 12 | `raw_cast_lift` | float in [-1, +1] or empty | FR-011. |
| 13 | `shrunk_cast_lift` | float in [-1, +1] or empty | FR-011. |
| 14–23 (odd) | `raw_color_lift_X` | float in [-1, +1] or empty | One per X ∈ {W, U, B, R, G}, FR-011. |
| 15–23 (even) | `shrunk_color_lift_X` | float in [-1, +1] or empty | One per X. |

Total: **24 columns**.

### Rules

- One header row + one data row per card included in the training label
  map (cards excluded by FR-012 — those with zero total
  `wins_when_in_deck + losses_when_in_deck` — are absent).
- Sorted by `shrunk_score_play` descending (Decision D-15). Cards with
  empty `shrunk_score_play` sort to the end.
- Floats formatted with five decimal places.
- Counters formatted as plain integers (no thousands separators).
- Cells whose slice denominator is zero are written as the empty string
  (no `0.0`, no `-`, no whitespace) in both the raw and shrunk column
  for that head — distinguishes "no signal" from "neutral signal".

### Lifecycle

Overwritten on every `train-encoder` invocation. Two runs differing only
in `--shrinkage-k` produce diff-able snapshots; this is how SC-005 is
verified.

## `models/sealed/encoder/{timestamp}.pt` and `models/sealed/encoder/latest.pt`

Torch-pickled `dict` with two top-level keys:

| Key | Value |
|---|---|
| `model_state_dict` | `dict[str, Tensor]` containing only `token_encoder.*` and `card_encoder.*` keys. `regression_heads.*` and `mlm_head.*` are filtered out at save time per FR-020. |
| `config` | `dict` from `dataclasses.asdict(SealedEncoderConfig)`. |

### Validation at load time

- `SealedEncoderModel(SealedEncoderConfig(**payload["config"])).load_state_dict(payload["model_state_dict"], strict=True)` MUST succeed
  for the encoder children. `SealedEncoderStore.load_encoder` enforces the
  prefix invariant and raises `RuntimeError` if a non-encoder key is found
  in the saved file (would indicate a save-time leak of regression heads
  or MLM head — FR-020 violation).
- `latest.pt` is a byte-for-byte copy of the chosen `{timestamp}.pt`.
  Copying (rather than symlinking) matches the price-side convention and
  is portable across Windows/Linux.

### Compatibility with existing `CardEncoder` (sealed)

The model's inference surface (`encode(input_ids, attention_mask) ->
Tensor[(B, 2*d_model)]`) is unchanged from v1, so
`src/sealed/domain/card_encoder.py:CardEncoder` works without
modification. The downstream `2 * d_model + FEATURE_COUNT` shape that
`encode-cards` writes to `.npz` is preserved (the new heads and MLM head
are training-only and bypassed by `encode()`).

## `models/sealed/encoder/vocab.txt` (gains a `[MASK]` token)

One token per line, token ID = 0-based line index. Already documented in
`data-model.md § models/sealed/encoder/vocab.txt`. The schema invariant
is unchanged except that **a `[MASK]` token MUST be present**; downstream
`train-encoder` rejects vocabularies without it (FR-023a).
