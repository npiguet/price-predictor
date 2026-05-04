# File-Format Contracts

Two new on-disk file formats are introduced by this feature, and
one existing artifact (`models/sealed/encoder/*.pt`) follows a
specific torch-serialization contract. Each schema is fixed by
spec; downstream readers may rely on these guarantees.

## `output/sealed/cards-played.txt`

Append-only, line-buffered. Written by the Java match worker.

### Line schema (eleven semicolon-separated fields, no trailing `;`)

```
timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter
```

Field-by-field:

1. **`timestamp`** — ISO 8601 UTC, e.g. `2026-05-03T14:22:01Z`.
   Same formatter as `match-outcomes.txt` (`ISO_INSTANT`).
2. **`run_id`** — UUID string (lowercase 8-4-4-4-12 hex). MUST be
   identical to the parent match's `run_id` in `match-outcomes.txt`.
3. **`set_code`** — uppercase ASCII (e.g. `BLB`, `RVR`). Both
   decks come from this set.
4. **`method_A`** — deck-A build-method tag. One of `forge-best`,
   `forge-3sub`, `forge-8sub`, `random`, or a `LABEL` string from a
   generated-decks file. Must match the value in the parent match.
5. **`method_B`** — deck-B build-method tag (same enum/free-form
   rules as `method_A`).
6. **`cards_played_A`** — `|`-separated **distinct** card names. Each
   name appears at most once. Empty string if no non-basic cards were
   played by side A in this game.
7. **`cards_played_B`** — same for side B.
8. **`cards_not_played_A`** — `|`-separated **distinct** card names of
   deck A's non-basic cards that did not enter the battlefield or stack.
   Disjoint from `cards_played_A` (FR-004).
9. **`cards_not_played_B`** — same for side B.
10. **`winner`** — `A` or `B`, matching the corresponding character
    in the parent match's `games` field.
11. **`starter`** — `A` or `B`, matching the corresponding character
    in the parent match's `play` field.

### Invariants

- `cards_played_X ∪ cards_not_played_X == set(distinct non-basic card
  names in deck X)` (FR-004, FR-004a). Both columns are sets — no
  duplicates within a column, and no name in both columns at the same
  time.
- A name appears in `cards_played_X` iff at least one copy entered
  the battlefield or the stack while controlled by side X (FR-003).
  A name with multiple copies in the deck appears at most once in
  `cards_played_X` and never also in `cards_not_played_X`.
- Game lines for one match appear contiguously and in game order
  (FR-005).
- Concurrent worker writes do not interleave: each write is
  open-write-close.
- Trailing partial line is tolerated by readers (Edge Cases). A
  reader treats the file as a stream of complete lines, discarding
  a non-newline-terminated suffix.

### Joinability

`cards-played.txt` joins to `match-outcomes.txt` by
`(run_id, set_code, method_A, method_B)` plus positional offset
within that group: the i-th game block of a match corresponds to
the i-th `(run_id, set_code, method_A, method_B)` row of
`match-outcomes.txt`.

## `output/sealed/cards-win-rates.txt`

Overwritten on each `train-encoder` run. Path is fixed (not
flag-configurable, FR-013a).

### Schema (one header row + N data rows, semicolon-separated)

```
card_name;wins_when_played;wins_when_in_deck;raw_ratio;shrunk_label
Black Lotus;142;143;0.99301;0.95423
Lightning Bolt;81;103;0.78641;0.78103
...
```

Rules:

- One row per card included in the training label map (cards with
  `wins_when_in_deck == 0` excluded, FR-012).
- Sorted by `raw_ratio` descending (FR-013a).
- `raw_ratio` and `shrunk_label` formatted with five decimal places.
- Header row is present so the file can be inspected with `csv` /
  `pandas.read_csv(sep=";")` without manual column setup.

## `models/sealed/encoder/{timestamp}.pt` and `latest.pt`

Torch-pickled `dict` with two top-level keys:

| Key                | Value                                                                     |
|--------------------|---------------------------------------------------------------------------|
| `model_state_dict` | `dict[str, Tensor]` containing only `token_encoder.*` and `card_encoder.*` keys. |
| `config`           | `dict` from `dataclasses.asdict(SealedEncoderConfig)`.                    |

Validation at load time:

- `SealedEncoderModel(SealedEncoderConfig(**payload["config"])).load_state_dict(payload["model_state_dict"], strict=True)`
  MUST succeed. If it does not, the saved file is invalid and the
  loader raises immediately.
- `latest.pt` is a byte-for-byte copy of the chosen
  `{timestamp}.pt`. Copying (rather than symlinking) matches the
  price-side convention and is portable across Windows/Linux.

### Compatibility with existing `CardEncoder` (sealed)

The new model exposes `encode(input_ids, attention_mask) ->
Tensor[(B, 2*d_model)]` so
`src/sealed/domain/card_encoder.py:CardEncoder` works without
modification: only the loader (`encoder_store.load_encoder`) and
the model class change. The `2*d_model + FEATURE_COUNT` shape that
`encode-cards` writes to `.npz` is preserved.
