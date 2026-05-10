# Phase 1 Data Model: Card Winnability Pretraining

This document enumerates every entity introduced or modified by spec 016,
with fields, validation rules, and lifecycles. The file/CLI contracts in
`contracts/` reference these names. Spec 016's first iteration shipped a
single-`shrunk_label` schema; this document reflects the current spec
(9 regression heads + MLM auxiliary, with per-color counters from a
two-pass aggregator).

## On-disk artifacts

### `output/sealed/cards-played.txt` (already in place)

Written by the Java match worker, one line per played game.

| Field | Type | Notes |
|---|---|---|
| `timestamp` | ISO 8601 UTC string | `DateTimeFormatter.ISO_INSTANT`. Same formatter as `match-outcomes.txt`. |
| `run_id` | UUID string | Same value as the parent match's `run_id` in `match-outcomes.txt`. |
| `set_code` | uppercase ASCII | E.g. `BLB`, `RVR`. |
| `method_A` | string token | Deck A build method (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or a `LABEL`). |
| `method_B` | string token | Same shape as `method_A`. |
| `cards_played_A` | `\|`-separated names | Distinct non-basic card names in deck A whose name entered the battlefield or stack at least once. |
| `cards_played_B` | `\|`-separated names | Same for deck B. |
| `cards_not_played_A` | `\|`-separated names | Distinct non-basic card names in deck A that were not played in this game. |
| `cards_not_played_B` | `\|`-separated names | Same for deck B. |
| `winner` | `A` or `B` | Which side won this game. |
| `starter` | `A` or `B` | Which side was on the play. |

Validation rules (already enforced by writers and reader):

- 11 fields, semicolon-separated, no trailing semicolon.
- The four card-list columns may be empty (encoded as the empty string).
- Each card-list column is a *set* of distinct card names (no duplicates).
- `cards_played_X ∪ cards_not_played_X` MUST equal the set of distinct
  non-basic card names in deck X (FR-004, FR-004a).
- `cards_played_X ∩ cards_not_played_X = ∅`.
- Lines for a given match appear contiguously and in game order (FR-005).
- Line-buffered: opens-writes-closes per line so concurrent workers cannot
  corrupt output.

Lifecycle: append-only across runs. Never truncated by automation.

### `output/sealed/cards-win-rates.txt` (overwritten per train run, **schema replaced** by this iteration)

Written by `train-encoder` after label aggregation. One header row + one
row per card included in the training label map.

**Header row** (23 columns, semicolon-separated, no trailing `;`):

```
card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G
```

**Data rows**: same 23 columns. Counters are integers; raw/shrunk values
are floats formatted to five decimal places. Cells whose slice
denominator is zero are written as the empty string in both the raw and
shrunk column for that head (cell present-but-empty per FR-012).

Validation rules:

- Header is present and matches FR-013a verbatim.
- One data row per card included in the label map (cards excluded by
  FR-012 — those with zero total `wins_when_in_deck + losses_when_in_deck`
  — are absent).
- Sorted by `shrunk_score_play` descending (FR-013a / Decision D-15).
  Cards with empty `shrunk_score_play` sort to the end.
- Path is fixed (`output/sealed/cards-win-rates.txt`); not flag-configurable.

Lifecycle: overwritten on every `train-encoder` invocation. Two runs
differing only in `--shrinkage-k` produce diff-able snapshots, which is
how SC-005 is verified.

### `models/sealed/encoder/vocab.txt` (already in place; schema gains `[MASK]`)

Written by `python -m sealed build-vocab`. One token per line, token ID =
0-based line index.

Validation rules:

- First four seeded specials are `[PAD]`, `[UNK]`, `cardname`, `[MASK]`,
  in that order. (Pre-existing `[PAD]`/`[UNK]`/`cardname` keep their IDs;
  `[MASK]` slots in at the next available position.)
- `[MASK]` does not collide with any corpus-derived token (square
  brackets are not part of converted card text — FR-009a).
- File is independent from the price-predictor vocab file (FR-008):
  writing one MUST NOT touch the other.

Lifecycle: rewritten in full on every `build-vocab` invocation. Stale
vocabularies are the user's responsibility (FR-022 / spec § Assumptions).

### `models/sealed/encoder/{timestamp}.pt` and `latest.pt` (already in place)

Written by `train-encoder` after training. Torch `.pt` payload with two
top-level keys:

| Key | Type | Contents |
|---|---|---|
| `model_state_dict` | `dict[str, Tensor]` | Filtered to keys under `token_encoder.*` and `card_encoder.*`. Regression heads (`regression_heads.*`) and MLM head (`mlm_head.*`) are excluded at save time (FR-020). |
| `config` | `dict` | `dataclasses.asdict(SealedEncoderConfig)`. Holds architecture knobs needed to instantiate the model. |

Validation rules:

- `latest.pt` is a byte-for-byte copy of the best-by-full-validation-loss
  checkpoint (FR-019, Clarification 2026-05-10).
- At load time, `SealedEncoderStore.load_encoder` instantiates a fresh
  `SealedEncoderModel(config)` and calls `load_state_dict(strict=True)` on
  each encoder child (`token_encoder`, `card_encoder`). The live model
  also owns freshly-initialized regression heads and an MLM head, but
  those are training-only and bypassed at inference time.
- The save path enforces FR-020 by filtering the state-dict through
  `_ENCODER_PREFIXES = ("token_encoder.", "card_encoder.")`. Any leak
  (regression head or MLM head appearing in the saved file) raises
  immediately at the next load.

Lifecycle: timestamped checkpoint kept; `latest.pt` rotated on each run.

## In-memory entities (Python)

### `SealedEncoderConfig` (already in place; no schema change)

Lives in `src/sealed/domain/encoder_model.py:17`.

| Field | Type | Source |
|---|---|---|
| `vocab_size` | int | `len(MtgTokenizer.tokens)` after vocab load (now includes `[MASK]`). |
| `d_model` | int | Hardcoded constant 256 (FR-022). |
| `n_layers` | int | `--n-layers` (default 6, FR-021). |
| `n_heads` | int | `--n-heads` (default 4, FR-021). |
| `ff_dim` | int | Hardcoded constant 1024 (FR-022). |
| `max_seq_len` | int | Computed from corpus at train start (FR-022). Round up to multiple of 8. |
| `dropout` | float | `--dropout` (default 0.1, FR-021). |
| `n_pool_queries` | int | `--n-pool-queries` (default 4, FR-021). MUST divide `d_model`. |

Validation (already enforced in `__post_init__`): `d_model %
n_pool_queries == 0`, `n_layers >= 1`, `n_heads >= 1`, `n_heads` divides
`d_model`, `dropout in [0, 1)`.

### `SealedEncoderModel` (modified — heads structure replaced + MLM head added)

Lives in `src/sealed/domain/encoder_model.py:157`. PyTorch `nn.Module`
exposing the following children (state-dict prefixes shown in
parentheses):

- `token_encoder` (`token_encoder.*`) — token embedding + position
  embedding + dropout. Saved.
- `card_encoder` (`card_encoder.*`) — `nn.TransformerEncoder` stack +
  multi-query attention pool + max pool, output dim `2 * d_model`. Saved.
- `regression_heads` (`regression_heads.*`) — `nn.ModuleDict` with five
  members, **not saved** (filtered at save time):
  - `score_play`: `Linear(2*d_model, 1)` followed by `Tanh`.
  - `score_draw`: `Linear(2*d_model, 1)` followed by `Tanh`.
  - `played_rate`: `Linear(2*d_model, 1)` followed by `Sigmoid`.
  - `cast_lift`: `Linear(2*d_model, 1)` followed by `Tanh`.
  - `color_lift`: `Linear(2*d_model, 5)` followed by `Tanh` (one column
    per WUBRG letter).
- `mlm_head` (`mlm_head.*`) — `Linear(d_model, vocab_size)` reading the
  contextualized token sequence (output of the transformer-layer stack,
  before the pool). **Not saved** (filtered at save time).

Forward methods:

- `encode(input_ids, attention_mask) -> Tensor[(B, 2*d_model)]` — no-grad
  inference path used by `CardEncoder` at inference; bypasses every head.
  Unchanged from v1.
- `forward(input_ids, attention_mask) -> dict` — training path. Returns:
  - `score_play: (B,)`, `score_draw: (B,)`, `played_rate: (B,)`,
    `cast_lift: (B,)` — each from its own head.
  - `color_lift: (B, 5)` — from the shared 5-output head.
  - `mlm_logits: (B, T, vocab_size)` — from `mlm_head` reading the
    contextualized token sequence.

The state-dict prefix layout (`token_encoder.*` / `card_encoder.*` /
`regression_heads.*` / `mlm_head.*`) lets the save path filter heads + MLM
out by key prefix at save time without monkey-patching.

### Multi-query attention pool (already in place)

Sub-module of the card encoder.

- Owns `K = n_pool_queries` learned query vectors of length `d_model / K`
  each.
- For each query, runs a single attention head against the contextualized
  token sequence.
- Concatenates the K outputs to a `d_model`-dim vector (FR-015 clarification
  from session 2026-05-03).
- Padding-masked via the same `attention_mask` as the encoder.

### Per-card counters and labels (in-memory only — replaces v1 `CardLabel`)

Built at the start of each `train-encoder` run. Two dataclasses, both
private to `train_encoder.py`.

```python
@dataclass(frozen=True)
class CardCounters:
    # Pass 1: primary (over both sides of every game)
    wins_when_played: int
    losses_when_played: int
    wins_when_in_deck: int
    losses_when_in_deck: int
    # Pass 1: @play subset (where the card's owner was the starter)
    wins_when_played_at_play: int
    losses_when_played_at_play: int
    wins_when_in_deck_at_play: int
    losses_when_in_deck_at_play: int
    # Pass 2: per-color (one entry per X ∈ {W, U, B, R, G})
    wins_when_played_with: dict[str, int]      # {"W": int, "U": int, ...}
    losses_when_played_with: dict[str, int]
    wins_when_in_deck_with: dict[str, int]
    losses_when_in_deck_with: dict[str, int]


@dataclass(frozen=True)
class CardLabels:
    card_name: str
    counters: CardCounters
    # Raw + shrunk per FR-011, with None for empty-denominator cells
    raw_score_play: float | None
    shrunk_score_play: float | None
    raw_score_draw: float | None
    shrunk_score_draw: float | None
    raw_played_rate: float | None
    shrunk_played_rate: float | None
    raw_cast_lift: float | None
    shrunk_cast_lift: float | None
    raw_color_lift: dict[str, float | None]      # {"W": v|None, "U": v|None, ...}
    shrunk_color_lift: dict[str, float | None]


CardLabelMap = dict[str, CardLabels]
```

Validation:

- Cards with `wins_when_in_deck + losses_when_in_deck == 0` are excluded
  from the map entirely (FR-012).
- Cells whose slice denominator is zero are stored as `None` in both the
  raw and shrunk column for that head (FR-012). At training time these
  cells produce a head_mask of 0 and contribute zero loss; at write time
  they produce empty strings in `cards-win-rates.txt`.
- Map keys are canonical card names as they appear in `cards-played.txt`.
- Not persisted across runs (FR-013); the `cards-win-rates.txt` snapshot
  is for inspection only, not a cache.

### `TrainEncoderConfig` (modified — two new fields)

Dataclass in `src/sealed/application/train_encoder.py:281`.

| Field | Default | Source |
|---|---|---|
| `cards_played_path` | `output/sealed/cards-played.txt` | FR-021 |
| `cards_folder` | `output/cardsfolder/` | FR-021 |
| `vocab_path` | `models/sealed/encoder/vocab.txt` | FR-021 |
| `model_output_dir` | `models/sealed/encoder/` | FR-021 |
| `batch_size` | 64 | FR-021 |
| `epochs` | 100 | FR-021 |
| `lr` | 1e-4 | FR-021 |
| `patience` | 20 | FR-021 |
| `dropout` | 0.1 | FR-021 |
| `n_layers` | 6 | FR-021 |
| `n_heads` | 4 | FR-021 |
| `n_pool_queries` | 4 | FR-021 |
| `shrinkage_k` | 20 | FR-021 |
| **`mlm_weight`** | **0.1** | **FR-021 (new)** |
| **`mlm_mask_prob`** | **0.15** | **FR-021 (new)** |
| `random_seed` | 42 | FR-022 (constant) |
| `val_fraction` | 0.2 | FR-022 (constant) |
| `d_model` | 256 | FR-022 (constant) |
| `ff_dim` | 1024 | FR-022 (constant) |

The four FR-022 constants are class attributes / private constants, not
flags; they appear in the dataclass for traceability but are not exposed
to the CLI. Optimizer (AdamW), gradient clip (max-norm 1.0), and LR
schedule (5%-step linear warmup → constant) are likewise hardcoded
constants per FR-022 / Clarification 2026-05-10 — they live in
`_make_optimizer` and `_train_epoch` rather than on the dataclass.

### `BuildVocabConfig` (already in place; no schema change)

Dataclass in `src/sealed/application/build_vocab.py`.

| Field | Default | Source |
|---|---|---|
| `cards_folder` | `output/cardsfolder/` | FR-009 |
| `vocab_path` | `models/sealed/encoder/vocab.txt` | FR-009 |
| `target_size` | 5000 | FR-009 |
| `printings_path` | `resources/AllPrintings.json` | inherited from upstream builder |

Validation: `target_size > seed_token_count` (else `ValueError`, per
Decision D-1). The seed_token count grows by one in this iteration to
include `[MASK]`.

## In-memory entities (Java — already in place)

### `CardsPlayedRow`

Java `record` in
`forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedRow.java:30`.

```java
public record CardsPlayedRow(
        Instant timestamp,
        String runId,
        String setCode,
        String methodA,
        String methodB,
        List<String> cardsPlayedA,
        List<String> cardsPlayedB,
        List<String> cardsNotPlayedA,
        List<String> cardsNotPlayedB,
        char winner,
        char starter
) { ... }
```

### `CardsPlayedWriter`

Sibling of `MatchResultWriter`. Same opens-writes-closes pattern, distinct
path (`output/sealed/cards-played.txt`) and distinct line formatter that
emits the eleven fields above.

### `PlayedCardCollector` and `MatchGenerator.BASIC_LAND_NAMES`

The Forge-side observer (`PlayedCardCollector.java:108`) and the basic-land
exclusion set (`MatchGenerator.java:259`) jointly satisfy FR-003, FR-004,
FR-004a. The observer captures `GameEventCardChangeZone` (filtered to
`ZoneType.Battlefield`) and `GameEventSpellAbilityCast`, applies four
filters (controller==owner, !isToken, !isBasicLand, gamePieceType==CARD),
and records `card.getPaperCard().getName()` so copy/clone effects credit
the cast card. None of this changes in this iteration.

## State transitions

The feature has no long-lived stateful entities. The only "transitions"
worth noting:

- `cards-played.txt`: monotonically grows during `match-outcomes`; never
  read by the writer. Read twice per `train-encoder` run (pass 1 + pass 2
  per Decision D-13).
- `models/sealed/encoder/latest.pt`: rewritten on every `train-encoder`
  run with the best-by-full-validation-loss checkpoint.
- `output/sealed/cards-win-rates.txt`: rewritten on every `train-encoder`
  run with the new 24-column schema.

Concurrency:

- Multiple `match-outcomes` workers can write concurrently to
  `cards-played.txt` because `CardsPlayedWriter` opens-writes-closes per
  line.
- `train-encoder` is single-process and reads `cards-played.txt` while it
  might still be growing — that is fine, because each pass is one read to
  EOF; lines added after a pass starts are seen by the next pass (or
  ignored if the run has moved past pass 2). A partially-written final
  line is silently skipped by `iter_rows`.
