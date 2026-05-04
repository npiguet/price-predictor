# Phase 1 Data Model: Card Winnability Pretraining

This document enumerates every entity introduced or modified by spec
016, their fields, validation rules, and lifecycles. It is the
implementation reference for the new code paths; the file/CLI
contracts in `contracts/` reference these names.

## On-disk artifacts

### `output/sealed/cards-played.txt` (new, append-only)

Written by the Java match worker, one line per played game.

| Field                | Type                  | Notes                                                                                       |
|----------------------|-----------------------|---------------------------------------------------------------------------------------------|
| `timestamp`          | ISO 8601 UTC string   | `DateTimeFormatter.ISO_INSTANT`. Matches `match-outcomes.txt` style.                        |
| `run_id`             | UUID string           | Same value as the parent match's `run_id` in `match-outcomes.txt`.                          |
| `set_code`           | uppercase ASCII       | E.g. `BLB`, `RVR`. The set both decks were drawn from.                                      |
| `method_A`           | string token          | Deck A build method (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or a `LABEL`).     |
| `method_B`           | string token          | Same enum/free-form rules as `method_A`.                                                    |
| `cards_played_A`     | `\|`-separated names  | Distinct non-basic card names in deck A whose name entered the battlefield or stack at least once. |
| `cards_played_B`     | `\|`-separated names  | Same for deck B.                                                                            |
| `cards_not_played_A` | `\|`-separated names  | Distinct non-basic card names in deck A that were not played in this game.                  |
| `cards_not_played_B` | `\|`-separated names  | Same for deck B.                                                                            |
| `winner`             | `A` or `B`            | Which side won this game.                                                                   |
| `starter`            | `A` or `B`            | Which side was on the play.                                                                 |

Validation rules:

- 11 fields, semicolon-separated, no trailing semicolon.
- The four card-list columns may be empty (no cards). Empty lists
  are encoded as the empty string between two `;` separators.
- Each card-list column is a *set* of distinct card names (no
  duplicates within a column).
- `cards_played_X ∪ cards_not_played_X` MUST equal the set of
  distinct non-basic card names in deck X (FR-004, FR-004a).
- `cards_played_X ∩ cards_not_played_X = ∅`: a name appears in at
  most one of the two columns. If a deck has multiple copies of a
  card and at least one was played, the name goes only into
  `cards_played_X`; the unplayed copies do *not* re-appear in
  `cards_not_played_X`.
- Lines for a given match appear contiguously and in game order
  (FR-005).
- Line-buffered: opens-writes-closes per line so concurrent workers
  cannot corrupt output (mirrors `MatchResultWriter`).

Lifecycle: append-only across runs. Never truncated by automation.
Manual deletion (or rotation) is the user's responsibility if disk
pressure becomes an issue.

### `output/sealed/cards-win-rates.txt` (new, overwritten per train run)

Written by `train-encoder` after label aggregation, one row per card
included in the training label map.

| Field               | Type           | Notes                                                                          |
|---------------------|----------------|--------------------------------------------------------------------------------|
| `card_name`         | string         | Canonical card name as it appears in `cards-played.txt`.                       |
| `wins_when_played`  | int            | Count of games where the winning side played the card.                         |
| `wins_when_in_deck` | int            | Count of games where the card was in the winning side's deck (whether played). |
| `raw_ratio`         | float in [0,1] | `wins_when_played / wins_when_in_deck`.                                        |
| `shrunk_label`      | float in [0,1] | `(wins_when_played + k/2) / (wins_when_in_deck + k)` for active `k`.           |

Validation rules:

- Sorted by `raw_ratio` descending (FR-013a).
- Only cards with `wins_when_in_deck > 0` appear (FR-012).
- Semicolon-separated, no trailing semicolon, one header row, then data.
- Path is fixed (`output/sealed/cards-win-rates.txt`); not flag-configurable.

Lifecycle: overwritten on every `train-encoder` invocation. Two
runs differing only in `--shrinkage-k` produce diff-able snapshots,
which is how SC-005 is verified.

### `models/sealed/encoder/vocab.txt` (new)

Written by `python -m sealed build-vocab`. Identical line format to
`models/price-predictor/transformer/vocab.txt`: one token per line,
token ID == 0-based line index, written by
`tokenizer_store.save_vocabulary`.

Validation rules:

- First three tokens are `[PAD]`, `[UNK]`, `cardname` (in that
  order), inherited from the upstream builder's seeded specials.
- File is independent from the price-predictor vocab file
  (FR-008): writing one MUST NOT touch the other.

Lifecycle: rewritten in full on every `build-vocab` invocation.
Stale vocabularies are the user's responsibility (FR-022 / spec
clarifications).

### `models/sealed/encoder/{timestamp}.pt` and `models/sealed/encoder/latest.pt` (new)

Written by `train-encoder` after training. Torch `.pt` payload with
two top-level keys:

| Key                | Type                         | Contents                                                                                       |
|--------------------|------------------------------|------------------------------------------------------------------------------------------------|
| `model_state_dict` | `dict[str, Tensor]`          | Filtered to keys under `token_encoder.*` and `card_encoder.*`. Regression head is excluded.    |
| `config`           | `dict` (SealedEncoderConfig) | Serialized via `dataclasses.asdict()`. Holds architecture knobs needed to instantiate the model. |

Validation rules:

- `latest.pt` is a copy of the best-by-validation-loss checkpoint
  written during the run (mirrors price-side
  `transformer_store.save`).
- `model_state_dict` must contain only encoder weights — at load
  time the loader instantiates a fresh `SealedEncoderModel(config)`
  and calls `load_state_dict(strict=True)`. If the regression head
  leaked in, `strict=True` raises immediately.

Lifecycle: timestamped checkpoint kept; `latest.pt` rotated on
each run.

## In-memory entities (Python)

### `SealedEncoderConfig` (new)

Lives in `src/sealed/domain/encoder_model.py` (or a sibling file).

| Field            | Type  | Source                                                              |
|------------------|-------|---------------------------------------------------------------------|
| `vocab_size`     | int   | `len(MtgTokenizer.tokens)` after vocab load.                        |
| `d_model`        | int   | Hardcoded constant 256 (FR-022).                                    |
| `n_layers`       | int   | `--n-layers` (default 6, FR-021).                                   |
| `n_heads`        | int   | `--n-heads` (default 4, FR-021).                                    |
| `ff_dim`         | int   | Hardcoded constant 1024 (FR-022).                                   |
| `max_seq_len`    | int   | Computed from corpus at train start (FR-022). Round up to mult-of-8.|
| `dropout`        | float | `--dropout` (default 0.1, FR-021).                                  |
| `n_pool_queries` | int   | `--n-pool-queries` (default 4, FR-021). MUST divide `d_model`.      |

Validation: `d_model % n_pool_queries == 0`; `n_layers >= 1`;
`n_heads >= 1`; `n_heads` divides `d_model`; `dropout in [0, 1)`.

### `SealedEncoderModel` (new)

Lives in the same module as `SealedEncoderConfig`. PyTorch
`nn.Module` exposing:

- `token_encoder: nn.Module` — `nn.Embedding(vocab_size, d_model)`
  + `nn.Embedding(max_seq_len, d_model)` + dropout. Mirrors the
  price-side embedding stack but isolated as a child module so the
  state-dict prefix is `token_encoder.*` (which simplifies the
  regression-head filter at save time).
- `card_encoder: nn.Module` — `nn.TransformerEncoder` with
  `n_layers` × `nn.TransformerEncoderLayer(d_model, n_heads,
  ff_dim, dropout, batch_first=True)` followed by the dual pool
  layer (multi-query attention pool ‖ max pool).
- `regression_head: nn.Module` — `Linear(2 * d_model, 1)` followed
  by `Sigmoid`. Used only during training; stripped at save.
- `encode(input_ids, attention_mask) -> Tensor[(B, 2*d_model)]` —
  no-grad path used by `CardEncoder` at inference. Exact shape
  match with `CardPriceTransformerModel.encode` so existing
  downstream code is drop-in compatible.
- `forward(input_ids, attention_mask) -> Tensor[(B,)]` — full
  training path: encoder → pool → head → sigmoid.

### Multi-query attention pool

Sub-module of the card encoder.

- Owns `K = n_pool_queries` learned query vectors of length
  `d_token / K` each (where `d_token = d_model`).
- For each query, runs a single attention head against the
  contextualized token sequence (key/value = encoder output).
- Concatenates the K outputs to a `d_token`-dim vector
  (FR-015 clarification).
- Padding-masked (uses the same `attention_mask` as the encoder).

### Per-card winnability map (in-memory only)

Built at the start of each `train-encoder` run.

```python
@dataclass(frozen=True)
class CardLabel:
    card_name: str
    wins_when_played: int
    wins_when_in_deck: int
    shrunk_label: float  # (FR-011)

WinnabilityMap = dict[str, CardLabel]
```

Validation:
- Cards with `wins_when_in_deck == 0` excluded (FR-012).
- Map keys are canonical card names as they appear in
  `cards-played.txt`.
- Not persisted across runs (FR-013); a snapshot is written to
  `cards-win-rates.txt` for inspection (FR-013a) but is not read
  back as a cache.

### `TrainEncoderConfig` (new)

Dataclass in `src/sealed/application/train_encoder.py`. Mirrors
`TrainScorerConfig` shape conventions.

| Field                | Default                                  | Source     |
|----------------------|------------------------------------------|------------|
| `cards_played_path`  | `output/sealed/cards-played.txt`         | FR-021     |
| `cards_folder`       | `output/cardsfolder/`                    | FR-021     |
| `vocab_path`         | `models/sealed/encoder/vocab.txt`        | FR-021     |
| `model_output_dir`   | `models/sealed/encoder/`                 | FR-021     |
| `batch_size`         | 64                                       | FR-021     |
| `epochs`             | 100                                      | FR-021     |
| `lr`                 | 1e-4                                     | FR-021     |
| `patience`           | 20                                       | FR-021     |
| `dropout`            | 0.1                                      | FR-021     |
| `n_layers`           | 6                                        | FR-021     |
| `n_heads`            | 4                                        | FR-021     |
| `n_pool_queries`     | 4                                        | FR-021     |
| `shrinkage_k`        | 20                                       | FR-021     |
| `random_seed`        | 42                                       | FR-022 (constant)  |
| `val_fraction`       | 0.2                                      | FR-022 (constant)  |
| `d_model`            | 256                                      | FR-022 (constant)  |
| `ff_dim`             | 1024                                     | FR-022 (constant)  |

The four FR-022 constants are class attributes / private constants,
not flags; they appear in the dataclass for traceability but are
not exposed to the CLI.

### `BuildVocabConfig` (new)

Dataclass in `src/sealed/application/build_vocab.py`.

| Field          | Default                              | Source |
|----------------|--------------------------------------|--------|
| `cards_folder` | `output/cardsfolder/`                | FR-009 |
| `vocab_path`   | `models/sealed/encoder/vocab.txt`    | FR-009 |
| `target_size`  | 5000                                 | FR-009 |
| `printings_path` | `resources/AllPrintings.json`      | inherited from upstream builder |

Validation: `target_size > seed_token_count` (else `ValueError`,
per Decision D-1).

## In-memory entities (Java)

### `CardsPlayedRow` (new record)

Java `record` in
`forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedRow.java`.

```java
public record CardsPlayedRow(
        Instant timestamp,
        String runId,
        String setCode,
        String methodA,
        String methodB,
        List<String> cardsPlayedA,     // multiset; multiplicities preserved
        List<String> cardsPlayedB,
        List<String> cardsNotPlayedA,
        List<String> cardsNotPlayedB,
        char winner,                    // 'A' or 'B'
        char starter                    // 'A' or 'B'
) {}
```

### `CardsPlayedWriter` (new)

Sibling of `MatchResultWriter`. Same opens-writes-closes pattern,
distinct path (`output/sealed/cards-played.txt`) and distinct line
formatter that emits the eleven fields above. Method:
`void write(CardsPlayedRow row)`.

### `GameOutcome` (extended)

`forge-connector/.../GamePlayer.java`. Existing fields
(`winner`, `playFirst`) gain two siblings:

```java
public record GameOutcome(
        String winner,
        String playFirst,
        Set<String> cardsPlayedA,   // non-basic, non-token, set membership, paperCard names
        Set<String> cardsPlayedB
) { ... }
```

The two new fields are populated by a per-game eventbus visitor
registered on the Forge `Game` instance inside `playMatch()` via
`game.subscribeToEvents(visitor)`. The visitor is an
`IGameEventVisitor.Base<Void>` subclass mirroring
`../jumpstart-tierlist/.../JumpstartMatch.java#CardCollector`
(see research.md decision D-3 for full rationale).

It listens to **two** event types:

1. `GameEventCardChangeZone` — only when `event.to().getZoneType()
   == ZoneType.Battlefield`. Catches permanents that resolved.
2. `GameEventSpellAbilityCast` — every spell cast. Catches
   instants and sorceries that never reach the battlefield.

For each event, before recording, the visitor applies these
filters:

- `card.getController() == card.getOwner()` (drops stolen cards).
- `!card.isToken()`.
- `!card.getType().isBasicLand()` (FR-004a).
- The recorded name is `card.getPaperCard().getName()` so
  copy/clone effects credit the cast card, not the copied permanent.

The card is bucketed by `card.getOwner().getName()` (Forge lobby
player name), then mapped to `A` / `B` via the
`LOBBY_NAME_A` / `LOBBY_NAME_B` constants in `GamePlayer`.

### `MatchGenerator.generateMatch()` (extended return)

Returns `(MatchResult, List<CardsPlayedRow>)` where the list has
one entry per played game. The worker loop in `MatchWorkerMain`
writes the match line via `MatchResultWriter` and the game lines
via `CardsPlayedWriter`.

## State transitions

The feature has no long-lived stateful entities. The only
"transitions" worth noting are:

- `cards-played.txt`: monotonically grows during `match-outcomes`;
  never read by the writer. Read once per `train-encoder` run.
- `models/sealed/encoder/latest.pt`: rewritten on every
  `train-encoder` run with the best-by-val-loss checkpoint.
- `output/sealed/cards-win-rates.txt`: rewritten on every
  `train-encoder` run.

Concurrency:
- Multiple match-outcomes workers can write concurrently to
  `cards-played.txt` because `CardsPlayedWriter` opens-writes-
  closes per line (same guarantee as `MatchResultWriter`).
- `train-encoder` is single-process and reads `cards-played.txt`
  while it might still be growing — that is fine, because the read
  is one pass to EOF; lines added after the read starts are
  ignored, but a partially-written final line (truncated mid-line
  by a JVM crash) is silently skipped by the reader (Edge Cases:
  trailing partial line tolerated).
