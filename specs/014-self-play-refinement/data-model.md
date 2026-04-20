# Data Model: Self-Play Refinement

**Feature**: 014-self-play-refinement
**Date**: 2026-04-20

## File Formats

### Pools File (`pools.txt`)

One pool per line. Each pool is the non-basic-land cards from 6 boosters.

**Format**: `SET_CODE;Card1|Card2|...|CardN`

| Field | Type | Description |
|---|---|---|
| `SET_CODE` | string | MTG set code (e.g. `MH3`, `BLB`). First semicolon-delimited field. |
| Card names | pipe-separated strings | Non-basic-land card names from the pool. Variable count (~80–100 per pool). |

**Producer**: `python -m sealed generate-pools`
**Consumers**: `python -m sealed build-decks`, `evaluate_scorer._parse_pools()`

**Example**:
```
MH3;Flare of Denial|Wight of the Reliquary|Ral and Niv-Mizzet|...
BLB;Moonrise Cleric|Bark-Knuckle Boxer|Valley Questcaller|...
```

**No backward compatibility**: Old format (`Card1|Card2|...` without set code) is not supported. Pool files are cheap to regenerate and mostly unused.

### Generated-Decks File (`generated-decks.txt`)

One complete 40-card deck per line, built by the scorer-guided greedy builder.

**Format**: `SET_CODE;Card1|Card2|...|Card40`

| Field | Type | Description |
|---|---|---|
| `SET_CODE` | string | Set code of the pool this deck was built from. |
| Card names | 40 pipe-separated strings | Complete deck: 23 nonland spells/nonbasic lands + 17 basic lands. Duplicates repeat. |

**Producer**: `python -m sealed build-decks`
**Consumers**: Java `MatchWorkerMain` (self-play mode), Java `SelfPlayMatchGenerator`

**Example**:
```
MH3;Flare of Denial|Wight of the Reliquary|...|Mountain|Mountain|Island|Island|...
```

### Match-Outcomes File (`match-outcomes.txt`)

Append-only. One match per line. Unchanged from Phase 0.

**Format**: `deck_A_cards;deck_B_cards;wins_A;wins_B`

| Field | Type | Description |
|---|---|---|
| `deck_A_cards` | pipe-separated strings | 40 card names for deck A |
| `deck_B_cards` | pipe-separated strings | 40 card names for deck B |
| `wins_A` | integer | Games won by deck A |
| `wins_B` | integer | Games won by deck B |

**Invariant**: `wins_A + wins_B` is 2 or 3 (best-of-3).

**Producers**: Java `MatchWorkerMain` (Phase 0 and self-play modes)
**Consumer**: `python -m sealed train-scorer`

## Entities (New)

### `BuildDecksConfig` (Python dataclass)

Configuration for the `build-decks` subcommand.

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | `Path` | `models/sealed/scorer/latest.pt` | Scorer model checkpoint path |
| `pools_path` | `Path` | (required) | Input pools file |
| `cards_path` | `Path` | `output/cardsfolder/` | Directory with `.npz` card embeddings |
| `output` | `Path` | `output/sealed/generated-decks.txt` | Output generated-decks file path |

### `GeneratedDeck` (Java record — in-memory only)

Parsed line from the generated-decks file, used by the self-play match generator.

| Field | Type | Description |
|---|---|---|
| `setCode` | `String` | Set code |
| `cardNames` | `List<String>` | 40 card names |

### `GeneratedDecksIndex` (Java class — in-memory only)

In-memory index of the generated-decks file for efficient random access.

| Field | Type | Description |
|---|---|---|
| `allDecks` | `List<GeneratedDeck>` | All loaded decks |
| `decksBySet` | `Map<String, List<GeneratedDeck>>` | Decks grouped by set code |

**Methods**:
- `randomDeck(Random)` → `GeneratedDeck` — pick any random deck
- `randomDeckFromSet(String setCode, GeneratedDeck exclude, Random)` → `GeneratedDeck` — pick a random deck from the same set, excluding a specific deck (for method 5)

## Entities (Extended)

### `EvaluateScorerConfig` — add optional `set_code`

| Field | Type | Default | Description |
|---|---|---|---|
| `set_code` | `str \| None` | `None` | When `None`, a random sealed-legal set is selected. When set, all pools use this set. |

### `MatchWorkerConnector` — add optional `generated_decks_path`

The `start()` method gains an optional `generated_decks_path: Path | None` parameter. When not `None`, passes `-Dgenerated.decks.file=<path>` as a system property to the Java worker.

### `MatchOutcomeSupervisor` — add optional `generated_decks_path`

Constructor gains `generated_decks_path: Path | None = None`. Forwarded to `MatchWorkerConnector.start()`.

### `PoolConnector` — support omitting set code

`generate()` gains support for `set_code=None`. When `None`, the `--set` argument is omitted from the Java command, and `PoolMain` selects random sets per pool.

## State Transitions

### Self-Play Refinement Loop

```
[pools.txt] ──generate-pools──→ [generated-decks.txt] ──build-decks──→
    │                                                         │
    │                                                         ▼
    │                                          [match-outcomes.txt] ──match-outcomes──→
    │                                                         │        (self-play mode)
    │                                                         ▼
    │                                              [scorer checkpoint] ──train-scorer──→
    │                                                         │
    └─────────────────────────── (repeat with new scorer) ◄───┘
```

Each iteration:
1. `generate-pools` → `pools.txt` (random sets, ~10k pools)
2. `build-decks` → `generated-decks.txt` (scorer-guided 40-card decks)
3. `match-outcomes --generated-decks-path` → appends to `match-outcomes.txt`
4. `train-scorer` → new scorer checkpoint
5. `evaluate-scorer` → win rate report
6. Repeat from step 1 with the new scorer
