# Data Model: Sealed Training Data Generation

## Entities

### MatchOutcome

A single recorded result of a best-of-3 sealed match between two AI-built decks.

| Field | Type | Description |
|-------|------|-------------|
| deck_a | list[str] | Card names in deck A (40 cards, includes basic lands, duplicates repeat) |
| deck_b | list[str] | Card names in deck B (40 cards, includes basic lands, duplicates repeat) |
| wins_a | int | Games won by deck A (0-2) |
| wins_b | int | Games won by deck B (0-2) |

**Validation rules**:
- `len(deck_a) == 40` and `len(deck_b) == 40`
- `wins_a + wins_b in {2, 3}`
- `wins_a in {0, 1, 2}` and `wins_b in {0, 1, 2}`
- Each card name is a non-empty string (Forge canonical name)

**Serialized format** (one line in `match-outcomes.txt`):
```
card1|card2|...|card40;card1|card2|...|card40;wins_a;wins_b
```

### BoosterPool

A collection of cards from 6 boosters of a single expansion set. Two pools are generated per match (one per player). Not persisted — intermediate in-memory entity.

| Field | Type | Description |
|-------|------|-------------|
| set_code | str | MTG expansion set code (e.g. "MH3") |
| cards | list[PaperCard] | 84-90 cards from 6 boosters (includes basic lands) |

**Validation rules**:
- `len(cards)` typically 84-90 (varies by set, 14-15 cards per booster x 6)
- All cards belong to the specified set (or basic lands)

### Deck

A 40-card deck constructed from a booster pool. Not persisted directly — serialized as part of MatchOutcome.

| Field | Type | Description |
|-------|------|-------------|
| cards | list[PaperCard] | Exactly 40 cards (nonland spells + basic lands) |
| construction_method | int | Which method was used (1-4), not serialized |

**Validation rules**:
- `len(cards) == 40`
- Each card instance appears at most once (matching physical sealed rules)
- For methods 2-4: basic lands rebalanced per pip-proportional algorithm

### DeckConstructionMethod

Enum-like selection determining how a deck is built from a pool. Not persisted.

| Value | Weight | Description |
|-------|--------|-------------|
| 1 | 40% | Standard Forge sealed deck builder |
| 2 | 30% | Forge builder + 3 nonland card swaps |
| 3 | 20% | Forge builder + 8 nonland card swaps |
| 4 | 10% | Random 23 nonland cards from pool |

### ExpansionSet

A sealed-legal MTG expansion. Not a stored entity — derived from Forge's booster database at runtime.

| Field | Type | Description |
|-------|------|-------------|
| code | str | Set code (e.g. "MH3", "BLB", "RVR") |

**Validation rules**:
- Must have a booster template in `StaticData.instance().getBoosters()`
- Implicitly excludes un-sets, aftermath sets, and unsupported sets

## Relationships

```
ExpansionSet  1 ──── * BoosterPool     (one set selected per match, 2 pools generated)
BoosterPool   1 ──── 1 Deck            (one deck built per pool)
Deck          2 ──── 1 MatchOutcome    (two decks produce one match result)
```

## State Transitions

### Worker Lifecycle

```
STARTING → RUNNING → EXITED (crash/completion)
                         ↓
                    RESTARTING → RUNNING  (supervisor auto-restart)
```

### Match Generation Pipeline (per worker iteration)

```
SELECT_SET → GENERATE_POOLS → BUILD_DECKS → PLAY_MATCH → WRITE_RESULT
     ↑                                                         |
     └─────────────────────────────────────────────────────────┘
```

## Storage

### Output File

- **Path**: `./output/sealed/match-outcomes.txt`
- **Format**: Append-only flat text, one MatchOutcome per line
- **Encoding**: UTF-8
- **Concurrency**: Multiple workers append independently; each write is a single complete line
- **Persistence**: Accumulates across runs (never truncated by the application)
