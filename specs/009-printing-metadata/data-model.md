# Data Model: 009 Printing Data Fields

## Domain Entities

### PrintingData (NEW — value object)

**Location**: `src/price_predictor/domain/value_objects.py`

Immutable value object representing the five printing data fields for a single card.

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `is_reserved` | `bool` | Whether the card is on the MTG reserve list | `False` |
| `rarity` | `str` | Rarity of the cheapest printing (lowercase: "common", "uncommon", "rare", "mythic") | `"rare"` |
| `printings_count` | `int` | Number of distinct sets the card has been printed in | `1` |
| `set_code` | `str` | Set code of the cheapest printing (lowercase) | `"ukn"` |
| `legalities` | `list[str]` | List of recognized formats where the card is legal (lowercase) | all 10 formats |

**Validation**:
- `rarity` must be one of: `"common"`, `"uncommon"`, `"rare"`, `"mythic"`, `"special"`, `"bonus"`
- `printings_count` must be >= 1
- `set_code` must be non-empty, lowercased
- Each entry in `legalities` must be from the 10 recognized formats

**Factory methods**:
- `PrintingData.defaults()` → returns instance with all default values (FR-005)
- `PrintingData.from_allprintings(card_data: dict, cheapest_uuid_data: dict)` → extracts fields from MTGJSON data

### Card (MODIFIED)

**Location**: `src/price_predictor/domain/entities.py`

Add one optional field:

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `printing_data` | `PrintingData \| None` | Printing metadata for this card | `None` |

The field is `None` when parsing raw card text that doesn't contain metadata lines. It is populated:
- During training, after joining to AllPrintings
- During prediction, after auto-fill or defaulting

## Data Relationships

```
AllPrintings.json                AllPricesToday.json
       │                               │
       ├── card name → [UUID, ...]      ├── UUID → price
       ├── UUID → rarity, setCode       │
       ├── card → isReserved            │
       ├── card → legalities            │
       └── card → printings (set list)  │
              │                         │
              └─────── join on UUID ────┘
                          │
                    MetadataMap
                 dict[str, PrintingData]
                   (card name → metadata
                    from cheapest printing)
                          │
              ┌───────────┴──────────┐
              ▼                      ▼
    Training Pipeline          Prediction API
    (enrich text with          (auto-fill from map,
     metadata at load time)     or apply defaults)
```

## Text Format (enriched card text)

Metadata lines are appended at the end of the converted card text:

```
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
reserved: false
rarity: uncommon
printings: 23
set: 2xm
legalities: modern, legacy, vintage, pauper, commander, penny, oathbreaker
```

### Parsing rules

The five metadata field keys are:
- `reserved:` → parsed as boolean string (`"true"` or `"false"`)
- `rarity:` → parsed as lowercase string
- `printings:` → parsed as integer string
- `set:` → parsed as lowercase string
- `legalities:` → parsed as comma-space-separated list of format names

These are recognized by the card text parser as regular key:value lines (they don't match the ability-line regex). The parser extracts them into the `Card.printing_data` field.

## Feature Engineering (sklearn)

17 new dense features appended to the existing feature vector:

| Feature | Type | Description |
|---------|------|-------------|
| `is_reserved` | binary | 1.0 if reserved, 0.0 otherwise |
| `rarity_common` | binary | 1.0 if rarity is "common" |
| `rarity_uncommon` | binary | 1.0 if rarity is "uncommon" |
| `rarity_rare` | binary | 1.0 if rarity is "rare" (or special/bonus) |
| `rarity_mythic` | binary | 1.0 if rarity is "mythic" |
| `printings_count` | numeric | Number of printings |
| `legalities_count` | numeric | Number of legal formats |
| `legal_standard` | binary | 1.0 if legal in Standard |
| `legal_pioneer` | binary | 1.0 if legal in Pioneer |
| `legal_modern` | binary | 1.0 if legal in Modern |
| `legal_brawl` | binary | 1.0 if legal in Brawl |
| `legal_legacy` | binary | 1.0 if legal in Legacy |
| `legal_vintage` | binary | 1.0 if legal in Vintage |
| `legal_pauper` | binary | 1.0 if legal in Pauper |
| `legal_commander` | binary | 1.0 if legal in Commander |
| `legal_penny` | binary | 1.0 if legal in Penny Dreadful |
| `legal_oathbreaker` | binary | 1.0 if legal in Oathbreaker |

When `Card.printing_data` is `None`, all 17 features default to 0.0.
