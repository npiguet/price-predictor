# Research: Card Price Predictor

**Date**: 2026-03-01 (revised)
**Feature**: 001-card-price-predictor

## R1: Card Data Source — Forge Card Scripts

### Decision: Parse Forge card scripts as the canonical card attribute source

**Rationale**: Forge card scripts are the authoritative source of card
game data in this project. They contain oracle-level card attributes
(one file per card, not per printing) in a simple key-value text
format. Using Forge scripts instead of MTGJSON's AllIdentifiers.json
aligns with the spec's assumption that card data comes from the local
Forge checkout at `../forge`.

**Format**: Each card is a `.txt` file in
`../forge/forge-gui/res/cardsfolder/` organized in alphabetical
subdirectories (a/ through z/). There are ~32,000 card scripts.

**Key fields extractable for feature engineering**:

| Forge Field | Example | Feature Use |
|-------------|---------|-------------|
| `Name` | `Lightning Bolt` | Join key to MTGJSON prices |
| `ManaCost` | `2 W W` | Mana value, color distribution |
| `Types` | `Legendary Creature Human Wizard` | Card type, supertypes, subtypes |
| `PT` | `3/4` | Power, toughness |
| `Loyalty` | `3` | Planeswalker loyalty |
| `K:` lines | `Flying`, `Kicker:2 R` | Keyword abilities |
| `Oracle` | Rules text | TF-IDF text features |
| `AlternateMode` | `DoubleFaced`, `Split` | Card layout |
| `A:` lines | Ability definitions | Ability count/complexity |
| `T:` lines | Triggered abilities | Trigger count |
| `S:` lines | Static abilities | Static ability count |

**Multi-face cards**: Cards with `AlternateMode` (DoubleFaced, Split,
Adventure, Modal, Flip) contain an `ALTERNATE` separator followed by
the back face. For pricing we use the front face name. For feature
engineering we use only the front face attributes (the card is priced
as a unit).

**Fields NOT available in Forge scripts** (confirmed by spec):
- `rarity` — printing-specific, excluded by design
- `setCode` — not present, not needed
- `uuid` — printing-specific, not present

**Alternatives considered**:
- AllIdentifiers.json (MTGJSON) — keyed by UUID (printing-level, not
  card-level). Would give rarity and set info but those are explicitly
  excluded. Adds a large file (~500MB) when Forge scripts already
  contain all needed game attributes. Rejected.

## R2: Price Data Source — Cardmarket EUR via MTGJSON

### Decision: Use AllPricesToday.json → Cardmarket → retail → EUR

**Rationale**: Per spec clarification, EUR from Cardmarket is the
authoritative price source. Cardmarket prices are less volatile than
TCGPlayer USD and the model trains natively on EUR.

**Price extraction path**:
```
data[uuid].paper.cardmarket.retail.normal["2026-02-26"] → EUR float
data[uuid].paper.cardmarket.retail.foil["2026-02-26"]  → EUR float
```

**File**: `resources/AllPricesToday.json` (~52 MB), frozen snapshot
dated 2026-02-26. Keyed by printing UUID.

**Currency**: EUR (confirmed via `currency` field).

**Finish types available**: `normal` and `foil` only (no `etched` for
Cardmarket). Buylist section is consistently empty.

**Date format**: ISO 8601 `YYYY-MM-DD`. Each finish type maps a single
date to a price float. Use the most recent (only) date entry.

**Alternatives considered**:
- TCGPlayer USD — original choice. Rejected per spec clarification:
  Cardmarket EUR is authoritative and less volatile.
- Scryfall API — requires runtime HTTP calls, violates frozen-data
  requirement. Rejected.

## R3: Name-to-UUID Mapping — AllPrintings.json

### Decision: Use AllPrintings.json to build card name → UUID(s) index

**Rationale**: Forge card scripts are keyed by card name.
AllPricesToday.json is keyed by printing UUID. A mapping is needed to
join them. AllPrintings.json (`resources/AllPrintings.json`, ~512 MB)
contains card entries organized by set code, where each entry has both
`name` and `uuid` fields.

**Strategy**:
1. Stream AllPrintings.json → extract all `(name, uuid)` pairs
2. Filter to paper-available, English, non-funny, non-online-only
   entries (using `availability`, `isFunny`, `isOnlineOnly` fields)
3. Build a `name → [uuid, uuid, ...]` mapping (one name, many
   printings)
4. For each Forge card name, look up all UUIDs
5. For each UUID, fetch Cardmarket EUR price from AllPricesToday
6. Take the **cheapest price** across all printings (normal + foil) per
   the spec clarification

**Cheapest price algorithm**:
```
For card_name in forge_scripts:
    uuids = name_to_uuids[card_name]
    prices = []
    for uuid in uuids:
        normal_price = allprices[uuid].paper.cardmarket.retail.normal
        foil_price = allprices[uuid].paper.cardmarket.retail.foil
        prices.extend([p for p in [normal_price, foil_price] if p])
    cheapest = min(prices) if prices else None
```

Cards with no price across any printing are excluded from training
(FR-004a).

**Alternatives considered**:
- AllIdentifiers.json — keyed by UUID, could also provide the mapping.
  But it's not in the resources directory and AllPrintings already
  serves the purpose. Rejected.
- Fuzzy name matching — unnecessary since Forge card names match
  MTGJSON names exactly (both use official Oracle names). Direct
  string matching is sufficient.

## R4: Technology Stack

### Decision: Python 3.11+ with scikit-learn

**Rationale**: Python is the standard language for tabular ML. The
project is a price prediction tool with a CLI interface. scikit-learn
provides gradient boosting out of the box with minimal setup. pandas
handles the large JSON data loading efficiently.

Note: Per Constitution v2.0.0, the main application MAY use any
technology stack. Python is the optimal choice for ML workloads. The
Java stub library for MTG Forge integration is a separate feature
(002-forge-api-integration).

**Dependencies**:

| Package | Purpose |
|---------|---------|
| scikit-learn | ML models (GradientBoostingRegressor), metrics |
| pandas | Data loading, feature engineering |
| numpy | Numeric operations |
| pytest | Testing framework |
| joblib | Model serialization (bundled with scikit-learn) |
| ruff | Linting and formatting |

**Alternatives considered**:
- Java/Kotlin with Smile or Tribuo — viable but far more friction for
  ML prototyping. Constitution v2.0.0 no longer requires JVM for the
  main app. Rejected per Simplicity First.
- XGBoost/LightGBM — more powerful gradient boosting but adds external
  C++ dependencies. Can upgrade later if needed. Rejected for v1 per
  Simplicity First.
- Deep learning (PyTorch/TensorFlow) — overkill for tabular data with
  ~15 features. Rejected per Simplicity First.

## R5: ML Approach

### Decision: Gradient Boosted Trees (scikit-learn)

**Rationale**: Card price prediction is a tabular regression problem.
The input is a mix of categorical (types, colors), numeric (mana
value, P/T), and text (oracle text) features. Gradient boosted trees
are the standard first choice for tabular regression.

**Feature engineering plan** (updated — no rarity):

| # | Source Field | Feature Type | Method |
|---|-------------|-------------|--------|
| 1 | `ManaCost` → mana value | numeric | Parse total CMC from Forge format |
| 2 | `ManaCost` → colors | multi-hot | W, U, B, R, G → 5 binary columns |
| 3 | `ManaCost` → color count | numeric | Number of distinct colors |
| 4 | `ManaCost` → generic mana | numeric | Generic mana component |
| 5 | `Types` → card types | multi-hot | Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle |
| 6 | `Types` → supertypes | multi-hot | Legendary, Basic, Snow, etc. |
| 7 | `Types` → subtypes | numeric | Count of subtypes |
| 8 | `K:` → keywords | multi-hot | Top-30 most common keywords |
| 9 | `K:` → keyword count | numeric | Total keyword count |
| 10 | `Oracle` → text | TF-IDF | Top 500 terms, limited vocabulary |
| 11 | `Oracle` → text length | numeric | Character count of oracle text |
| 12 | `PT` → power | numeric | Parse "*" as NaN + indicator column |
| 13 | `PT` → toughness | numeric | Parse "*" as NaN + indicator column |
| 14 | `Loyalty` | numeric | Null for non-planeswalkers |
| 15 | `A:`/`T:`/`S:` → ability count | numeric | Total defined abilities |
| 16 | `AlternateMode` | categorical | Layout type (one-hot) |

**Excluded features** (per spec):
- `rarity` — not available in Forge scripts, excluded by design
- `setCode` — not available, would leak set-specific pricing info

**Target variable**: log-transformed EUR price from Cardmarket retail.
Log transform because price distribution is heavily right-skewed
(many cheap cards, few expensive ones). Predictions are
exp-transformed back to EUR.

**Train/test split**: 80/20 random split with fixed seed (42) for
reproducibility. No stratification by rarity (not available).

**Alternatives considered**:
- Random Forest — simpler, slightly less accurate. Good baseline.
- Linear regression — too limited for nonlinear price relationships.
  Rejected.
- Neural network on oracle text embeddings — violates Simplicity First
  for v1. Could be a future enhancement.

## R6: Data Filtering

### Decision: Filter to paper-available, English, non-funny cards

**Rationale**: Many entries in AllPrintings.json are online-only,
oversized, joke cards, or non-English printings. These either lack
paper Cardmarket prices or would confuse the name-to-UUID mapping.

**Filters applied during name-to-UUID mapping**:
- `availability` must include `"paper"`
- `language` must be `"English"` (skip foreignData entries)
- `isFunny` must be false or absent
- `isOversized` must be false or absent
- `isOnlineOnly` must be false or absent

**Filters applied during price extraction**:
- Must have a non-null Cardmarket EUR price in AllPricesToday.json
- Price must be > €0.00

**Forge script filtering**:
- Skip files that fail to parse (report errors per FR-007)
- Use only front face of multi-face cards for name matching
- Skip token/emblem scripts if any exist

## R7: Forge Card Script Parsing

### Decision: Custom line-based parser for Forge `.txt` format

**Rationale**: Forge card scripts are simple key-value text files. No
existing library parses this format. A custom parser is straightforward
and aligns with Simplicity First.

**Parsing strategy**:
1. Read file line by line
2. Split on first `:` to get key-value pairs
3. Handle multi-face cards: split on blank line + `ALTERNATE`
4. Extract front face only for pricing/feature purposes
5. Parse `ManaCost` field: space-separated mana shards → total CMC,
   per-color counts, generic mana
6. Parse `Types` field: split by spaces, classify into supertypes,
   card types, and subtypes using known type lists
7. Parse `PT` field: split on `/` → power, toughness strings
8. Collect `K:` lines: extract keyword name (before first `:` param)
9. Collect `A:`, `T:`, `S:` lines: count for complexity features
10. Extract `Oracle` text for TF-IDF

**Known type classification**:
- Supertypes: `Legendary`, `Basic`, `Snow`, `World`, `Ongoing`, `Host`
- Card types: `Creature`, `Instant`, `Sorcery`, `Enchantment`,
  `Artifact`, `Planeswalker`, `Land`, `Battle`, `Tribal`, `Kindred`
- Everything else in `Types` line → subtypes

**ManaCost parsing** (Forge format):
- Space-separated shards: `2 W W` → CMC 4, W=2
- Hybrid: `WU` → 0.5 W + 0.5 U for color, 1 CMC
- Phyrexian: `WP` → W color, 1 CMC
- X: `X` → 0 for CMC calculation, flag X-present
- `no cost` → CMC 0, no colors

**Error handling**: Cards that fail parsing are logged with the
filename and error reason, then skipped. Training continues with
valid cards (FR-007).
/o  