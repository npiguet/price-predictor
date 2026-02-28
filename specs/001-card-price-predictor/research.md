# Research: Card Price Predictor

**Date**: 2026-02-26
**Feature**: 001-card-price-predictor

## R1: MTGJSON Data Format

### Decision: Use AllPricesToday.json + AllIdentifiers.json

**Rationale**: AllPricesToday.json contains prices keyed by UUID but no
card attributes. AllIdentifiers.json contains card attributes keyed by
the same UUID, enabling a direct join. Both files are downloaded once
and frozen in `resources/`.

**Alternatives considered**:
- AtomicCards.json — keyed by card name, not UUID. Would require a
  two-hop lookup. Lacks printing-specific fields (rarity, setCode).
  Rejected.
- AllPrintings.json — organized by set code, not UUID. Requires
  navigating set hierarchy. Much larger file. Rejected.
- Scryfall API — would require runtime HTTP calls, violating the
  frozen-data requirement. Rejected.

### AllPricesToday.json Schema

```
data → {uuid} → paper → tcgplayer → retail → normal → {"2026-02-26": 0.25}
```

- Top-level: `meta` + `data` (dict keyed by UUID)
- Each UUID maps to `PriceFormats`: `paper` and/or `mtgo`
- Paper providers: tcgplayer (USD), cardkingdom (USD), cardmarket (EUR),
  cardsphere (USD)
- Each provider has: `currency`, optional `retail`, optional `buylist`
- Each of those has optional `normal`, `foil`, `etched` finish types
- Each finish maps date strings to price floats

**Price extraction strategy**: Use `paper → tcgplayer → retail → normal`
as the primary USD price. Fall back to `cardkingdom → retail → normal`
if tcgplayer is missing. Take the most recent date's price.

### AllIdentifiers.json Schema

```
data → {uuid} → Card (Set) object
```

Key fields for feature engineering:
- `manaCost` (string, e.g., "{2}{W}{W}") — optional
- `manaValue` (number, converted mana cost) — required
- `colors` (string[], e.g., ["W", "U"]) — required
- `colorIdentity` (string[]) — required
- `types` (string[], e.g., ["Creature"]) — required
- `supertypes` (string[], e.g., ["Legendary"]) — required
- `subtypes` (string[], e.g., ["Human", "Wizard"]) — required
- `text` (string, oracle text) — optional
- `keywords` (string[], e.g., ["Flying"]) — optional
- `power` (string, handles "*") — optional
- `toughness` (string, handles "*") — optional
- `loyalty` (string) — optional
- `rarity` (string: common/uncommon/rare/mythic) — required
- `setCode` (string) — required
- `layout` (string: normal/transform/split/etc.) — required
- `edhrecRank` (number) — optional

## R2: Technology Stack

### Decision: Python 3.11+ with scikit-learn

**Rationale**: Python is the standard language for tabular ML. The
project is a price prediction tool (not a web service), so a CLI +
library approach is simplest. scikit-learn provides Random Forest and
gradient boosting out of the box with minimal setup. pandas handles
the JSON data loading and transformation efficiently.

**Alternatives considered**:
- Java/Kotlin with Smile or Tribuo — viable but far more friction for
  ML prototyping. Fewer community examples for this type of task.
  Rejected per Simplicity First principle.
- Python with XGBoost/LightGBM — more powerful gradient boosting, but
  adds external C++ dependencies. Can be introduced later if
  scikit-learn's GradientBoostingRegressor is insufficient. Rejected
  for v1 per Simplicity First.
- Deep learning (PyTorch/TensorFlow) — massive overkill for tabular
  data with ~10 features. Rejected per Simplicity First.

### Dependencies

| Package | Purpose |
|---------|---------|
| scikit-learn | ML models (RandomForest, GradientBoosting), metrics |
| pandas | Data loading, feature engineering |
| numpy | Numeric operations |
| pytest | Testing framework |
| joblib | Model serialization (bundled with scikit-learn) |

## R3: ML Approach

### Decision: Gradient Boosted Trees (scikit-learn)

**Rationale**: Card price prediction is a tabular regression problem.
The input is a mix of categorical (types, colors, rarity), numeric
(mana value, P/T), and text (oracle text) features. Gradient boosted
trees are the standard first choice for tabular regression — they
handle mixed feature types well, are fast to train, and produce
interpretable feature importances.

**Feature engineering plan**:
1. `manaValue` — numeric, use directly
2. `colors` — multi-hot encode (W, U, B, R, G → 5 binary columns)
3. `colorIdentity` — multi-hot encode (5 binary columns)
4. `types` — multi-hot encode (Creature, Instant, Sorcery,
   Enchantment, Artifact, Planeswalker, Land, Battle)
5. `supertypes` — multi-hot encode (Legendary, Basic, Snow, etc.)
6. `subtypes` — frequency-based: count of subtypes only (too many
   unique values for one-hot)
7. `keywords` — multi-hot for top-N most common keywords (e.g., top
   30), remainder bucketed
8. `text` (oracle text) — TF-IDF vectorization with limited
   vocabulary (top 500 terms)
9. `power`, `toughness` — numeric (parse "*" as NaN, handle with
   imputation or indicator column)
10. `loyalty` — numeric (null for non-planeswalkers)
11. `rarity` — ordinal encode (common=0, uncommon=1, rare=2, mythic=3)
12. `setCode` — not used directly (too many unique values, leaks
    set-specific pricing info). May revisit.

**Target variable**: log-transformed USD price from TCGPlayer retail
normal. Log transform because price distribution is heavily
right-skewed (many cheap cards, few expensive ones).

**Train/test split**: 80/20 random split, stratified by rarity to
ensure representation of rare/mythic cards in both sets.

**Alternatives considered**:
- Random Forest — simpler but generally slightly less accurate than
  gradient boosting on tabular data. Good baseline.
- Linear regression with engineered features — too limited for the
  nonlinear price relationships. Rejected.
- Neural network on oracle text embeddings — interesting but violates
  Simplicity First for v1. Could be a future enhancement.

## R4: Data Filtering

### Decision: Filter to paper-available, English, non-funny cards

**Rationale**: Many entries in AllIdentifiers.json are online-only
(MTGO, Arena), oversized, joke cards, or non-English printings. These
either lack paper prices or would confuse the model.

**Filters**:
- `availability` must include "paper"
- `language` must be "English"
- `isFunny` must be false or absent
- `isOversized` must be false or absent
- `isOnlineOnly` must be false or absent
- Must have a non-null price in AllPricesToday.json
- Price must be > $0.00 (exclude cards with zero/missing prices)
