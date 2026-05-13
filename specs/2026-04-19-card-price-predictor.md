# Goal

Predict Magic: The Gathering card EUR market prices from game-visible attributes (mana cost, types, oracle text,
power/toughness, keywords) and printing metadata (reserve list status, rarity, reprint count, format legalities). The
system works for both real cards and hypothetical ones — predicting the price of custom or unreleased cards based on
learned patterns from the entire real card catalog is the primary use case. A secondary role: the transformer model's
internal representations serve as card embeddings for the sealed deck picker (see `specs/2026-03-28-sealed-deck-picker.md`).

# Overall Approach

Two model architectures coexist, trained on the same data pipeline and exposed through the same CLI and REST API:

- **sklearn GradientBoostingRegressor** — tabular regression on hand-engineered features. Fast to train (~2 minutes on
  CPU), interpretable feature importances, good baseline accuracy.
- **Custom transformer encoder** — encoder-only transformer trained on tokenized card text with a metadata side-channel.
  Learns directly from the full card text; produces reusable card embeddings for downstream ML tasks.

Both models are always returned when a prediction is requested — the consumer can compare or use whichever is more
appropriate for their context.

## Why Two Models

The sklearn model was the original and remains useful: it trains in minutes, runs without a GPU, and its feature
importances are directly inspectable. The transformer was added for three reasons:

1. **Learn from text without lossy feature engineering.** TF-IDF over oracle text (which the sklearn model uses) captures
   word frequency but not word order, negation, or ability interactions. The transformer reads the full structured card
   text and can learn patterns like "this removal spell also draws a card" or "this creature has hexproof and is
   undercosted."
2. **Produce reusable card embeddings.** The transformer's internal representation of a card — a 512-dimensional vector
   capturing what the card does — is the foundation for the sealed deck picker's card encoder. This was not possible
   with the sklearn model.
3. **Capture domain structure.** Mana symbols, ability types, and keywords have structural relationships
   (e.g., `{W}{W}` signals strong white commitment) that the transformer can learn as embedding geometry.

# Data Pipeline

## Card Text Conversion

MTG Forge stores card data as internal scripts optimized for its game engine — compact but full of engine metadata
(`SVar:`, `DeckHints:`, `AI:` lines), Forge-specific syntax, and non-standard formatting. These are not suitable as
model input.

The Java converter (`ConvertMain` in the `forge-connector` module) transforms each Forge script into a clean,
lowercase, English-keyed text format. The conversion:

- Strips all engine metadata and Forge-internal syntax
- Translates abilities into Oracle-text-matching English with classified type prefixes (`activated[N]:`, `triggered:`,
  `static:`, `spell[N]:`, `planeswalker[N]:`, etc.)
- Removes reminder text (parenthetical rules explanations)
- Preserves `CARDNAME` and `NICKNAME` as literal uppercase placeholders for self-reference
- Writes one `.txt` file per card to `output/cardsfolder/`, mirroring Forge's alphabetical subdirectory structure

A typical converted card (Lightning Bolt):

```
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
```

A planeswalker (Jace, the Mind Sculptor):

```
name: jace, the mind sculptor
mana cost: {2}{U}{U}
types: legendary planeswalker jace
loyalty: 3
planeswalker[1]: [+2]: look at the top card of target player's library. ...
planeswalker[2]: 0: draw three cards, then put two cards from your hand ...
planeswalker[3]: [-1]: return target creature to its owner's hand.
planeswalker[4]: [-12]: exile all cards from target player's library, ...
replacement: CARDNAME enters with three loyalty counters on it.
```

Multi-face cards use `ALTERNATE` as a separator, with each face carrying its own properties and abilities:

```
layout: adventure
name: bonecrusher giant
mana cost: {2}{R}
types: creature giant
power toughness: 4/3
triggered: when CARDNAME becomes the target of a spell, CARDNAME deals 2 damage ...

ALTERNATE

name: stomp
mana cost: {1}{R}
types: instant adventure
spell[1]: damage can't be prevented this turn. CARDNAME deals 2 damage to any target.
```

```
layout: modal
name: emeria's call
mana cost: {4}{W}{W}{W}
types: sorcery
spell[1]: create two 4/4 white angel warrior creature tokens with flying. ...

ALTERNATE

name: emeria, shattered skyclave
types: land
activated[1]: {T}: add {W}.
replacement: as CARDNAME enters, you may pay 3 life. if you don't, it enters tapped.
```

### Name Line Handling

The `name:` line is present in every converted card but is **stripped before model input** for both the sklearn parser
and the transformer tokenizer. The model should learn to predict prices from what the card does, not from its name.
A card named "Jace" is not inherently expensive — it's expensive because of what its abilities do. Stripping the name
also ensures the model generalizes to hypothetical cards that have no real name.

For the transformer, the tokenizer replaces the name value with the `CARDNAME` placeholder token before tokenization,
so the model sees that a name line exists without learning name-specific patterns.

## Price Data

Prices come from MTGJSON's `AllPricesToday.json`, specifically the Cardmarket EUR section. USD/TCGPlayer prices are
ignored — EUR from Cardmarket is the sole training signal, chosen for its lower volatility compared to alternatives.

Since Forge card scripts represent cards at the oracle level (one entry per unique card, not per printing), many cards
have multiple printings with different prices. The system selects the **cheapest price across all printings** — including
foil, non-foil, promo, and special versions — as the training label. This represents the floor price attributable to
the card's game mechanics alone, stripped of collectibility and scarcity premiums from limited printings.

A price floor of **€0.01** is applied: any card priced below €0.01 is clamped to €0.01 before use as a training label.
This avoids `log(0)` errors during training while still representing near-worthless cards.

Cards with no price in any printing are excluded from the training set entirely.

## Printing Metadata

Five fields enrich each card's representation with information from `AllPrintings.json` that is not visible in the
oracle text but strongly influences price:

| Field | Source | Default (unknown cards) |
|-------|--------|------------------------|
| `is_reserved` | MTGJSON `isReserved` flag | `false` |
| `rarity` | Rarity of the cheapest printing | `rare` |
| `printings_count` | Number of distinct sets containing the card | `1` |
| `release_year` | Release year of the cheapest printing's set | `1993` |
| `legalities` | Constructed formats where the card is legal | all 10 formats |
| `is_abu` | Whether the cheapest printing is from Alpha/Beta/Unlimited | `false` |

The 10 recognized constructed formats: standard, pioneer, modern, brawl, legacy, vintage, pauper, commander, penny,
oathbreaker.

For **known cards** (name matches an entry in AllPrintings), all fields are auto-filled from the cheapest printing's
data. For **unknown cards** (hypothetical or unreleased), defaults are applied — these defaults are deliberately
optimistic (legal everywhere, not reserved, rare) so that the model does not penalize cards it has never seen.

### Why Printing Metadata

Some of the strongest price signals are invisible in oracle text:

- **Reserve list** cards cannot be reprinted, creating artificial scarcity that makes even mediocre cards expensive.
  Without this signal, the model cannot explain why Revised dual lands cost hundreds of euros.
- **Reprint count** inversely correlates with price — a card printed in 15 sets has far more supply than one printed
  once.
- **Format legality** directly affects demand — a card banned in Modern loses a major demand source.
- **Release year** captures historical context — older cards tend to have lower supply.
- **Alpha/Beta/Unlimited** printings have extreme collectibility premiums.

These fields are provided as a metadata side-channel rather than embedded in the card text, because they are
numeric or categorical values that would be poorly represented as text tokens.

## Training Data Assembly

The pipeline to produce training examples:

1. **Scan** `output/cardsfolder/` for converted `.txt` files (~32,000 cards)
2. **Load** `AllPricesToday.json` and match each card by name to its cheapest EUR Cardmarket price
3. **Load** `AllPrintings.json` and build a `PrintingData` object for each matched card (reserve list, rarity,
   printings count, release year, legalities, ABU status)
4. **Pair** each card's text + metadata with its price to produce `(text, price, PrintingData)` training tuples
5. **Split** into 80% train / 20% test sets (random, seeded for reproducibility)

Cards that fail parsing or have no matching price are skipped and reported. Typical yield: ~24,000 usable training
examples from ~32,000 converted scripts.

# Converted Card Text Format

The canonical converted format uses lowercase property lines and classified ability lines. This is the input consumed
by both the sklearn card parser and the transformer tokenizer.

## Property Lines

| Prefix | Description | Example |
|--------|-------------|---------|
| `name:` | Card name (lowercase) | `name: lightning bolt` |
| `mana cost:` | Brace-delimited mana symbols | `mana cost: {2}{U}{U}` |
| `types:` | All types in order: supertypes, card types, subtypes | `types: legendary creature human wizard` |
| `power toughness:` | P/T separated by `/` | `power toughness: 3/4` |
| `loyalty:` | Starting loyalty (planeswalkers) | `loyalty: 3` |
| `defense:` | Starting defense (battles) | `defense: 5` |
| `layout:` | Multi-face layout type (omitted for normal cards) | `layout: adventure` |
| `text:` | Rules text that doesn't fit other categories (casting restrictions, etc.) | `text: cast CARDNAME only during combat ...` |
| `draft:` | Draft-matters text (Conspiracy-style cards) | `draft: reveal CARDNAME as you draft it.` |

## Ability Line Prefixes

| Prefix | Usage | Numbered? |
|--------|-------|-----------|
| `spell[N]:` | Instant/sorcery effects | Yes |
| `activated[N]:` | Activated abilities (`cost: effect`) | Yes |
| `triggered:` | Triggered abilities (`when/whenever/at`, including ETB/LTB) | No |
| `static:` | Static abilities and passive keywords (flying, lifelink, etc.) | No |
| `replacement:` | Replacement effects | No |
| `planeswalker[N]:` | Loyalty abilities with `[+N]/[-N]/[0]:` prefix | Yes |
| `option[N]:` | Modal choice options | Yes |
| `chapter:` | Saga chapter abilities (value starts with roman numeral) | No |
| `level[N]:` | Class level abilities | Yes |
| `alternate cost:` | Alternative casting costs (flashback, etc.) | No |
| `additional cost:` | Additional costs (kicker, sacrifice, etc.) | No |
| `cost reduction:` | Cost reduction abilities (convoke, etc.) | No |

Numbered prefixes use sequential `[N]` labels starting at 1, resetting per face for multi-face cards.

Chapter lines encode the chapter number as part of the value: `chapter: I — effect`, `chapter: II, III — effect`.

## Multi-Face Cards

Multi-face cards (transform, split, adventure, modal DFC) use `ALTERNATE` on its own line to separate faces. Each
face carries its own property lines and abilities with independent numbering.

# Custom Tokenizer

The transformer reads card text through a word-level tokenizer with a compact domain vocabulary, not a general-purpose
tokenizer like BERT's WordPiece (~30,000 tokens).

## Vocabulary

The vocabulary is built by `python -m price_predictor vocabulary`, which:

1. **Collects domain terms** from structured fields: all card types, supertypes, subtypes, keyword abilities, game
   zones, colors, and property-line prefixes from the converted card corpus.
2. **Scans corpus frequency**: every word appearing 5+ times across all converted card files is included.
3. **Seeds set codes**: alphabetic fragments of every set code from `AllPrintings.json` are added, so metadata-enriched
   card texts never produce UNK for set identifiers.
4. **Normalizes multi-word keywords** to underscore form: "first strike" becomes `first_strike`, "double strike"
   becomes `double_strike`. These compound keywords are recognized before word splitting.

The result: ~5,000-6,000 tokens covering ~98% of all token occurrences in the corpus. The remaining ~2% is almost
entirely card-specific proper nouns (character names, named cards) that do not generalize and are mapped to `[UNK]`.

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `[PAD]` | 0 | Padding for batch alignment |
| `[UNK]` | 1 | Out-of-vocabulary fallback |

No `[CLS]` token is needed because pooling is done over all token positions (no dedicated aggregation token). No
`[SEP]` or `[MASK]` tokens — single card input, no masking tasks.

## Tokenization Process

1. Strip the `name:` line value (replace with `CARDNAME` placeholder)
2. Replace multi-word keywords with their underscore forms (longest match first)
3. Lowercase all text (mana symbols like `{W}` are preserved as-is after vocabulary lookup)
4. Split on whitespace
5. Map each word to its vocabulary ID, or `[UNK]` (ID 1) if absent

## Why Not BPE or WordPiece

MTG uses a constrained language — a relatively small set of game-defined terms covers nearly all of the text. BPE and
WordPiece are designed for open-vocabulary natural language where rare words need subword decomposition. In the MTG
domain, the words that matter are game terms that should always be single tokens, and the words that don't matter
(proper nouns) can safely be `[UNK]`. A word-level tokenizer is simpler, produces more interpretable token sequences,
and shrinks the embedding table from ~30,000 entries (BERT) to ~5,000.

# Sklearn Model

## Architecture

A `GradientBoostingRegressor` trained on a dense feature vector assembled from 7 feature groups plus a TF-IDF oracle
text matrix.

### Feature Groups

| Group | Width | Description |
|-------|-------|-------------|
| Mana cost | 13 | has_mana_cost, CMC, WUBRG pip counts, color count, generic, colorless, has_X, has_hybrid, has_phyrexian |
| Card types | 15 | One-hot: Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle, Scheme, Plane, Conspiracy, Vanguard, Phenomenon, Tribal, Kindred |
| Supertypes | 7 | One-hot: Legendary, Basic, Snow, World, Ongoing, Host + subtype count |
| Keywords | 32 | Top-30 keywords (one-hot) + total keyword count + oracle text length |
| P/T & combat | 6 | Power, toughness, is_variable_power, is_variable_toughness, loyalty, ability count |
| Layout | 6 | One-hot: normal, doublefaced, split, adventure, modal, flip |
| Printing | 19 | is_reserved, is_abu, rarity one-hot (4), printings count, normalized release year, legality count, per-format legality multi-hot (10) |
| Oracle TF-IDF | 500 | TF-IDF vectorizer over oracle text (max 500 features, English stop words removed) |

Total: ~598 features (exact count depends on vocabulary).

The top-30 keywords are learned from the training set. The TF-IDF vocabulary is also learned at training time.

## Training

The target is `log(price)` — the natural logarithm of the EUR price. The skew of the price distribution (many
cheap cards, few expensive ones) would otherwise let a handful of expensive cards dominate the loss. The model
predicts in log space; predictions are transformed back via `exp()` for display.

Training uses the full 80% training split with default `GradientBoostingRegressor` hyperparameters and
`random_state=42`. The trained model, including the fitted `FeatureEngineering` pipeline (TF-IDF vocabulary, top-30
keywords), is serialized as a `.joblib` file to `models/price-predictor/sklearn/`.

Two artifacts are saved: a timestamped version (`{timestamp}.joblib`) and a `latest.joblib` symlink for convenience.

# Transformer Model

## Architecture

An encoder-only transformer that predicts shifted-log card prices from tokenized card text. The architecture:

```
Token IDs → Embedding(vocab_size, d_model) + Positional(max_seq_len, d_model) → Dropout
  → TransformerEncoder(n_layers layers, n_heads heads, ff_dim feedforward)
  → cat([max_pool, mean_pool])   →   (2 × d_model,)
  → cat([pooled_text, metadata]) →   (2 × d_model + 15,)
  → Linear(2 × d_model + 15, regression_hidden_dim) → ReLU → Linear(regression_hidden_dim, 1)
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Token embedding dimension |
| `n_layers` | 2 | Transformer encoder layers |
| `n_heads` | 4 | Attention heads per layer |
| `ff_dim` | 1024 | Feedforward inner dimension (4× d_model) |
| `max_seq_len` | (from vocab) | Maximum token sequence length |
| `dropout` | 0.1 | Dropout rate |
| `regression_hidden_dim` | 64 | Hidden dimension of the output MLP |
| `log_offset` | 2.0 | Price transform: `log(price + offset)` |
| `meta_dim` | 15 | Metadata side-channel width |

All architecture hyperparameters are CLI flags, enabling experimentation without code changes.

### Pooling

The encoder's output (one vector per token position) is reduced to a fixed-size card representation via
**dual pooling**: max-pool and mean-pool across the token dimension, concatenated.

- **Max-pool** captures peak activations — the presence of rare keywords, specific mana symbols, or unusual ability
  patterns. If any token triggers a feature strongly, max-pool preserves it.
- **Mean-pool** captures overall texture — the average characteristics across all tokens. A card full of creature
  keywords has a different mean signature than one full of spell effects.

Both operations mask padding positions: max-pool fills padding with `-inf` (so padding never wins), mean-pool fills
padding with `0.0` and divides by the true sequence length. The result is a vector of shape `(2 × d_model,)`.

### Metadata Side-Channel

After pooling, a 15-dimensional metadata vector is concatenated before the regression head:

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | is_reserved | 0.0 / 1.0 |
| 1 | rarity | ordinal: common=0.0, uncommon=0.33, rare=0.67, mythic=1.0 |
| 2 | printings count | log(count) / log(50), clamped [0, 1] |
| 3 | release year | (year - 1992) / 34, clamped [0, 1] |
| 4–13 | format legalities | multi-hot over 10 formats |
| 14 | is_abu | 0.0 / 1.0 |

The metadata enters after the transformer encoder, not as input tokens. This is deliberate: numeric and categorical
features like "printings count = 12" or "rarity = mythic" would be poorly represented as text tokens, and mixing them
into the token stream would dilute the text signal. The side-channel lets the regression head learn the interaction
between text features and metadata without burdening the encoder.

## Training

**Loss function:** Huber loss (delta=1.0) on shifted-log prices. The shifted-log transform compresses the price
distribution: `target = log(price + log_offset)`, `prediction = exp(output) - log_offset`. The default log_offset
of 2.0 — roughly the "bulk threshold" in EUR — compresses price differences below ~€2, making the model focus its
gradient budget on cards worth distinguishing by price. Huber loss further reduces sensitivity to outlier prices
compared to MSE.

**Optimizer:** AdamW with learning rate 1e-4.

**Batch size:** 64 (fits comfortably in 8GB VRAM on a GeForce RTX 3060 Ti).

**Early stopping:** patience of 20 epochs. The best checkpoint is selected by validation accuracy (fraction of test-set
cards where the predicted price bucket matches the actual price bucket), not by loss — this favors models that get the
rank order right over models that minimize average error.

**Price-bucket oversampling:** Training batches are resampled so that expensive cards (which are rare in the dataset)
appear more frequently. The `--sampler-exponent` flag controls the strength: 0 = uniform sampling, 0.5 = square root
of inverse frequency, 1.0 = full inverse frequency weighting. This prevents the model from learning to predict
"everything is €0.10" just because most cards are bulk.

## Card Embeddings for Downstream Use

The `encode()` method returns `cat([max_pool, mean_pool])` — a `(2 × d_model,)` vector (512 dimensions at default
settings). This is the card embedding used by the sealed deck picker as its frozen card encoder.

Before encoding, the `name:` line is stripped from the card text so that the embedding captures what the card does,
not its identity. Two cards with identical abilities but different names produce identical embeddings.

# CLI and REST API

Entry point: `python -m price_predictor <subcommand>`

## Commands

| Command | Description |
|---------|-------------|
| `convert` | Launch Java batch converter to transform Forge scripts → converted text format |
| `check-convert` | Compare converted files against Forge Oracle text, flag low-similarity cards |
| `vocabulary` | Build custom tokenizer vocabulary from the converted card corpus |
| `train sklearn` | Train the sklearn GradientBoostingRegressor |
| `train transformer` | Train the transformer model (GPU required) |
| `predict sklearn` | Predict price from `--file` or inline `--card` text using sklearn |
| `predict transformer` | Predict price using the transformer model |
| `evaluate sklearn` | Compute accuracy metrics on the held-out test split |
| `evaluate transformer` | Compute accuracy metrics with per-price-bucket breakdown |
| `serve` | Start the FastAPI prediction service |

## REST API

`POST /api/v1/predict` accepts `text/plain` converted card text and returns JSON with predictions from every loaded
model:

```json
{
  "sklearn": {
    "predicted_price_eur": 3.45,
    "model_version": "latest"
  },
  "transformer": {
    "predicted_price_eur": 4.12,
    "model_version": "transformer-v1"
  }
}
```

The service loads the sklearn model (required — fails fast if absent), the transformer model (optional — graceful
degradation to `null` if absent), the custom tokenizer, and the MTGJSON metadata map for auto-filling printing data
on known cards.

When a request arrives, the service:

1. Extracts the card name from the `name:` line
2. Looks up `PrintingData` from the metadata map (if the card is known)
3. Parses the text into a `Card` entity for sklearn
4. Runs sklearn prediction (always)
5. Runs transformer prediction (if loaded)
6. Returns both results

Each request is logged as structured JSON with timestamp, status code, latency, card attributes, and predicted prices.

## Java Connector (forge-connector)

The `forge-connector` Maven module serves two roles:

1. **Client library for Forge**: `PricePredictorClient` wraps HTTP calls to the prediction API. Zero external
   dependencies — Forge can call `new PricePredictorClient("http://host:8000").predict(cardAttributes)` to get a price
   estimate during deck building.
2. **CLI workers invoked by Python**: The fat JAR (`mvn package -DskipTests`) provides `ConvertMain` for batch card
   conversion, invoked by `python -m price_predictor convert`.

# Evaluation

Both models are evaluated on the same held-out 20% test split. Five metrics are reported:

| Metric | What it measures |
|--------|-----------------|
| Mean Absolute Error (EUR) | Average distance between predicted and actual prices in euros |
| Median Percentage Error | Typical relative error — less sensitive to outliers than mean |
| Median Absolute Error (log) | Error in log-price space — treats a €1→€2 error the same as €10→€20 |
| Top-20% Overlap | Precision of expensive-card detection: what fraction of the predicted top 20% actually are in the true top 20% |
| Sample Count | Number of test examples evaluated |

The transformer evaluation additionally reports a **per-price-bucket breakdown** — accuracy and error within price
ranges (e.g., €0-€1, €1-€5, €5-€20, €20+). This reveals whether the model is good across the price spectrum or
only accurate for bulk cards.

Top-20% overlap is the most practically useful metric: if the model correctly identifies which cards are expensive,
it is doing something meaningful even if the exact EUR predictions are noisy.

# Future Directions

- The transformer's card embeddings are already consumed by the sealed deck picker as a frozen feature extractor.
  Fine-tuning the embeddings for sealed play quality (Phase B of the scorer training) could improve both systems.
- The unified `{model}` subcommand pattern (`train {model}`, `predict {model}`, `evaluate {model}`) is designed for
  extensibility — additional model architectures can be added without restructuring the CLI.
- The custom tokenizer's vocabulary could be extended with set-specific terms if per-set price prediction is pursued
  in the future.
