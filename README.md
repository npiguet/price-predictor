REA# Price Predictor

Predicts Magic: The Gathering card EUR market prices from game-visible
attributes (mana cost, card types, oracle text, power/toughness, keywords).
Designed for both real and made-up cards — the primary use case is estimating
what a hypothetical card would cost based on its characteristics.

## Prerequisites

- Python 3.14+
- pip
- MTG Forge repository checkout at `../forge` (sibling directory)
- MTGJSON data files in `resources/` (one-time download)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install the package with dev dependencies (includes CUDA-enabled PyTorch)
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126

# Download MTGJSON data files (one-time)
# Place these in the resources/ directory:
#   - AllPricesToday.json  from https://mtgjson.com/api/v5/AllPricesToday.json
#   - AllPrintings.json    from https://mtgjson.com/api/v5/AllPrintings.json

# Verify Forge checkout exists
ls ../forge/forge-gui/res/cardsfolder/
# Should show alphabetical subdirectories (a/ through z/)
```

## Workflows

All commands are run as `python -m price_predictor <subcommand>`.

### Build domain vocabulary (required before training the transformer)

Scans the converted card corpus and builds a compact, MTG-specific word-level
vocabulary stored as `models/transformer/vocab.txt`. This step is required
before training or running the transformer model.

**Inputs**: Converted card text files in `./output/` (from `convert`).
**Output**: `models/transformer/vocab.txt` (one token per line, ~5,000 tokens).

```bash
python -m price_predictor vocabulary
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `models/transformer/` | Directory where `vocab.txt` is written |
| `--cards-path` | `./output` | Path to converted card corpus |
| `--freq-threshold` | `5` | Min corpus occurrences for a word to be included |

**Output** (JSON to stdout):
```json
{
  "vocab_path": "models/transformer/vocab.txt",
  "vocab_size": 5064,
  "domain_token_count": 41,
  "freq_threshold_token_count": 4961,
  "coverage_pct": 98.4,
  "unk_pct": 1.6
}
```

### Train a model

Reads converted card text files, joins them with Cardmarket EUR prices from
MTGJSON data, and trains a prediction model. Two model types are available:
`sklearn` (Gradient Boosted Trees) and `transformer` (custom word-level tokenizer).

**Inputs**: Converted card text files in `./output/` (from `convert`),
`AllPrintings.json` (name-to-UUID mapping), `AllPricesToday.json` (EUR prices).

```bash
# Train the sklearn model
python -m price_predictor train sklearn

# Train the transformer model (requires vocabulary to be built first)
python -m price_predictor vocabulary           # step 1: build vocab
python -m price_predictor train transformer --epochs 20 --batch-size 64
```

Models are saved to `models/sklearn/` and `models/transformer/` respectively,
with timestamped filenames and a `latest` copy.

Options: `--output-dir`, `--prices-path`, `--printings-path`, `--model-output`,
`--test-split`, `--random-seed`. Transformer adds `--batch-size`, `--epochs`,
`--lr`, `--patience`, `--vocab-path`. See `python -m price_predictor train sklearn --help`.

### Predict a card price

Takes a card in converted text format (from a file or inline) and returns an
EUR price estimate. Runs locally — no REST service needed.

```bash
# From a file
python -m price_predictor predict sklearn --file output/l/lightning_bolt.txt

# From inline text
python -m price_predictor predict transformer --card "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target."
```

**Output**: JSON to stdout with `predicted_price_eur` and `model_version`.

### Evaluate model accuracy

Computes accuracy metrics on a held-out test set.

```bash
python -m price_predictor evaluate sklearn
python -m price_predictor evaluate transformer
```

Options: `--model-path`, `--output-dir`, `--prices-path`, `--printings-path`,
`--test-split`, `--random-seed`, `--output-csv` (sklearn only).

### Serve the prediction API

Starts a REST service exposing both models over HTTP. Callers send a card in
converted text format and receive JSON predictions from all available models.

```bash
python -m price_predictor serve
```

Options: `--model-path` (default: `models/sklearn/latest.joblib`),
`--printings-path` (default: `resources/AllPrintings.json`),
`--prices-path` (default: `resources/AllPricesToday.json`),
`--vocab-path` (default: `models/transformer/vocab.txt`),
`--host` (default: `0.0.0.0`), `--port` (default: `8000`).

The server loads AllPrintings and AllPricesToday at startup to build a metadata
lookup. For known cards, it auto-fills printing data fields (reserved, rarity,
printings count, set code, legalities) from the cheapest printing. Unknown cards
receive sensible defaults. Client-provided fields in the card text override
auto-filled values.

Test with curl:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: text/plain" \
  -d "name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target."
```

Response:
```json
{
  "sklearn": {"predicted_price_eur": 0.16, "model_version": "latest"},
  "transformer": {"predicted_price_eur": 0.22, "model_version": "transformer"}
}
```

### Structured request logging

Every request to `POST /api/v1/predict` is logged to stderr as a single-line
JSON object with: event type, timestamp, status code, latency, card attributes,
and prediction results from each model.

### Batch convert Forge card scripts

Converts the entire Forge card script library (~32,000 `.txt` files) into
LLM-friendly text format. Each output file contains lowercase property lines
(name, mana cost, types, etc.) followed by classified ability lines with action
counters.

**Prerequisites**: Java 17+, Forge built (`cd ../forge && mvn install -DskipTests`),
forge-connector fat JAR built (`cd forge-connector && mvn package -DskipTests`).

```bash
python -m price_predictor convert \
  --cards-path ../forge/forge-gui/res/cardsfolder \
  --output-path ./output
```

Options: `--cards-path` (default: `../forge/forge-gui/res/cardsfolder/`),
`--output-path` (default: `./output`).

The output mirrors the input directory structure. Example converted card:
```
name: lightning bolt
mana cost: R
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
```

#### Printing data fields

During training and prediction, 5 printing data fields are appended to the
card text. These are derived from AllPrintings.json for known cards:

```
reserved: false
rarity: uncommon
printings: 23
set: 2xm
legalities: commander, legacy, modern, pauper, vintage, penny, oathbreaker
```

| Field | Source | Default (unknown cards) |
|-------|--------|------------------------|
| `reserved` | `isReserved` from AllPrintings (absent = false) | `false` |
| `rarity` | Rarity of the cheapest printing | `rare` |
| `printings` | Count of sets the card has been printed in | `1` |
| `set` | Set code of the cheapest printing (lowercase) | `ukn` |
| `legalities` | Formats where the card is "Legal" (10 recognized formats) | all 10 formats |

The 10 recognized formats: Standard, Pioneer, Modern, Brawl, Legacy, Vintage,
Pauper, Commander, Penny, Oathbreaker.

Multi-face cards include a `layout:` line and separate faces with `ALTERNATE`:
```
layout: transform
name: delver of secrets
mana cost: U
types: creature human wizard
power toughness: 1/1
keyword[1]: transform

ALTERNATE

name: insectile aberration
types: creature human insect
power toughness: 3/2
keyword: flying
```

You can also run the Java converter directly:
```bash
java -cp "forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar;../forge/forge-game/target/forge-game-2.0.10-SNAPSHOT.jar;../forge/forge-core/target/forge-core-2.0.10-SNAPSHOT.jar;../forge/forge-game/target/dependency/*" \
  com.pricepredictor.connector.ConvertMain \
  --cards-path ../forge/forge-gui/res/cardsfolder \
  --output-path ./output
```

### Java Connector (forge-connector)

A zero-dependency Java 17+ library that lets MTG Forge (or any Java application)
get price predictions from the running service with 5 lines of code. See
[`forge-connector/README.md`](forge-connector/README.md) for full documentation.

```java
var client = new PricePredictorClient();
var estimate = client.predict(CardAttributes.builder()
    .type("Creature").manaCost("1 G G")
    .power("2").toughness("2").build());
System.out.println(estimate.predictedPriceEur());
```

## ML Approach

### Why Gradient Boosted Trees

Card price prediction is a **tabular regression problem**. The input is a mix
of categorical features (card types, colors), numeric features (mana value,
power/toughness), and text features (oracle text). For this class of problem,
gradient boosted trees are the standard first choice.

The model uses scikit-learn's `GradientBoostingRegressor`.

**Alternatives considered and rejected**:
- **Random Forest**: Simpler and a good baseline, but slightly less accurate
  than gradient boosting on tabular data.
- **Linear regression**: Too limited for the nonlinear relationships between
  card attributes and prices (e.g., certain keyword combinations command
  price premiums that are not additive).
- **Deep learning** (PyTorch/TensorFlow): Overkill for tabular data with ~17
  feature groups. Adds significant complexity for marginal gains on small
  feature sets. Could be explored in the future for oracle text embeddings.
- **XGBoost/LightGBM**: More powerful gradient boosting implementations, but
  add external C++ dependencies. Can upgrade later if the scikit-learn model
  hits accuracy ceilings.

### Log-transformed target

Card prices follow a heavily right-skewed distribution (thousands of cards at
a few cents, a handful above 100 EUR). The model trains on **log-transformed
EUR prices** and exp-transforms predictions back to EUR. This prevents
expensive cards from dominating the loss function and improves predictions
across all price ranges.

### Feature Engineering

The feature pipeline transforms card attributes into a numeric vector with 17
feature groups:

| # | Source | Feature | Type |
|---|--------|---------|------|
| 1 | Mana cost | Total mana value (CMC) | numeric |
| 2 | Mana cost | Color indicators (W, U, B, R, G) | 5 binary columns |
| 3 | Mana cost | Color count | numeric |
| 4 | Mana cost | Generic mana component | numeric |
| 4a | Mana cost | Colorless mana pips ({C}) | numeric |
| 5 | Types | Card types (Creature, Instant, etc.) | multi-hot |
| 6 | Types | Supertypes (Legendary, Basic, etc.) | multi-hot |
| 7 | Types | Subtype count | numeric |
| 8 | Keywords | Top-30 keywords | multi-hot |
| 9 | Keywords | Total keyword count | numeric |
| 10 | Oracle text | TF-IDF (top 500 terms) | sparse vector |
| 11 | Oracle text | Text length | numeric |
| 12 | Power | Numeric value (* as NaN + indicator) | numeric |
| 13 | Toughness | Numeric value (* as NaN + indicator) | numeric |
| 14 | Loyalty | Planeswalker starting loyalty | numeric |
| 15 | Abilities | Count of defined abilities | numeric |
| 16 | Layout | Card layout (normal, split, etc.) | one-hot |
| 17 | Printing data | Reserve list status | binary |
| 18 | Printing data | Rarity (common/uncommon/rare/mythic) | one-hot (4) |
| 19 | Printing data | Printings count | numeric |
| 20 | Printing data | Legalities count | numeric |
| 21 | Printing data | Per-format legality (10 formats) | multi-hot (10) |

Colorless mana ({C}) is tracked separately from generic mana ({1}--{N}).
Printing data fields (features 17--21) are derived from AllPrintings.json at
training time. Set code is excluded from sklearn features (too high-cardinality)
but is present in the tokenized text for the transformer model.

## Artifacts

### Trained model files

Model artifacts are organized by model type:

- **sklearn**: `models/sklearn/{timestamp}.joblib` + `models/sklearn/latest.joblib`
  Contains the GradientBoostingRegressor and fitted FeatureEngineering pipeline.
- **transformer**: `models/transformer/{timestamp}.pt` + `models/transformer/latest.pt`
  Contains the transformer state_dict and architecture config.

Both use timestamped filenames (e.g., `20260315-143000`) with a `latest` copy
pointing to the most recent version.

### Training output (JSON, stdout)

```json
{
  "model_version": "20260301-143000",
  "model_path": "models/20260301-143000.joblib",
  "cards_used": 24531,
  "cards_skipped": 7569,
  "skipped_reasons": {
    "no_price": 6200,
    "parse_error": 169,
    "no_printings_match": 1200
  },
  "price_range_eur": {"min": 0.02, "max": 425.00}
}
```

### Prediction output (JSON, stdout)

```json
{
  "predicted_price_eur": 12.45,
  "model_version": "20260301-143000"
}
```

### Evaluation output (JSON, stdout)

```json
{
  "model_version": "20260301-143000",
  "mean_absolute_error_eur": 2.87,
  "median_percentage_error": 38.5,
  "top_20_overlap": 0.64,
  "sample_count": 6132
}
```

### Per-card evaluation CSV (optional)

Generated with `--output-csv`. Contains one row per test card with columns:
card name, actual price (EUR), predicted price (EUR), absolute error.

## Running Tests

```bash
# Python: Fast unit tests (default)
pytest

# Python: Integration tests only
pytest tests/integration/

# Python: All tests
pytest tests/

# Python: Linting
ruff check src/ tests/

# Java: Connector tests
cd forge-connector && mvn test
```

## Project Structure

```
src/price_predictor/
  domain/           Pure game entities and value objects (no dependencies)
    entities.py     Card, PriceEstimate, TrainingExample, TrainedModel, EvaluationMetrics
    value_objects.py ManaCost (parsed Forge mana format)
  application/      Use cases (depends on domain only)
    train.py        TrainModelUseCase (sklearn)
    train_transformer.py  Transformer training pipeline
    predict.py      PredictPriceUseCase (sklearn)
    predict_transformer.py  PredictTransformerUseCase
    evaluate.py     EvaluateModelUseCase (sklearn)
    evaluate_transformer.py  Transformer evaluation
    card_enrichment.py  Printing data enrichment (auto-fill, defaults, merge)
    feature_engineering.py  Card -> numeric feature vector
  infrastructure/   External integrations (depends on application)
    cli.py          argparse CLI (train/predict/evaluate {sklearn,transformer}, serve, convert)
    server.py       FastAPI app, POST /api/v1/predict endpoint
    forge_parser.py Forge card script parser (used by convert)
    converted_card_parser.py  Converted text → Card parser (used by train/predict/evaluate)
    mtgjson_loader.py AllPrintings/AllPricesToday loaders
    model_store.py  sklearn model save/load (joblib)
    transformer_store.py  Transformer model save/load (.pt)
    transformer_model.py  CardPriceTransformerModel architecture
    transformer_dataset.py  PyTorch Dataset for transformer training
forge-connector/    Java Maven module for Forge integration
  src/main/java/    PricePredictorClient, CardScriptConverter, BatchConverter, ConvertMain
  src/test/java/    JUnit 5 tests (unit + @Tag("integration") for Forge-dependent tests)
tests/
  unit/             Fast unit tests (fixture-based)
  integration/      End-to-end pipeline + server integration tests
  fixtures/         Sample card scripts and JSON data
models/             Trained model artifacts (.gitignored)
resources/          Frozen MTGJSON data files
```

---

## `python -m sealed` — Sealed Dataset Preparation

The `sealed` module provides two independent CLI commands for building training data for sealed-deck ML models.

### Workflow

```
1. encode-cards   → produce one .npz embedding per card script
2. generate-pools → produce sealed pool lists for a given set
```

These commands are independent and can run in any order. `encode-cards` must complete before pool embeddings can be assembled downstream.

### encode-cards

Scans a directory of converted card scripts, strips the `name:` line from each card, and produces a `.npz` embedding file alongside every `.txt`. Already-encoded cards are skipped — re-running is safe and fast.

```
python -m sealed encode-cards \
    [--encoder-path models/price-predictor/transformer/latest.pt] \
    [--vocab-path   models/price-predictor/transformer/vocab.txt] \
    [--cards-path   output/cardsfolder/]
```

| Argument | Default | Description |
|---|---|---|
| `--encoder-path` | `models/price-predictor/transformer/latest.pt` | Pretrained transformer model |
| `--vocab-path` | `models/price-predictor/transformer/vocab.txt` | Tokenizer vocabulary |
| `--cards-path` | `output/cardsfolder/` | Card script directory (recursive) |

**Output**: One `.npz` per `.txt` card script in the same directory as the source. Each file stores a single array under the key `"embedding"`.

**`.npz` format**: `np.load(path)["embedding"]` → `float32` array of shape `(2 * d_model,)` (e.g. `(512,)` for a 256-dim model). The embedding is `cat([max_pool, mean_pool])` over transformer encoder outputs with the `name:` line stripped.

**Exit codes**: `0` = success, `1` = one or more cards failed (processing continues), `2` = fatal error (missing model/vocab).

### generate-pools

Invokes the forge-connector JAR to generate sealed pools (6 boosters each) for a given MTG set code. Writes `pools.txt` to the output directory, overwriting any existing file.

```
python -m sealed generate-pools \
    [--set        RVR] \
    [--size       10000] \
    [--pools-path output/sealed/pools/{set}/]
```

| Argument | Default | Description |
|---|---|---|
| `--set` | `RVR` | MTG set code (e.g. `RVR`, `MH3`, `BLB`) |
| `--size` | `10000` | Number of pools to generate |
| `--pools-path` | `output/sealed/pools/{set}/` | Output directory; `{set}` is replaced with the set code |

**`pools.txt` format**: One pool per line, card names separated by semicolons. Basic lands are excluded. Duplicate names within a line are valid.

```
Ponder;Lightning Bolt;Counterspell;Dark Ritual;...
Giant Growth;Serra Angel;Wrath of God;...
```

**Exit codes**: `0` = success, `2` = fatal error (invalid set, JAR not found, Java not on PATH).

### match-outcomes

Generates a large dataset of sealed-format match outcomes for training a deck scorer. Spawns a configurable number of Java worker processes that each independently pick a random sealed-legal set, generate two 6-booster pools, construct decks, play a best-of-N match via Forge AI, and append the result to a shared flat file. The supervisor monitors workers and restarts any that crash.

The deck source for each side is controlled by the optional `--side-a-decks` / `--side-b-decks` flags. When both are absent, the run is Phase 0 (the 4 Forge methods on both sides). When one or both point at a generated-decks file (output of `build-decks`), decks for that side are sampled from the file. See the table below.

**Prerequisites**: Java 17+, Forge built (`cd ../forge && mvn install -DskipTests`), forge-connector fat JAR built (`cd forge-connector && mvn package -DskipTests`).

```bash
# Phase 0: 12 workers, both decks built by the 4 Forge methods on both sides
python -m sealed match-outcomes

# Use fewer workers (e.g. on a machine with limited RAM)
python -m sealed match-outcomes --workers 4

# Self-play: side A is always sampled from a scorer-built deck file; side B
# is the 4 Forge methods (weights 4:3:2:1).
python -m sealed match-outcomes \
    --side-a-decks output/sealed/generated-decks-gen2.txt

# Cross-generation: side A is gen2-only; side B is the 4 Forge methods PLUS
# decks sampled from a concat of all prior generations (gen1+gen2). Boost
# the file-sampled side-B weight to 8 to oversample scorer-built opponents.
python -m sealed match-outcomes \
    --side-a-decks output/sealed/generated-decks-gen2.txt \
    --side-b-decks output/sealed/generated-decks-prior.txt \
    --side-b-decks-weight 8
```

| Argument | Default | Description |
|---|---|---|
| `--workers` | `12` | Number of parallel Java worker processes |
| `--side-a-decks` | _(none)_ | Optional path to a generated-decks file. When given, deck A is sampled from this file every match and the 4 Forge methods are not used for side A. The deck's `LABEL` (set by `build-decks --label`) becomes the `method_A` value. |
| `--side-b-decks` | _(none)_ | Optional path to a generated-decks file. When given, deck B is rolled between the 4 Forge methods (weights 4:3:2:1) and sampling from this file (weight `--side-b-decks-weight`). When omitted, side B uses the 4 Forge methods only. |
| `--side-b-decks-weight` | `4` | Weight of the file-sampled side-B method relative to the 4 Forge methods (which carry total weight 10). Only meaningful with `--side-b-decks`; supplying it without `--side-b-decks` is an error. |
| `--best-of` | `7` | Games per match; positive odd integer. |

Same-set constraint: deck A's set code drives everything. Forge-built side-B decks are pool-generated in deck A's set; file-sampled side-B decks are filtered to deck A's set code. If no non-mirror file deck exists for that set, the file roll falls back to Forge methods. Mirror matches are excluded by content equality (sorted card-name multiset), so a deck with identical content on both sides is never played.

**Stop**: Press **Ctrl+C** to terminate all workers and exit cleanly.

**Output**: `output/sealed/match-outcomes.txt` — one match per line, appended indefinitely. The file is never truncated — runs accumulate.

**Output format** (`match-outcomes.txt`):
```
card1|card2|...|card40;card1|card2|...|card40;wins_a;wins_b
```
Four semicolon-separated fields:
1. `deck_a`: Pipe-separated card names (40 cards, Forge canonical names, duplicates repeat)
2. `deck_b`: Same format as `deck_a`
3. `wins_a`: Games won by deck A (0–2)
4. `wins_b`: Games won by deck B (0–2)

**Invariants**: `wins_a + wins_b` is 2 or 3; each deck contains exactly 40 cards.

**Architecture**:
```
python -m sealed match-outcomes
        │
        ├── Worker 0 (java MatchWorkerMain) ──┐
        ├── Worker 1 (java MatchWorkerMain) ──┤
        ├── ...                               ├──► output/sealed/match-outcomes.txt
        └── Worker N (java MatchWorkerMain) ──┘
```

Each Java worker uses one of four weighted deck-building methods (40/30/20/10%) to produce decks of varying quality, from Forge's optimal sealed builder to a fully random 23-card selection. This variation produces a richer training signal for the deck scorer.

**Exit codes**: `0` = clean shutdown, `2` = configuration error (Java not found, JAR not built).

### train-scorer

Trains the Set Transformer deck scorer on a `match-outcomes-*.txt` file. Each match in the file becomes a pairwise (winner, loser) training example optimized with Bradley-Terry binary cross-entropy on the score difference. By default every match contributes equally regardless of how decisive it was; `--margin-weighting linear|log` scales each match's contribution by its absolute game-win margin so 4-0 sweeps carry more gradient signal than 4-3 squeakers. The trainer holds out a fraction of examples for validation, runs validation every epoch, and saves the best-by-val-accuracy checkpoint to `models/sealed/scorer/`.

**Prerequisites**: card embeddings have been written by `encode-cards` (one `.npz` per card under `output/cardsfolder/`); a match-outcomes file exists at the path passed to `--outcomes-path`.

```bash
# Default training run (reads output/sealed/match-outcomes.txt)
python -m sealed train-scorer

# Train against the full archived corpus with mild dropout
python -m sealed train-scorer \
    --outcomes-path output/sealed/match-outcomes-all.txt \
    --n-layers 4 --dropout 0.1 --lr 1e-5

# Resume from a checkpoint and unfreeze embeddings (Phase B fine-tuning)
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_l4_h4_s4_ff1088_mlp256_lr1e-05.pt \
    --unfreeze-embeddings --embedding-lr 1e-6
```

| Argument | Default | Description |
|---|---|---|
| `--outcomes-path` | `output/sealed/match-outcomes.txt` | Path to the match-outcomes file to train on |
| `--checkpoint-dir` | `models/sealed/scorer/` | Directory where checkpoints are written |
| `--resume` | _(none)_ | Path to a checkpoint to resume training from |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `1e-3` | Learning rate for the scorer parameters |
| `--n-layers` | `6` | Number of Set Transformer SAB layers |
| `--n-heads` | `4` | Number of attention heads |
| `--n-seeds` | `4` | Number of PMA seed vectors |
| `--d-ff` | `1088` | Feed-forward dimension in SAB layers |
| `--mlp-hidden` | `256` | Hidden dimension of the scoring MLP head |
| `--dropout` | `0.2` | Dropout rate applied in SAB attention/FF, PMA attention, and the scoring MLP |
| `--margin-weighting` | _(unset)_ | If `linear` or `log`, weight each match's pairwise loss contribution by the absolute game-win margin (Bo7: 1–4). `linear` uses the margin directly; `log` uses `log(1 + margin)` as a dampened fallback. Unset = unweighted (current default; every match contributes equally). Weighted runs gain an `_mwlin` / `_mwlog` suffix in the best/latest checkpoint name so they don't clobber unweighted checkpoints at the same architecture. |
| `--val-interval` | `1` | Run validation every N epochs |
| `--unfreeze-embeddings` | `False` | Enable embedding fine-tuning (Phase B) |
| `--embedding-lr` | `1e-5` | Learning rate for embedding fine-tuning |
| `--val-fraction` | `0.2` | Fraction of examples held out for validation |
| `--random-seed` | `42` | RNG seed for the train/val split |

**Stop**: Press **Ctrl+C** to stop training cleanly. The best-by-val-accuracy checkpoint up to that point is preserved.

**Output**: `models/sealed/scorer/best_l<N>_h<H>_s<S>_ff<FF>_mlp<MH>_lr<LR>.pt` — the best checkpoint for that hyperparameter combination, updated whenever a new validation-accuracy high is reached.

### build-decks

Builds one 40-card deck per pool using a trained scorer plus greedy or simulated-annealing search. Writes a `generated-decks.txt` file consumed by `match-outcomes --side-a-decks` / `--side-b-decks` for self-play.

**Prerequisites**: card embeddings have been written by `encode-cards`; a trained scorer checkpoint exists; a pools file exists at the path passed to `--pools-path`.

```bash
# Build decks at default search settings (T=0.8, cooling=0.85, restarts=1)
python -m sealed build-decks \
    --pools-path output/sealed/pools/RVR/pools.txt \
    --label gen-3

# Pure greedy with an explicit checkpoint and output path
python -m sealed build-decks \
    --pools-path output/sealed/pools/MH3/pools.txt \
    --label gen-3-greedy \
    --checkpoint models/sealed/scorer/best_l6_full_training.pt \
    --sa-temperature 0 \
    --output output/sealed/generated-decks-greedy.txt

# Color-pair seeded init: one search per MTG two-color pair
python -m sealed build-decks \
    --pools-path output/sealed/pools/RVR/pools.txt \
    --label gen-3-cp \
    --restarts color-pairs
```

| Argument | Default | Description |
|---|---|---|
| `--pools-path` | _(required)_ | Input pools file (`SET_CODE;Card1\|...` format) |
| `--label` | _(required)_ | Generation-method tag written as the first column of every output line (e.g. `gen-3`). Recorded by `match-outcomes` self-play as the `method_A` / `method_B` value when this deck is sampled into a match. Must be a non-empty string without `;`, `\|`, or whitespace. |
| `--checkpoint` | `models/sealed/scorer/latest.pt` | Trained scorer checkpoint |
| `--cards-path` | `output/cardsfolder/` | Card-embedding directory (one `.npz` per card) |
| `--output` | `output/sealed/generated-decks.txt` | Output path for generated decks |
| `--sa-temperature` | `0.8` | Initial temperature for simulated annealing; `0` is pure greedy |
| `--sa-cooling` | `0.85` | Per-iteration temperature multiplier (ignored when `--sa-temperature 0`) |
| `--sa-max-iterations` | `200` | Hard cap on iterations per restart |
| `--restarts` | `1` | Either a positive integer N (run N searches from random 23-spell inits and keep the best deck) or the literal `color-pairs` (run one search per MTG two-color pair — WU, WB, WR, WG, UB, UR, UG, BR, BG, RG — with each search seeded by an on-color initial 23-spell deck; the color filter applies only to the seed deck, the search itself is unconstrained). |
| `--print-decks` | `False` | After building, dump every deck to stdout in the human-readable format used by `evaluate-scorer` (sorted by mana value, lands at bottom). Labels are not included in the printed output. |
| `--resume` | `False` | Resume from a partial run: count complete lines already in `--output`, skip that many pools from the front of `--pools-path`, and append remaining decks to the existing file. Without this flag the output file is truncated at the start of the run. |

**Output**: `output/sealed/generated-decks.txt` (or `--output`) — one line per pool that produced a viable deck, in `LABEL;SET_CODE;Card1|Card2|...|Card40` format. Pools with fewer than 23 embeddable cards are skipped silently. Concatenating multiple generated-decks files with different `--label` values is supported: `match-outcomes` reads the per-deck label out of the first column.

**Resuming an interrupted run**: `--resume` is the safe way to recover from a crash mid-run. The output file is line-buffered (every newline flushes to disk), so an interrupted run leaves a clean per-deck checkpoint; pass `--resume` on the next invocation to continue from where it stopped. Caveat: the line-count heuristic assumes every input pool produced a deck on the prior run. If a pool was skipped (fewer than 23 embeddable cards — vanishingly rare for real 6-booster pools), resume will be off by however many pools were skipped.

See [`experiments/2026-04-25-sa-deck-builder-tuning.md`](experiments/2026-04-25-sa-deck-builder-tuning.md) for empirical guidance on `--sa-temperature`, `--sa-cooling`, and the restart strategy.

### train-picker

Trains a one-shot deck **picker** — a policy transformer over a sealed pool that emits a full 23-spell deck in a single forward pass — from random initialization via REINFORCE against a frozen scorer. The picker is the inference-cheap replacement for `build-decks`' simulated-annealing search; once trained, `pick-decks` builds a deck per pool with one forward instead of tens of thousands of scorer evaluations.

**Prerequisites**: card embeddings have been written by `encode-cards`; a trained scorer checkpoint exists (default `models/sealed/scorer/latest.pt`); a pools file exists at `--pools-path`. GPU strongly recommended at full scale.

```bash
# Train a picker from scratch against the default scorer
python -m sealed train-picker \
    --pools-path output/sealed/pools/pools.txt

# Explicit scorer + auditor scorer for the cross-scorer reward-hacking audit
python -m sealed train-picker \
    --pools-path output/sealed/pools/pools.txt \
    --scorer-checkpoint models/sealed/scorer/best_gen4_512.pt \
    --auditor-scorer-checkpoint models/sealed/scorer/best_gen3_256.pt

# Resume a stopped run (architecture inherited from the checkpoint)
python -m sealed train-picker \
    --pools-path output/sealed/pools/pools.txt \
    --resume models/sealed/picker/latest.pt
```

| Argument | Default | Description |
|---|---|---|
| `--pools-path` | _(required)_ | Pre-generated pools file (`SET_CODE;Card1\|...`). One shuffled pass = one epoch. |
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer used as the reward function. Must exist or the run fails fast. |
| `--auditor-scorer-checkpoint` | _(none)_ | Optional second scorer; enables the per-epoch cross-scorer Spearman audit on the validation decks. |
| `--cards-path` | `output/cardsfolder/` | Card-embedding directory (one `.npz` per card). |
| `--checkpoint-dir` | `models/sealed/picker/` | Output dir for `latest.pt` + `best_{timestamp}.pt`. |
| `--resume` | _(none)_ | Continue a stopped run (weights, optimizer, epoch, best val reward). Architecture flags forbidden. Mutually exclusive with `--picker-checkpoint`. |
| `--picker-checkpoint` | _(none)_ | Bootstrap a fresh run from this checkpoint's weights only. Architecture flags forbidden. Required when `--kl-coef` is non-zero. Mutually exclusive with `--resume`. |
| `--d-model` | _(derived = embedding width)_ | Picker internal width; a value other than the embedding width inserts an input projection. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--n-layers` | `4` | Number of SAB layers. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--n-heads` | `8` | Attention heads per SAB; must divide `d_model`. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--ff-dim` | `4 * d_model` | Feed-forward dim. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--dropout` | `0.0` | Dropout in SAB layers. Forbidden alongside `--resume` / `--picker-checkpoint`. |
| `--aux-weight` | `0.1` | Coefficient on the auxiliary pool-quality MSE loss (`0` disables it, head stays present). |
| `--batch-size` | `16` | Pools per gradient step. |
| `--n-samples` | `64` | Sampled decks per pool per step. |
| `--temperature` | `1.0` | Softmax temperature for sampling. |
| `--entropy-coef` | `0.01` | Initial entropy coefficient. |
| `--entropy-decay-after` | `5` | Consecutive improving-val epochs before entropy starts decaying. |
| `--lr` | `3e-4` | AdamW learning rate. |
| `--max-grad-norm` | `1.0` | Per-parameter-group L2-norm cap. |
| `--epochs` | `100` | Maximum epochs. |
| `--val-fraction` | `0.2` | Front fraction of the pools file held out for validation (excluded from shuffles, reused each epoch). |
| `--patience` | `10` | Early-stop after this many epochs without validation-reward improvement. |
| `--kl-coef` | `0.0` | KL penalty against `--picker-checkpoint`'s reference distribution. Non-zero requires `--picker-checkpoint`. |

The random seed is hardcoded to `42` (weight init, pool shuffle, deck sampling, train/val split). Each epoch logs the loss decomposition (`policy_loss`, `entropy_loss`, `aux_loss`), `val_reward`, the distributional summaries (`colors_mean`, `creatures_mean`, `type_creature_share`, 5-bin `cmc_hist`), and — when an auditor is configured — `audit_corr`.

**Exit codes**: `0` success; `2` argument/configuration error (mutually-exclusive flags, architecture flag with `--resume`/`--picker-checkpoint`, missing scorer, width mismatch, `--kl-coef` without `--picker-checkpoint`); `6` architecture validation error (`n_heads` does not divide `d_model`); `130` interrupted.

**Output**: `models/sealed/picker/latest.pt` (overwritten each epoch — the resume point) and `models/sealed/picker/best_{timestamp}.pt` (overwritten whenever validation reward sets a new best). Each checkpoint stores picker weights only (no scorer/encoder weights), the `PickerConfig`, optimizer state, epoch, best validation reward, and a flattened `train_config` for resume precedence.

### pick-decks

The inference counterpart to `build-decks`. Loads a trained picker, runs one deterministic forward + the pick-decomposition walk per pool (23 spells in ranked order plus any nonbasic lands ranked above the 23rd spell), fills basic lands with the existing manabase heuristic, and writes a `generated-decks.txt` that is a drop-in input for `match-outcomes --side-a-decks` / `--side-b-decks`.

**Prerequisites**: card embeddings written by `encode-cards`; a trained picker checkpoint exists; a pools file exists at `--pools-path`.

```bash
python -m sealed pick-decks \
    --pools-path output/sealed/pools/RVR/pools.txt \
    --label picker-gen5

python -m sealed pick-decks \
    --pools-path output/sealed/pools/MH3/pools.txt \
    --label picker-gen5 \
    --picker-checkpoint models/sealed/picker/best_20260520_101500.pt \
    --output output/sealed/generated-decks-picker.txt
```

| Argument | Default | Description |
|---|---|---|
| `--pools-path` | _(required)_ | Input pools file (`SET_CODE;Card1\|...` format). |
| `--label` | _(required)_ | Generation-method tag written verbatim as the first column of every output line. No `;`, `\|`, or whitespace. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Trained picker checkpoint. |
| `--cards-path` | `output/cardsfolder/` | Card-embedding directory (one `.npz` per card). |
| `--output` | `output/sealed/generated-decks.txt` | Output path for generated decks. |
| `--resume` | `False` | Append-and-skip resume (matches `build-decks --resume`): count complete lines in `--output`, skip that many pools, append the rest. Without it `--output` is truncated. |

**Exit codes**: `0` success; `2` argument error or picker/cache width mismatch.

**Output**: `output/sealed/generated-decks.txt` (or `--output`) — one line per pool that produced a viable deck, in `LABEL;SET_CODE;Card1|Card2|...|Card40` format (exactly 40 cards: 23 spells + the picker's nonbasic lands + basic lands from the manabase heuristic). Pools with fewer than 23 embeddable cards are skipped silently.

### analyze-generated-decks

Aggregate composition statistics over one or more generated-decks files — color presence, color-count distribution, pip-share-by-rank, mana curve, type balance, basic/nonbasic land split, pip distribution, and (unless `--no-rarity`) rarity distribution — with a per-label breakdown when more than one `LABEL` is loaded. Useful for inspecting how a builder's decks are shaped (e.g. a self-play generation's color/curve drift).

```bash
python -m sealed analyze-generated-decks                                   # default output/sealed/generated-decks.txt
python -m sealed analyze-generated-decks gen1.txt gen2.txt --no-rarity     # compare labels, skip MTGJSON
```

The same engine powers `python -m draft analyze-generated-decks`, which sources the per-seat decks from a `drafts.jsonl` corpus and adds a `deck_score` summary. It takes a required `--agent` that scopes the report to one agent/mix label (e.g. `--agent draft-agent`), so you compare agents by running it once each.

### ML rationale — `cat([max_pool, mean_pool])` pooling

The pretrained transformer encoder produces a sequence of hidden states (one per token). To get a fixed-size card representation we apply two pooling operations over the token dimension:

- **Max pooling** (`max_pool`): captures the strongest signal from any individual token — good for rare but important features like mana symbols or keyword abilities.
- **Mean pooling** (`mean_pool`): averages signal across all tokens — captures the overall "texture" of the card text.

Concatenating both (`cat([max_pool, mean_pool])`) gives a `2 * d_model` vector that combines both views. This is a well-established technique in sentence-embedding literature. Padding tokens are masked out before both operations so they do not contaminate the result.

The `@torch.no_grad()` decorator is applied because `encode-cards` is inference-only — disabling gradient tracking reduces memory usage and speeds up encoding of large card sets.

**Alternatives rejected** (see `specs/011-sealed-dataset/research.md §2`):

| Alternative | Reason rejected |
|---|---|
| `[CLS]` token pooling | Our tokenizer has no `[CLS]` token; adding one would change the vocabulary |
| Max pooling only | Loses the average-magnitude signal that mean pooling captures |
| Mean pooling only | Loses the peak-feature signal that max pooling captures |
| Pass zero meta to `forward()` | Confusing and fragile — `encode()` makes the intent explicit |

models/             Trained model artifacts (.gitignored)
resources/          Frozen MTGJSON data files
```

## `python -m draft` — Draft Agent

The `draft` module trains a generation-1 MTG-draft agent: a two-headed set
transformer (imitation **policy** over the cards in the current pack + a
Monte-Carlo **critic** on a context token) learned offline from a corpus of
Forge-generated drafts. It reuses the sealed scorer, picker, greedy builder,
embedding cache, and Forge worker pattern. Per-command flags and the data
formats are documented in `CLAUDE.md` and `specs/018-draft-agent/quickstart.md`;
the launch entries are:

```bash
# 1. (optional) decide picker vs SA as the labeling builder
python -m draft validate-builder --pools-from output/draft/drafts.jsonl

# 2. generate a labeled draft corpus (drives Forge's draft AI for all pod seats)
python -m draft generate-draft-data --n-drafts 1000          # random set per draft
python -m draft generate-draft-data --n-drafts 1000 --set BLB
python -m draft generate-draft-data --n-drafts 1000 --resume # continue a stopped run

# 2b. let a trained agent pilot live seats (self-play corpus + in-pod vs Forge)
python -m draft generate-draft-data --n-drafts 500 --set BLB \
  --agent-mix forge-full:7,draft-agent:1 \
  --agent-checkpoint draft-agent=models/draft/agent/latest.pt

# 3. train the gen-1 two-headed agent (policy + critic), offline imitation + critic
python -m draft train-draft-agent
python -m draft train-draft-agent --imitation-weight 0       # critic-only ablation
python -m draft train-draft-agent --resume models/draft/agent/latest.pt

# 4. gen-2: RL self-play cycle (operator-driven; repeat to advance generations)
#  a) sample-mode self-play corpus generated BY the reference checkpoint
python -m draft generate-draft-data --n-drafts 5000 --set BLB \
  --agent-checkpoint gen-k=models/draft/agent/<champion>.pt \
  --agent-mix gen-k:6,forge-full:1,forge-r100:1 \
  --pick-mode sample --temperature 1.0 --seed 42 \
  --output-path output/draft/drafts-genK.jsonl
#  b) on-policy RL fine-tune (REINFORCE + GAE + KL anchor) → next generation
python -m draft train-draft-agent-rl \
  --checkpoint models/draft/agent/<champion>.pt \
  --drafts-path output/draft/drafts-genK.jsonl \
  --learner-agents gen-k --rollout-temperature 1.0   # MUST match step (a)
#  c) cross-generation yardstick: one greedy fixed-mix co-seated run, then
#     read each generation's mean deck_score and decide promotion by hand
python -m draft generate-draft-data --n-drafts 3000 --pick-mode argmax \
  --agent-checkpoint cand=models/draft/agent/<new>.pt \
  --agent-checkpoint gen-k=models/draft/agent/<champion>.pt \
  --agent-mix cand:1,gen-k:1,forge-full:1,forge-r100:1 \
  --output-path output/draft/yardstick-genK.jsonl

# 5. gen-3: online self-play GRPO — one command owns generate→update→discard→repeat
python -m draft train-draft-agent-online \
  --learner gen-3=models/draft/agent/<champion>.pt \
  --frozen  gen-1=models/draft/agent/<champion>.pt \
  --mix "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1" \
  --build-method greedy -T 2.0 --lr 1e-4 --drafts-per-round 10 --set BLB \
  2>&1 | tee output/draft/gen3-run.log
#   then pause on a margin plateau and run the SAME step-4c yardstick + analysis

# (any time) inspect one agent's decks: deck-score + composition stats
python -m draft analyze-generated-decks --agent draft-agent   # then --agent forge-full to compare
```

Gen-2 (`train-draft-agent-rl`) fine-tunes a trained agent past the imitation
ceiling with on-policy actor-critic RL on its own self-play rollouts. The
reference `--checkpoint` warm-starts the actor + critic and is the KL anchor; the
`--drafts-path` corpus it generated is the **sole** policy-gradient source
(add older corpora as `--critic-corpus` for critic coverage only).
`--rollout-temperature` is required and **must equal** the `--temperature` the
corpus was sampled at. The cross-generation yardstick and the generate→train→
evaluate→promote loop are operator runbooks over existing commands (no new code);
promotion is a manual judgment. Pairing a corpus to its generating checkpoint is
by operator convention (e.g. the `drafts-genK.jsonl` filename) — the trainer only
**warns** on an apparent mismatch, never rejects.

Gen-3 (`train-draft-agent-online`) replaces that manual cycle with one
long-running in-process loop: each round it generates `--drafts-per-round` fresh
drafts whose learner seats are piloted by the **live in-training policy**, takes
one pass of the single critic-free term `−A·logπ_T(a|s)` over those seats' picks,
discards the batch, and drafts the next round with the updated weights. There is
no critic term, GAE, KL anchor, entropy bonus, validation split, or early stop —
the operator tunes exactly three knobs (`--lr`, `-T`, `--drafts-per-round`).
`--learner LABEL=PATH` (exactly one, label required) is the agent under training;
`--frozen LABEL=PATH` binds untrained references, and `--anchor` picks which one
the margin is measured against (defaulting to the sole frozen label). A frozen
anchor is required and must stay fixed for the whole campaign — the moment it
moves, the margin stops meaning "improvement over a fixed point".

Every round prints four diagnostic axes plus a consolidated summary line, so
progress, stagnation and collapse are all readable from the run log alone:
**reward** (reward mean/std, advantage spread, near-zero fraction → "nothing to
learn"), **explore** (entropy, perplexity, off-argmax rate → collapse toward
`ppl → 1`), **movement** (mean logπ, policy loss, pre-clip gradient norm, *two* KLs and the
current LR), and **progress** (the live anchor margin over a sliding
`--anchor-window` of drafts, plus every label's raw mean). The two KLs answer
different questions: `KL(prev||new)` is this round's step — large or erratic means
the step is too big — while `KL(init||new)` is the total distance from the run's
warm start, and a flat one means the LR is too small to move the policy at all.
Neither implies the other, since small steps in a consistent direction accumulate.
A degenerate round (fewer than two surviving learner rewards, or zero variance)
is a safe no-op logged as `skipped (no signal)`.

Three checkpoint kinds land under `models/draft/agent/`: `latest.pt` every round,
`best_{timestamp}.pt` on each new best anchor margin, and `{timestamp}.pt` on the
`--snapshot-every` cadence plus once at run end. The best is **advisory** — it is
a maximum over a correlated series, so it reads high, and because the margin
trails the policy by about half the anchor window (in rounds) the genuinely best
weights may sit a few rounds earlier, which is what the periodic snapshots are
for. No best is recorded until the window is full, so an early lucky round cannot
pin it.

Two opt-in run-control knobs, both off by default: `--patience N` stops after N
rounds with no new best margin, and `--lr-decay-patience N` anneals the LR by
`--lr-decay-factor` (0.1) down to `--min-lr` (`lr·1e-3`) instead. They share one
stall counter and a decay resets it, so with both armed a run anneals its way
down and only stops at the LR floor — the pattern that extracted extra quality
from gen-1. Each must exceed the anchor window measured in rounds
(`--anchor-window / --drafts-per-round`) and annealing must trigger before
stopping, or the command fails fast; the startup echo prints the window's length
in rounds and its implied lag so you can size them.

Two operational notes. The `-Ddraft.required.agent` property that guarantees
every pod carries a learner seat is new, so the **fat JAR must be rebuilt**
(`cd forge-connector && mvn package -DskipTests`) before the first gen-3 run.
And `models/draft/agent/latest.pt` is rewritten every round, so during a run it
tracks the in-progress gen-3 — pin a timestamped snapshot, not `latest.pt`, when
you want a stable candidate. Stopping and promotion stay operator judgments:
there is no `--resume` (point `--learner` at the last snapshot and restart, which
intentionally re-runs the LR warmup), and the cross-generation yardstick is the
unchanged gen-2 procedure above — a greedy fixed-mix co-seated
`generate-draft-data --pick-mode argmax` run, then one
`analyze-generated-decks --agent <label>` per agent to read each generation's
mean `deck_score` on one shared absolute scale. See
`specs/021-draft-online-grpo/quickstart.md` for the full runbook.

A draft-agent checkpoint can **pilot live seats**: bind a mix label to a
checkpoint with `--agent-checkpoint LABEL=PATH` (repeatable; bare `PATH` ⇒ label
`draft-agent`) and any seat sampled to that label asks the trained policy for its
pick over a worker↔supervisor side-channel. `--pick-mode {argmax,sample}`
(default `argmax`), `--temperature`, and `--seed` control pick determinism
(seeded `sample` is reproducible); `--max-consecutive-faults` (default 5) aborts
the run if picks keep faulting. A pick fault (policy error, protocol desync, or
every legal card un-embeddable) **abandons the whole draft** — no substitute is
ever recorded — and the worker's per-run stderr log lands at
`output/draft/worker-<run_id>.log`. With no `--agent-checkpoint` the command is
byte-for-byte the gen-1 behavior.

Requires the `forge-connector` fat JAR (now also containing `DraftWorkerMain`),
a frozen sealed scorer + picker, and a populated `.npz` card cache. The corpus
is `output/draft/drafts.jsonl` (one self-contained record per line); checkpoints
land under `models/draft/agent/`. Full per-command detail intentionally lives in
`CLAUDE.md` and the spec's `quickstart.md` to avoid duplicating it here.
