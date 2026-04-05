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

# Train the transformer with auxiliary supervision (mana-aware embeddings)
python -m price_predictor train transformer --aux-lambda 0.2
```

Models are saved to `models/sklearn/` and `models/transformer/` respectively,
with timestamped filenames and a `latest` copy.

Options: `--output-dir`, `--prices-path`, `--printings-path`, `--model-output`,
`--test-split`, `--random-seed`. Transformer adds `--batch-size`, `--epochs`,
`--lr`, `--patience`, `--vocab-path`. See `python -m price_predictor train sklearn --help`.

The `--aux-lambda` flag (default `0.04`) enables auxiliary supervision during transformer
training: 20 linear heads teach the encoder to represent mana cost, card color, pip
counts, and mana production. After training, validate with `python -m sealed validate-embeddings`.
See `specs/015-embeddings-retraining/quickstart.md` for detailed tuning guidance.

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

### train

Runs a PPO reinforcement-learning training loop that teaches the pool-level transformer to make 40 consecutive legal picks from a 96-card sealed pool (Stage 1). Training resumes automatically from the latest checkpoint if one exists.

```
python -m sealed train \
    --stage      1 \
    [--set        RVR] \
    [--pools-path output/sealed/pools/{set}/] \
    [--cards-path output/cardsfolder/] \
    [--model-path models/sealed/stage1/latest.pt] \
    [--batch-size 32]
```

| Argument | Default | Description |
|---|---|---|
| `--stage` | *(required)* | Training stage; currently only `1` is supported |
| `--set` | `RVR` | MTG set code for the pool dataset |
| `--pools-path` | `output/sealed/pools/{set}/` | Directory containing `pools.txt` |
| `--cards-path` | `output/cardsfolder/` | Directory containing `.npz` embedding files |
| `--model-path` | `models/sealed/stage1/latest.pt` | Path to load and save the checkpoint |
| `--batch-size` | `32` | Episodes collected per PPO update |

**Console output** (one line per batch):
```
[ep 64] batch runs: 1,3,2,1,...  best_run=5  mean_reward=-0.712
```

**Checkpoints**: `latest.pt` is overwritten atomically after every batch. Timestamped checkpoints are saved every 1000 episodes under `models/sealed/stage1/checkpoints/`.

**Stage 1 complete** when 100 consecutive episodes achieve 40 legal picks. Exit code `0`.

**Exit codes**: `0` = success, `1` = unknown stage, `2` = missing `pools.txt` or card embeddings.

### sample

Loads a trained checkpoint and prints N human-readable pick sequences with SUCCESS/ILLEGAL PICK results. Useful for qualitative inspection of model behaviour during or after training.

```
python -m sealed sample \
    [--set        RVR] \
    [--pools-path output/sealed/pools/{set}/] \
    [--cards-path output/cardsfolder/] \
    [--model-path models/sealed/stage1/latest.pt] \
    [--n-samples  10]
```

| Argument | Default | Description |
|---|---|---|
| `--set` | `RVR` | MTG set code for the pool dataset |
| `--pools-path` | `output/sealed/pools/{set}/` | Directory containing `pools.txt` |
| `--cards-path` | `output/cardsfolder/` | Directory containing `.npz` embedding files |
| `--model-path` | `models/sealed/stage1/latest.pt` | Checkpoint to load |
| `--n-samples` | `10` | Number of pick sequences to generate |

**Sample output**:
```
Sample 1:
  Pick  1: Skyknight Legionnaire
  Pick  2: Mountain
  ...
  Pick 40: Swamp
  Result: SUCCESS (40/40 legal picks)

Sample 2:
  Pick  1: Counterspell
  Pick  2: Counterspell
  Result: ILLEGAL PICK at step 2 (1/40 legal picks)
```

**Exit codes**: `0` = success (sampling is non-destructive), `2` = missing checkpoint.

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
