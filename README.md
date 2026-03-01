# Price Predictor

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

# Install the package with dev dependencies
pip install -e ".[dev]"

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

### Train a model

Reads card attributes from Forge card scripts, joins them with Cardmarket EUR
prices from MTGJSON data, and trains a prediction model.

**Inputs**: Forge cardsfolder directory (~32,000 `.txt` card scripts),
`AllPrintings.json` (name-to-UUID mapping), `AllPricesToday.json` (EUR prices).

**Processing**:
1. Parse each Forge card script into structured attributes (name, mana cost,
   types, oracle text, power/toughness, keywords, abilities).
2. Build a card-name-to-UUID mapping from `AllPrintings.json`, filtering to
   paper-available, English, non-funny, non-online-only entries.
3. For each card, look up all printing UUIDs and find the cheapest Cardmarket
   EUR price across all printings (normal and foil).
4. Exclude cards with no price data or that failed to parse.
5. Engineer features from card attributes (see [Feature Engineering](#feature-engineering)).
6. Log-transform prices (to handle the skewed distribution of card values).
7. Train a Gradient Boosted Trees model on an 80/20 train/test split
   (fixed random seed 42 for reproducibility).
8. Save the trained model artifact to `models/`.

**Output**: A `.joblib` model file and a JSON summary to stdout reporting
cards used, cards skipped (with reasons), and price range. Progress messages
are printed to stderr so the user can follow each stage. To capture clean
JSON only: `python -m price_predictor train ... > result.json` (progress
remains visible on the console).

```bash
python -m price_predictor train \
  --forge-cards-path ../forge/forge-gui/res/cardsfolder \
  --prices-path resources/AllPricesToday.json \
  --printings-path resources/AllPrintings.json
```

Options: `--output-path`, `--test-split`, `--random-seed`. See full options
with `python -m price_predictor train --help`.

### Predict a card price

Takes card attributes as input and returns an EUR price estimate. Works for
any combination of valid attributes, including cards that do not exist.

**Inputs**: Card attributes provided as CLI flags (types, mana cost, oracle
text, power, toughness, keywords, etc.). A trained model file.

**Processing**:
1. Parse CLI arguments into a Card entity.
2. Validate input (at least one card type is required).
3. Load the trained model from disk.
4. Transform the card through the same feature engineering pipeline used
   during training.
5. Run the model's predict function on the feature vector.
6. Exp-transform the log-price prediction back to EUR.

**Output**: JSON to stdout with `predicted_price_eur` and `model_version`.

```bash
python -m price_predictor predict \
  --types "Creature" \
  --supertypes "Legendary" \
  --subtypes "Human,Wizard" \
  --mana-cost "1 U R" \
  --power "2" \
  --toughness "2" \
  --keywords "Prowess" \
  --oracle-text "Whenever you cast a noncreature spell, draw a card."
```

Options: `--model-path`, `--loyalty`, `--colors`. See full options
with `python -m price_predictor predict --help`.

### Evaluate model accuracy

Computes accuracy metrics on a held-out test set to measure how well the
model predicts real card prices.

**Inputs**: A trained model file, the same Forge cards and MTGJSON data used
for training.

**Processing**:
1. Load the trained model.
2. Re-derive the train/test split using the same random seed as training
   (ensuring the test set was not seen during training).
3. Predict prices for all cards in the test set.
4. Compute metrics: mean absolute error (EUR), median percentage error,
   top-20% price tier overlap.
5. Optionally write per-card results to a CSV file.

**Output**: JSON to stdout with accuracy metrics. Progress messages appear
on stderr (same behaviour as `train`).

```bash
python -m price_predictor evaluate \
  --model-path models/latest.joblib
```

Options: `--forge-cards-path`, `--prices-path`, `--printings-path`,
`--test-split`, `--random-seed`, `--output-csv`. See full options
with `python -m price_predictor evaluate --help`.

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

Colorless mana ({C}) is tracked separately from generic mana ({1}--{N}).
Rarity is deliberately excluded — the model predicts from game
characteristics only, which keeps it consistent with the made-up card use
case where rarity is undefined.

## Artifacts

### Trained model files

- **Location**: `models/`
- **Format**: `.joblib` (serialized scikit-learn model + fitted feature
  engineering pipeline)
- **Naming**: `{ISO-timestamp}.joblib` (e.g., `20260301-143000.joblib`)
- **Latest**: `models/latest.joblib` (copy of the most recent model)
- **Contents**: The GradientBoostingRegressor model and the fitted
  FeatureEngineering object (including the TF-IDF vectorizer vocabulary)
  bundled as a single artifact for consistent predictions.

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
# Fast unit tests (default)
pytest

# Integration tests only
pytest tests/integration/

# All tests
pytest tests/

# Linting
ruff check src/ tests/
```

## Project Structure

```
src/price_predictor/
  domain/           Pure game entities and value objects (no dependencies)
    entities.py     Card, PriceEstimate, TrainingExample, TrainedModel, EvaluationMetrics
    value_objects.py ManaCost (parsed Forge mana format)
  application/      Use cases (depends on domain only)
    train.py        TrainModelUseCase
    predict.py      PredictPriceUseCase
    evaluate.py     EvaluateModelUseCase
    feature_engineering.py  Card -> numeric feature vector
  infrastructure/   External integrations (depends on application)
    cli.py          argparse CLI (train, predict, evaluate subcommands)
    forge_parser.py Forge card script parser
    mtgjson_loader.py AllPrintings/AllPricesToday loaders
    model_store.py  Model save/load (joblib)
tests/
  unit/             Fast unit tests (fixture-based)
  integration/      End-to-end pipeline tests
  fixtures/         Sample card scripts and JSON data
models/             Trained model artifacts (.gitignored)
resources/          Frozen MTGJSON data files
```
