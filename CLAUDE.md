# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An ML system that predicts Magic: The Gathering card EUR prices from game-visible attributes (mana cost, types, oracle text, power/toughness, keywords, printing metadata). Works for both real cards and hypothetical ones. Also hosts a sealed-format ML pipeline (card embeddings, pool generation, deck scorer) built on top of the price predictor's transformer encoder.

## Tech stack

- **Python 3.14+** (required — see `pyproject.toml`); manage with `venv` + `pip`. Python executable is `python`, pip is `pip`.
- **Java 17+** for the `forge-connector` Maven module (used for card script conversion and Forge-AI match simulation).
- Dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib`, `ijson`, `fastapi`, `uvicorn`, `torch`, `transformers`. PyTorch is installed with CUDA 12.6 wheels via `--extra-index-url https://download.pytorch.org/whl/cu126`.
- Sibling checkout of MTG Forge expected at `../forge` (built with `mvn install -DskipTests`). MTGJSON data files (`AllPrintings.json`, `AllPricesToday.json`) expected in `resources/`.

## Common commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate    # Windows (git-bash: source .venv/Scripts/activate)
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126

# Python tests
pytest                                    # unit + integration (default)
pytest tests/unit/                        # fast unit tests only
pytest tests/integration/                 # integration tests only
pytest tests/unit/application/test_train.py::test_name   # single test
pytest -m "not integration"               # skip integration-marked tests

# Python lint
ruff check src/ tests/

# Java (forge-connector)
cd forge-connector && mvn package -DskipTests    # build fat JAR used by convert/pools/match-outcomes
cd forge-connector && mvn test                   # run JUnit 5 tests
```

Test/lint commands run from the project root. `pyproject.toml` configures `testpaths = ["tests"]` and an `integration` pytest marker.

## Architecture

Two independent Python packages live under `src/`, each laid out in hexagonal (ports-and-adapters) style: `domain` → `application` → `infrastructure`.

### `price_predictor` — the price prediction system

Entry point: `python -m price_predictor <subcommand>` (see `src/price_predictor/__main__.py` and `infrastructure/cli.py`).

Subcommands:
- **`convert`** — shells out to the Java `ConvertMain` to transform Forge's `.txt` card scripts (at `../forge/forge-gui/res/cardsfolder/`) into a compact LLM-friendly text format written to `./output/`. One output file per card with lowercase property lines and classified ability lines.
- **`check-convert`** — compares converted files against MTGJSON oracle text and flags low-similarity cards.
- **`vocabulary`** — scans the converted corpus and writes `models/price-predictor/transformer/vocab.txt` (custom MTG word-level tokenizer, ~5k tokens). **Required before training or running the transformer model.** Also seeds set-code fragments from `AllPrintings.json` when available.
- **`train sklearn`** — trains a `GradientBoostingRegressor` on `log(price)` using a 17-group feature pipeline (mana cost, types, keywords, TF-IDF oracle text, power/toughness, printing metadata). Saves to `models/price-predictor/sklearn/{timestamp}.joblib` + `latest.joblib`.
- **`train transformer`** — trains a custom transformer (tunable `--d-model/--n-layers/--n-heads/--ff-dim`) on tokenized card text. Uses log-price target with `--log-offset`, price-bucket oversampling (`--sampler-exponent`), and validation-accuracy-based best checkpoint selection. Saves to `models/price-predictor/transformer/`.
- **`predict {sklearn,transformer}`** — reads converted card text (from `--file` or inline `--card`) and prints JSON with `predicted_price_eur` + `model_version`. Attaches printing metadata from MTGJSON when the card name matches.
- **`evaluate {sklearn,transformer}`** — computes MAE/median % error/`top_20_overlap` on a held-out test split. Transformer adds per-bucket breakdown.
- **`serve`** — FastAPI + uvicorn REST service exposing `POST /api/v1/predict`. Loads the sklearn model (required), transformer (optional, graceful degradation), tokenizer, and MTGJSON metadata map for auto-fill. Accepts `text/plain` card text and returns predictions from every available model.

Key modules inside `price_predictor`:
- `domain/` — `entities.py` (Card, PriceEstimate, TrainingExample, TrainedModel, EvaluationMetrics), `value_objects.py` (ManaCost, PrintingData), `tokenizer.py` (custom MTG tokenizer).
- `application/` — one file per use case (`train.py`, `train_transformer.py`, `predict.py`, `predict_transformer.py`, `evaluate.py`, `evaluate_transformer.py`, `build_vocabulary.py`, `feature_engineering.py`, `check_convert.py`).
- `infrastructure/` — `cli.py` (argparse wiring), `server.py` (FastAPI app), `converted_card_parser.py`, `mtgjson_loader.py`, `model_store.py` (joblib), `transformer_model.py`, `transformer_store.py` (`.pt`), `transformer_dataset.py`, `tokenizer_store.py`, `metadata_encoder.py`.

The two model types have **different input contracts**:
- `sklearn` takes a parsed `Card` object and runs it through `FeatureEngineering` to produce a numeric vector.
- `transformer` takes raw converted card text, tokenizes it with the custom vocab, and receives `PrintingData` as a side-channel (not embedded in the text).

Prices are trained on `log(price + offset)` and exp-transformed back on inference — the skew of the EUR distribution would otherwise let a handful of expensive cards dominate the loss.

### `sealed` — sealed-format ML pipeline

Entry point: `python -m sealed <subcommand>` (see `src/sealed/infrastructure/cli.py`). Depends on the price-predictor transformer as a frozen card encoder.

Subcommands:
- **`encode-cards`** — loads the pretrained price-predictor transformer, strips the `name:` line from each card, and writes a `.npz` file next to every `.txt` in `output/cardsfolder/`. Each `.npz` stores a `float32` array of shape `(2 * d_model,)` under key `"embedding"`, produced by `cat([max_pool, mean_pool])` over the encoder's token outputs (padding masked). Re-running is idempotent; `--clean` forces a full re-encode.
- **`generate-pools`** — invokes the forge-connector JAR (`PoolMain`) to generate N sealed pools (6 boosters each) for a given set code; writes `pools.txt` (one pool per line, semicolon-separated card names, basics excluded) to `output/sealed/pools/{set}/`.
- **`match-outcomes`** — long-running supervisor that spawns Java `MatchWorkerMain` workers; each worker picks a random sealed-legal set, generates two pools, builds decks (four weighted strategies: 40/30/20/10% from Forge optimal to fully random), plays a best-of-3 match via Forge AI, and appends the result to `output/sealed/match-outcomes.txt`. The supervisor restarts crashed workers — long Forge AI games can crash the JVM and that's expected. Ctrl-C to stop.
- **`train-scorer`** — trains a Set Transformer deck scorer on `match-outcomes.txt`. Architecture is a stack of SAB layers + PMA pooling + MLP head; all hyperparameters are CLI flags (`--n-layers/--n-heads/--n-seeds/--d-ff/--mlp-hidden`). Checkpoints saved to `models/sealed/scorer/`; best checkpoint selected by validation accuracy. Supports `--resume` and two-phase training with `--unfreeze-embeddings` for card embedding fine-tuning.
- **`evaluate-scorer`** — generates fresh sealed pools, has the scorer greedily build a deck from each, and has Forge AI play matches between the scorer's deck and Forge's own optimal builder. Outputs win/loss stats. Spawns Java workers via `evaluation_connector.py`.

Match-outcome file format (`output/sealed/match-outcomes.txt`): one line per match, four semicolon-separated fields `deck_a;deck_b;wins_a;wins_b`, where each deck is a pipe-separated list of 40 Forge canonical card names (duplicates repeat) and `wins_a + wins_b` is 2 or 3.

Key modules inside `sealed`:
- `domain/` — `card_encoder.py` (wraps the transformer for inference), `scorer_model.py` (Set Transformer architecture), `deterministic_features.py`.
- `application/` — `encode_cards.py`, `generate_pools.py`, `match_outcomes.py`, `train_scorer.py`, `evaluate_scorer.py`.
- `infrastructure/` — `cli.py`, `embedding_store.py`, `pool_connector.py`, `match_worker_connector.py`, `evaluation_connector.py`, `match_data_loader.py`, `scorer_store.py`, `card_name_corrections.py`.

### `forge-connector` — Java Maven module

Zero-dependency (stdlib-only) Java 17+ library at `forge-connector/` with two roles:

1. **Client library for Forge**: `PricePredictorClient` + `CardAttributes` give Forge a 5-line API for hitting the `POST /api/v1/predict` endpoint. Used by Forge's deck-building heuristics.
2. **CLI workers invoked by the Python side**: fat JAR built with `mvn package -DskipTests` → `target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`. Main classes invoked by Python subprocess:
   - `ConvertMain` — `python -m price_predictor convert`
   - `PoolMain` — `python -m sealed generate-pools`
   - `MatchWorkerMain` — one per worker in `python -m sealed match-outcomes`
   - `ValidationWorkerMain` — `python -m sealed evaluate-scorer`
   - `DeckBuilderMain` — used during scorer evaluation to build decks from pools

These workers import `forge-game` / `forge-core` from the sibling `../forge` checkout (classpath assembled in `run_convert` in `infrastructure/cli.py`). `ForgeEnvironmentInitializer` bootstraps Forge's static state in each JVM.

### Cross-package dependencies

`sealed` **imports from** `price_predictor` (tokenizer, transformer model/store) — the price-predictor transformer is a frozen feature extractor for the sealed pipeline. Don't create the reverse dependency.

## Tests

- `tests/unit/` — fast, fixture-based unit tests (`application/`, `domain/`, `infrastructure/`, `sealed/`).
- `tests/integration/` — end-to-end pipeline tests (convert → train → predict → serve) and transformer training smoke tests.
- `tests/fixtures/` — sample Forge card scripts and trimmed MTGJSON snippets.
- Java tests in `forge-connector/src/test/java/`; Forge-dependent ones are tagged `@Tag("integration")`.

## Model artifact layout

```
models/
  price-predictor/
    sklearn/        {timestamp}.joblib, latest.joblib
    transformer/    {timestamp}.pt, latest.pt, vocab.txt
  sealed/
    scorer/         checkpoints (best_* selected by val accuracy)
```

Inputs the code expects to find on disk:
- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps.
- `../forge/forge-gui/res/cardsfolder/` — Forge card scripts (source for `convert`).
- `output/cardsfolder/` — converted card text files, each paired with a `.npz` after `encode-cards` runs.
- `output/sealed/pools/{set}/pools.txt` — generated sealed pools.
- `output/sealed/match-outcomes.txt` — append-only training data for the scorer.

## Specs

Feature specs live under `specs/NNN-name/` (001 through 013 so far) and are the primary source of "why". Before starting non-trivial work in an area, read the relevant spec's `spec.md` / `plan.md` / `research.md`. The `.specify/` directory holds spec-kit templates — invoke them via the `speckit.*` skills rather than editing by hand.
