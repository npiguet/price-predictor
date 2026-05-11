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

Entry point: `python -m sealed <subcommand>` (see `src/sealed/infrastructure/cli.py`). The sealed pipeline is built around a sealed-trained card encoder; the price-predictor transformer is a valid alternate encoder source via `--encoder-checkpoint`.

Subcommands:
- **`build-vocab`** — scans the converted card corpus (default `output/cardsfolder/`) and writes a sealed-side tokenizer vocab to `models/sealed/encoder/vocab.txt`. Wraps `price_predictor.application.build_vocabulary`; `--target-size` (default 5000) post-truncates the corpus-frequency vocab while preserving seeded specials and domain tokens. The seeded specials include `[PAD]`, `[UNK]`, `cardname`, and `[MASK]` (the last reserved for `train-encoder`'s MLM auxiliary loss). Independent from the price-side `vocab.txt` — building one does not modify the other.
- **`train-encoder`** — reads `output/sealed/cards-played.txt` in one streaming pass, aggregating per-card winnability inline (primary + `@play` counters + per-color slices keyed off each card's `mana cost:` line all in the same pass), and computes nine per-card regression labels (`score_play`, `score_draw`, `played_rate`, `cast_lift`, and `color_lift_X` for each WUBRG letter) in raw and shrunk form (`--shrinkage-k`, default 20). Splits cards stratified by `score_play` quartile (20% val, card-disjoint, FR-018 fallback chain for cards with empty `score_play`), tokenizes the corpus once, and trains a token encoder + card encoder + five regression heads + an MLM head from random init on `L_reg + (--mlm-weight) · L_mlm`. `L_reg` uses per-head per-batch sum-to-1 weighted MSE with FR-017a per-card weights; `L_mlm` is cross-entropy at the masked positions only (`--mlm-mask-prob` default 0.15, mask drawn fresh per batch over non-special, non-pad tokens; the MLM head is applied only at those positions). Batches are length-bucketed (similar-length cards grouped, batch order reshuffled per epoch) and padded per-batch to the batch's longest card. Optimizer: AdamW with per-parameter-group max-norm 1.0 gradient clipping; LR schedule: linear warmup over the first 5% of `--epochs × batches_per_epoch` scheduled steps, then constant `--lr`. Saves only the encoder weights to `models/sealed/encoder/{timestamp}.pt` plus `latest.pt`; the regression heads and MLM head are filtered out at save time. Writes the per-card label snapshot to `output/sealed/cards-win-rates.txt` (23 columns, sorted by `shrunk_score_play` desc) for SC-005 inspection. Best checkpoint selected by full validation loss (`L_reg + (--mlm-weight) · L_mlm`). Cards referenced in `cards-played.txt` with no converted `.txt` under `--cards-folder` are reported via a log warning (naming up to 20 + the total) and dropped — they don't block the run. Each epoch logs the loss decomposition plus, on the val set, MLM perplexity / top-1 masked-token accuracy and the per-head Pearson correlation between predictions and targets, and a one-batch gradient-norm probe of the regression-vs-MLM pull on the shared encoder (`‖∂L_reg/∂trunk‖` vs `‖∂(w·L_mlm)/∂trunk‖`, measured on the epoch's first batch via two extra `autograd.grad` passes). Hardcoded constants: `d_model=256`, `ff_dim=1024`, `val_fraction=0.2`, `random_seed=42`; `max_seq_len` (the position-embedding table size) is computed from the corpus.
- **`encode-cards`** — strips the `name:` line from each card and writes a `.npz` file next to every `.txt` in `output/cardsfolder/`. Each `.npz` stores a `float32` array of shape `(2 * d_model + FEATURE_COUNT,)` under key `"embedding"`, produced by `cat([attn_pool, max_pool])` (sealed encoder) or `cat([max_pool, mean_pool])` (price encoder) over the encoder's token outputs (padding masked) plus deterministic game features. Encoder weights default to `models/sealed/encoder/latest.pt` via `--encoder-checkpoint`; pass `--scorer-checkpoint <phaseB>.pt` instead to extract encoder weights from a Phase B sealed scorer checkpoint (the two flags are mutually exclusive when both are explicitly passed). The `--vocab-path` default is `models/sealed/encoder/vocab.txt`. Re-running is idempotent; `--clean` forces a full re-encode.
- **`generate-pools`** — invokes the forge-connector JAR (`PoolMain`) to generate N sealed pools (6 boosters each); writes `pools.txt` (one pool per line in `SET_CODE;Card1|Card2|...|CardN` format, basics excluded). With `--set RVR` all pools come from the given set and are written to `output/sealed/pools/{set}/`; without `--set`, each pool uses an independently-selected random sealed-legal set and output defaults to `output/sealed/pools/`.
- **`build-decks`** — reads a pools file (`SET_CODE;Card1|...` format), loads a trained scorer checkpoint, and greedily builds one 40-card deck per pool (23 nonlands chosen by the scorer + basic lands from the manabase heuristic). Writes `generated-decks.txt` (one deck per line in `LABEL;SET_CODE;Card1|...|Card40` format) to `output/sealed/` by default. `--label` is **required** and identifies the generation method (e.g. `gen-2`); it is later read back by `match-outcomes` as the `method_A` / `method_B` tag for self-play matches that play this deck. Consumed by `match-outcomes --side-a-decks` / `--side-b-decks` for self-play.
- **`match-outcomes`** — long-running supervisor that spawns Java `MatchWorkerMain` workers. Each match independently chooses how to produce deck A and deck B:
  - **Side A**: sampled from `--side-a-decks <path>` if given (`method_A` = the deck's `LABEL`), else built by the 4 Forge methods (weights 4:3:2:1) on a freshly-generated pool from a random sealed-legal set.
  - **Side B**: rolled between the 4 Forge methods (combined weight 10) on a fresh pool of deck A's set, and — if `--side-b-decks <path>` is given — sampling from that file (weight `--side-b-decks-weight`, default 4) filtered to deck A's set code. `--side-b-decks-weight` requires `--side-b-decks`. Mirror matches are excluded by content equality on the card-name multiset, so deck B never matches deck A. If no non-mirror file deck exists for deck A's set, the file roll falls back to Forge methods.

  Phase 0 (both flags absent) reproduces the original "4 Forge methods on both sides, random eligible set" behavior. `--best-of` (default 7) controls games per match; must be a positive odd integer.

  Match-level outcomes are appended to `output/sealed/match-outcomes.txt`; per-game card-play data (one line per played game) is appended to `output/sealed/cards-played.txt` by the same worker. The supervisor restarts crashed workers — long Forge AI games can crash the JVM and that's expected. Ctrl-C to stop.
- **`train-scorer`** — trains a Set Transformer deck scorer on `match-outcomes.txt` using AdamW with per-parameter-group max-norm 1.0 gradient clipping and `--patience`-driven early stopping (default 5). Architecture flags (`--n-layers/--n-heads/--n-seeds/--d-ff/--mlp-hidden/--dropout`) configure a fresh run; on `--resume` or `--scorer-checkpoint` they are forbidden (architecture inherits from the loaded checkpoint). Two phases: **Phase A** (default, `--embedding-lr 0`) consumes the `.npz` cache with the encoder frozen; **Phase B** (`--embedding-lr <nonzero>`) jointly fine-tunes the encoder alongside the scorer, requiring either `--scorer-checkpoint <phaseA>.pt` (fresh kickoff, encoder weights from `--encoder-checkpoint` defaulting to `models/sealed/encoder/latest.pt`) or `--resume <phaseB>.pt` (continuation, encoder weights from the resumed checkpoint). When the default sealed encoder is missing on a Phase B fresh kickoff, the run errors out and points the user at `train-encoder` (or at passing `--encoder-checkpoint <path>` explicitly). Checkpoints saved to `models/sealed/scorer/`; Phase B checkpoints carry an additional `encoder_state_dict` + `encoder_config` so the saved file is self-contained. Best checkpoint selected by validation accuracy.
- **`evaluate-scorer`** — generates fresh sealed pools, has the scorer greedily build a deck from each, and has Forge AI play matches between the scorer's deck and Forge's own optimal builder. Outputs win/loss stats. Spawns Java workers via `evaluation_connector.py`. With `--set BLB` all pools are from the given set; without `--set`, a random sealed-legal set is selected.

Match-outcome file format (`output/sealed/match-outcomes.txt`): one line per match, ten semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s`. `timestamp` is ISO 8601 UTC, `run_id` is a UUID identifying the supervisor invocation, `set_code` is the MTG set both pools were drawn from, `method_A`/`method_B` are build-method tags (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or the per-deck label set by `build-decks --label` when the deck came from a generated-decks file), `deck_A`/`deck_B` are pipe-separated lists of 40 Forge canonical card names (duplicates repeat), `games` is the per-game winner sequence (`AA`, `ABA`, `ABABABA`, etc. — length depends on `--best-of`, between `ceil(N/2)` and `N` chars), `play` is the per-game play-first sequence (same length), and `duration_s` is the match wall-clock duration in whole seconds. See `specs/sealed-deck-picker.md` §Phase 0 Step 4 for full details.

Cards-played file format (`output/sealed/cards-played.txt`): one line per played game, eleven semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter`. `timestamp`/`run_id`/`set_code`/`method_A`/`method_B` mirror the parent match's row (joinable by `run_id` plus positional offset within a `(run_id, set_code, method_A, method_B)` group). The four card-list columns are pipe-separated **sets** of distinct card names (no duplicates within a column; basic lands excluded by the writer). For a side X, `cards_played_X` and `cards_not_played_X` are disjoint — if any copy of a card was played, the name appears only in `cards_played_X`. `winner` and `starter` are single chars `A`/`B` matching the corresponding chars of the parent's `games`/`play` strings. Trailing partial lines are tolerated by readers (JVM-crash-mid-write recovery).

Cards-win-rates file format (`output/sealed/cards-win-rates.txt`): overwritten by every `train-encoder` run; one header row + N data rows, semicolon-separated, sorted by `shrunk_score_play` descending (cards with empty `shrunk_score_play` sort to the end). Schema (23 columns): `card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G`. Floats formatted to five decimals; counters as plain integers. Cells whose slice denominator is zero are written as the empty string in both raw and shrunk columns (distinguishes "no signal" from "neutral signal"). Cards with `wins_when_in_deck + losses_when_in_deck == 0` are excluded entirely.

Pool file format (`output/sealed/pools/*/pools.txt`): one pool per line, `SET_CODE;Card1|Card2|...|CardN`. The set-code prefix lets downstream tools (`build-decks`, self-play `match-outcomes`) honor same-set constraints.

Generated-decks file format (`output/sealed/generated-decks.txt`): one finished 40-card deck per line, `LABEL;SET_CODE;Card1|Card2|...|Card40`. `LABEL` is the value passed to `build-decks --label` and is recorded as the `method_A` / `method_B` tag whenever this deck is sampled into a self-play match. Concatenating multiple generated-decks files with different labels into one self-play corpus is supported.

Key modules inside `sealed`:
- `domain/` — `card_encoder.py` (wraps the encoder for inference), `encoder_model.py` (sealed transformer + multi-query attention pool), `scorer_model.py` (Set Transformer architecture), `deterministic_features.py`.
- `application/` — `build_vocab.py`, `train_encoder.py`, `encode_cards.py`, `generate_pools.py`, `build_decks.py`, `match_outcomes.py`, `train_scorer.py`, `evaluate_scorer.py`.
- `infrastructure/` — `cli.py`, `cards_played_reader.py`, `encoder_store.py`, `embedding_store.py`, `pool_connector.py`, `match_worker_connector.py`, `evaluation_connector.py`, `match_data_loader.py`, `scorer_store.py`, `card_name_corrections.py`.

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

`sealed` **imports from** `price_predictor` (tokenizer, vocabulary builder, transformer model/store) — `price_predictor` provides the shared MTG tokenizer, vocab-build utility, and the alternate price-trained encoder available via `--encoder-checkpoint`. Don't create the reverse dependency.

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
    encoder/        {timestamp}.pt, latest.pt, vocab.txt
    scorer/         checkpoints (best_* selected by val accuracy)
```

Inputs the code expects to find on disk:
- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps.
- `../forge/forge-gui/res/cardsfolder/` — Forge card scripts (source for `convert`).
- `output/cardsfolder/` — converted card text files, each paired with a `.npz` after `encode-cards` runs.
- `output/sealed/pools/{set}/pools.txt` or `output/sealed/pools/pools.txt` — generated sealed pools (`SET_CODE;Card1|...` per line).
- `output/sealed/generated-decks.txt` — scorer-built 40-card decks from `build-decks` (`LABEL;SET_CODE;Card1|...|Card40` per line); input to `match-outcomes --side-a-decks` / `--side-b-decks`.
- `output/sealed/match-outcomes.txt` — append-only training data for the scorer.
- `output/sealed/cards-played.txt` — append-only per-game card-play log; the input to `train-encoder` after label aggregation.

## Specs

Feature specs live under `specs/NNN-name/` (001 through 013 so far) and are the primary source of "why". Before starting non-trivial work in an area, read the relevant spec's `spec.md` / `plan.md` / `research.md`. The `.specify/` directory holds spec-kit templates — invoke them via the `speckit.*` skills rather than editing by hand.
