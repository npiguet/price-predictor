# Tasks: Card Price Predictor

**Input**: Design documents from `/specs/001-card-price-predictor/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per plan.md: `src/price_predictor/{domain,application,infrastructure}/` with `__init__.py` files, `tests/{unit/{domain,application,infrastructure},integration,fixtures}/` directories, and `models/` directory (gitignored)
- [x] T002 Initialize Python project with `pyproject.toml` — configure package name `price_predictor`, Python >=3.14, dependencies (scikit-learn, pandas, numpy), dev dependencies (pytest, ruff), and `[project.scripts]` or `__main__.py` entry point
- [x] T003 [P] Configure ruff (linting/formatting) in `pyproject.toml` and pytest settings (test discovery paths, markers for integration tests)
- [x] T004 [P] Create test fixtures directory `tests/fixtures/` with: sample Forge card scripts in `tests/fixtures/forge_cards/` (~10 diverse cards: vanilla creature, legendary creature, planeswalker, instant, sorcery, enchantment, artifact, land, split card, transform card — include at least one card with colorless mana cost {C}), minimal `tests/fixtures/allprintings_sample.json` (matching sample cards with name→uuid mappings), and minimal `tests/fixtures/allprices_sample.json` (matching UUIDs with Cardmarket EUR prices)
- [x] T005 Create shared test configuration in `tests/conftest.py` — fixtures for loading sample Forge cards, sample AllPrintings data, sample AllPricesToday data, and a pre-built small trained model fixture

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Domain entities, value objects, and shared application logic that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase (MANDATORY per Constitution)

- [x] T006 [P] Write tests for Card, PriceEstimate, TrainingExample, TrainedModel, and EvaluationMetrics entities in `tests/unit/domain/test_entities.py` — cover construction, validation rules (non-empty name, at least one type, valid power/toughness format, positive prices), and edge cases (null optional fields, "*" power)
- [x] T007 [P] Write tests for ManaCost value object in `tests/unit/domain/test_value_objects.py` — cover parsing of Forge mana format: simple costs ("2 W W"), hybrid ("WU"), phyrexian ("WP"), X costs ("X R"), colorless ("C" — must be tracked separately from generic mana), snow ("S"), two-or-colored hybrid ("W2"), "no cost", and computed properties (total CMC, generic_mana, colorless_mana, per-color counts, color count, has_x, has_hybrid, has_phyrexian). Verify that colorless mana contributes to CMC but adds no color
- [x] T008 [P] Write tests for feature engineering in `tests/unit/application/test_feature_engineering.py` — cover Card→feature vector transformation: mana value extraction, color multi-hot encoding (W/U/B/R/G), generic mana count, colorless mana count (distinct from generic), type multi-hot encoding, keyword multi-hot (top-30), oracle text TF-IDF, power/toughness numeric parsing with NaN indicator, ability count, layout one-hot. Test with fixture cards from `tests/fixtures/forge_cards/` including a card with colorless mana cost
- [x] T009 [P] Write tests for model store in `tests/unit/infrastructure/test_model_store.py` — cover save model artifact (joblib), load model by path, load latest model, handle missing model file, and model versioning (timestamp-based naming)

### Implementation for Foundational Phase

- [x] T010 [P] Implement Card, PriceEstimate, TrainingExample, TrainedModel, and EvaluationMetrics dataclasses in `src/price_predictor/domain/entities.py` — pure Python stdlib only (dataclasses, typing). Include validation methods per data-model.md rules
- [x] T011 [P] Implement ManaCost value object in `src/price_predictor/domain/value_objects.py` — parse Forge mana format (space-separated shards), extract: total_mana_value, generic_mana, colorless_mana (count of {C} pips — distinct from generic), per-color counts (w, u, b, r, g), color_count, has_x, has_hybrid, has_phyrexian. Handle "no cost" as None. Handle snow ("S") and two-or-colored hybrid ("W2", "U2", etc.). Pure Python stdlib only
- [x] T012 Implement feature engineering in `src/price_predictor/application/feature_engineering.py` — transform Card→numeric feature vector per research.md R5 feature table (17 feature groups including colorless mana as feature 4a). Use scikit-learn TfidfVectorizer for oracle text (top 500 terms). Include a FeatureEngineering class that fits on training data and transforms new cards consistently. Fixed random seed for reproducibility
- [x] T013 [P] Implement model store in `src/price_predictor/infrastructure/model_store.py` — save/load sklearn model + fitted FeatureEngineering as a single joblib artifact. Model version = ISO timestamp. Support saving to `models/{version}.joblib` and symlinking/copying as `models/latest.joblib`. Handle file-not-found gracefully

**Checkpoint**: Foundation ready — all domain entities, value objects, feature engineering, and model persistence verified by tests. User story implementation can now begin.

---

## Phase 3: User Story 1 — Predict Price from Card Attributes (Priority: P1)

**Goal**: Given card attributes (mana cost, types, oracle text, P/T, keywords), return an EUR price estimate. Works for real and made-up cards.

**Independent Test**: Submit a known card's attributes with a trained model loaded, verify a positive EUR float is returned.

### Tests for User Story 1 (MANDATORY per Constitution)

- [x] T014 [P] [US1] Write tests for PredictPriceUseCase in `tests/unit/application/test_predict.py` — cover: predict with complete attributes returns positive EUR float, predict with partial attributes (missing oracle text, missing P/T) still works, predict with made-up card attributes works, same input + same model = same output (FR-005), validation errors for invalid input (no types), model-not-found error
- [x] T015 [P] [US1] Write tests for predict CLI subcommand in `tests/unit/infrastructure/test_cli_predict.py` — cover: valid args produce JSON stdout with `predicted_price_eur` and `model_version`, missing --types produces exit code 1 and stderr error, missing model file produces exit code 2, partial args (only --types and --mana-cost) succeed

### Implementation for User Story 1

- [x] T016 [US1] Implement PredictPriceUseCase in `src/price_predictor/application/predict.py` — accept Card attributes (or a Card entity), load trained model via model store, transform card through feature engineering, run sklearn predict, exp-transform log-price back to EUR, return PriceEstimate. Validate input at boundary (FR-006). Ensure deterministic output (FR-005)
- [x] T017 [US1] Implement `predict` CLI subcommand in `src/price_predictor/infrastructure/cli.py` — argparse subparser with all options per contracts/cli.md (--mana-cost, --types, --supertypes, --subtypes, --oracle-text, --keywords, --power, --toughness, --loyalty, --colors, --model-path). Parse comma-separated lists. Build Card entity from args. Call PredictPriceUseCase. Output JSON to stdout. Errors to stderr with exit codes 1/2
- [x] T018 [US1] Implement `__main__.py` entry point in `src/price_predictor/__main__.py` — wire up argparse with `predict` subcommand (train and evaluate added in later phases). Enable `python -m price_predictor predict ...`

**Checkpoint**: `python -m price_predictor predict --types Creature --mana-cost "1 G" --power 2 --toughness 2` returns a JSON price estimate (requires a pre-trained model in `models/`). User Story 1 is independently testable with a fixture model.

---

## Phase 4: User Story 2 — Train Model on Historical Card Data (Priority: P2)

**Goal**: Parse Forge card scripts, join with Cardmarket EUR prices via MTGJSON, train a GradientBoostingRegressor, save the model artifact.

**Independent Test**: Point at Forge cardsfolder + MTGJSON files, run train, verify a `.joblib` model is produced and a JSON summary reports card counts.

### Tests for User Story 2 (MANDATORY per Constitution)

- [x] T019 [P] [US2] Write tests for Forge card script parser in `tests/unit/infrastructure/test_forge_parser.py` — cover: parse vanilla creature (Grizzly Bears), parse legendary creature with keywords and abilities (Ragavan), parse planeswalker with loyalty (Jace), parse instant (Lightning Bolt), parse land ("no cost"), parse card with colorless mana cost (e.g., Matter Reshaper with "2 C"), parse transform card (Delver of Secrets — front face only), parse split card (Fire // Ice — front face only), handle malformed file gracefully (report error, don't crash), extract ability count from A:/T:/S: lines, extract keywords from K: lines (strip parameters)
- [x] T020 [P] [US2] Write tests for MTGJSON loader in `tests/unit/infrastructure/test_mtgjson_loader.py` — cover: build name→UUIDs mapping from AllPrintings sample (filter paper-available, English, non-funny, non-online-only), extract Cardmarket EUR price from AllPricesToday sample (retail.normal and retail.foil), cheapest-price-across-printings algorithm (min of normal + foil across all UUIDs for a name), handle missing price data (return None), handle missing cardmarket section
- [x] T021 [P] [US2] Write tests for TrainModelUseCase in `tests/unit/application/test_train.py` — cover: train on fixture data produces a TrainedModel with valid metadata, skipped cards are reported with reasons (no_price, parse_error, no_printings_match) per FR-007, model is saved via model store, train/test split uses fixed seed for reproducibility, training with too few valid cards raises error (exit code 2)

### Implementation for User Story 2

- [x] T022 [US2] Implement Forge card script parser in `src/price_predictor/infrastructure/forge_parser.py` — walk cardsfolder directory tree, read each `.txt` file, split on first `:` per line for key-value pairs, handle ALTERNATE sections (extract front face only), extract: Name, ManaCost→ManaCost value object (handles colored W/U/B/R/G, generic 1-N, colorless C, hybrid WU, phyrexian WP, snow S, two-or-colored W2, X costs, and "no cost"), Types→(supertypes, types, subtypes) using known type lists from research.md R7, PT→power/toughness, Loyalty, K: lines→keywords (strip parameters after first `:`), Oracle text, count A:/T:/S: lines for ability_count, determine layout from AlternateMode. Return list of Card entities. Log parse errors with filename (FR-007)
- [x] T023 [US2] Implement MTGJSON loader in `src/price_predictor/infrastructure/mtgjson_loader.py` — two functions: (1) `build_name_to_uuids(allprintings_path)`: stream AllPrintings.json, iterate sets→cards, filter per research.md R6, build dict[str, list[str]] name→uuids. (2) `get_cheapest_price(allprices_path, uuids)`: for each UUID, check `data[uuid].paper.cardmarket.retail.{normal,foil}`, take most recent date's price, return min across all UUIDs and finishes. Use `ijson` or chunked reading for the large 512MB AllPrintings file if pandas is too slow
- [x] T024 [US2] Implement TrainModelUseCase in `src/price_predictor/application/train.py` — orchestrate: (1) parse Forge cards via forge_parser, (2) build name→UUIDs via mtgjson_loader, (3) join cards to cheapest prices, (4) filter to cards with valid prices (FR-004, FR-004a), (5) log-transform prices, (6) fit FeatureEngineering on training set, (7) train GradientBoostingRegressor, (8) save model+feature_eng via model_store, (9) return TrainedModel metadata + skip report JSON. Use 80/20 split with seed=42
- [x] T025 [US2] Implement `train` CLI subcommand in `src/price_predictor/infrastructure/cli.py` — add `train` subparser with options per contracts/cli.md (--forge-cards-path, --prices-path, --printings-path, --output-path, --test-split, --random-seed). Call TrainModelUseCase. Output JSON summary to stdout. Errors to stderr with exit codes 1/2
- [x] T026 [US2] Wire `train` subcommand into `src/price_predictor/__main__.py` entry point

**Checkpoint**: `python -m price_predictor train` parses Forge cards, joins prices, trains model, saves to `models/`. JSON summary shows cards_used, cards_skipped, and price_range_eur. Model can now be loaded by US1's predict command.

---

## Phase 5: User Story 3 — Evaluate Model Accuracy (Priority: P3)

**Goal**: Given a trained model and test data, compute accuracy metrics (MAE, median % error, top-20% overlap) and optionally output per-card results.

**Independent Test**: Load a trained model, run evaluate on fixture data, verify metrics JSON is returned with all required fields.

### Tests for User Story 3 (MANDATORY per Constitution)

- [x] T027 [P] [US3] Write tests for EvaluateModelUseCase in `tests/unit/application/test_evaluate.py` — cover: evaluate on fixture data returns EvaluationMetrics with all fields (mean_absolute_error_eur, median_percentage_error, top_20_overlap, sample_count), per-card breakdown includes predicted vs actual for each card, same seed + same model = same metrics (reproducible), handle model-not-found error

### Implementation for User Story 3

- [x] T028 [US3] Implement EvaluateModelUseCase in `src/price_predictor/application/evaluate.py` — load model, re-derive test split from data (same seed as training to get held-out set), predict on test cards, compute: mean absolute error (EUR), median percentage error, top-20% overlap (fraction of actual top-20% cards that appear in predicted top-20%), sample count. Optionally output per-card CSV (card name, actual price, predicted price, absolute error)
- [x] T029 [US3] Implement `evaluate` CLI subcommand in `src/price_predictor/infrastructure/cli.py` — add `evaluate` subparser with options per contracts/cli.md (--model-path, --forge-cards-path, --prices-path, --printings-path, --test-split, --random-seed, --output-csv). Call EvaluateModelUseCase. Output JSON metrics to stdout. Errors to stderr with exit codes 1/2
- [x] T030 [US3] Wire `evaluate` subcommand into `src/price_predictor/__main__.py` entry point

**Checkpoint**: `python -m price_predictor evaluate` loads model, computes metrics, outputs JSON. All three CLI subcommands are now functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Integration testing, end-to-end validation, documentation, cleanup

- [x] T031 Write end-to-end integration test in `tests/integration/test_end_to_end.py` — full pipeline: train on fixture data → predict a known card → evaluate on fixture test set. Verify: model file is created, prediction returns positive EUR, evaluation metrics are within expected ranges, predictions are reproducible across runs
- [x] T032 Run quickstart.md validation — verify all commands in `specs/001-card-price-predictor/quickstart.md` work as documented (train, predict, evaluate, pytest)
- [x] T033 [P] Add `.gitignore` entries for `models/`, `*.joblib`, `.venv/`, `__pycache__/`, `*.egg-info/`
- [x] T034 Create `README.md` at project root per Constitution Article VI — must include: (1) how to install dependencies and set up the environment, (2) how to launch each executable (`train`, `predict`, `evaluate`), (3) textual description of each workflow covering inputs, processing steps, and outputs, (4) description of the ML approach chosen (Gradient Boosted Trees on log-transformed EUR prices) with rationale for why it was selected over alternatives (Random Forest, linear regression, neural networks — see research.md R5), (5) description of the feature engineering pipeline (17 feature groups from card attributes — see research.md R5 feature table), (6) description of all artifacts produced (trained model `.joblib` files, evaluation JSON output, prediction JSON output, per-card CSV), (7) how to run tests (`pytest` for fast suite, `pytest tests/integration/` for integration)
- [x] T035 [P] Verify documentation quality gate — confirm `README.md` covers all workflows, CLI commands, artifacts, and ML processes as required by Constitution Article VI quality gate. Ensure documentation is consistent with contracts/cli.md and quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2)
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2)
- **User Story 3 (Phase 5)**: Depends on Foundational (Phase 2)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2. Needs a pre-trained fixture model for testing (created in conftest.py T005). No dependency on US2 for testing, but needs US2 for production use with real data.
- **User Story 2 (P2)**: Can start after Phase 2. Independent of US1. Produces the trained model that US1 consumes in production.
- **User Story 3 (P3)**: Can start after Phase 2. Independent of US1/US2 for unit testing. Needs US2's data pipeline in production (reuses Forge parser + MTGJSON loader).

### Recommended Execution Order (Solo Developer)

1. Phase 1 (Setup) → Phase 2 (Foundational)
2. Phase 4 (US2 — Train) — build the data pipeline first since it produces the model US1 needs
3. Phase 3 (US1 — Predict) — now has both fixture model (tests) and real model (from US2)
4. Phase 5 (US3 — Evaluate)
5. Phase 6 (Polish)

### Within Each User Story

- Tests MUST be written first and FAIL before implementation
- Infrastructure modules (parser, loader) before use cases
- Use cases before CLI commands
- CLI wiring last

### Parallel Opportunities

- T003 and T004 can run in parallel (Setup phase)
- T006, T007, T008, T009 can all run in parallel (Foundational tests)
- T010, T011, T013 can run in parallel (Foundational implementation, different files)
- T014 and T015 can run in parallel (US1 tests)
- T019, T020, T021 can all run in parallel (US2 tests)
- T034 and T035 can run in parallel (Polish documentation tasks)
- User stories 1, 2, 3 can run in parallel after Phase 2 (if staffed)

---

## Implementation Strategy

### MVP First (US2 → US1)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 4: User Story 2 (Train) — builds data pipeline and produces model
4. Complete Phase 3: User Story 1 (Predict) — core value proposition, now has a trained model
5. **STOP and VALIDATE**: Train a model on real data, predict a few known cards, verify EUR prices are reasonable

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US2 (Train) → Can produce models from Forge + MTGJSON data
3. Add US1 (Predict) → Can predict card prices (MVP!)
4. Add US3 (Evaluate) → Can measure model accuracy
5. Polish → End-to-end validation + documentation (Article VI)

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently testable via unit tests with fixtures
- All prices are EUR (Cardmarket) — no USD fields anywhere
- Rarity is deliberately excluded from all entities and features
- Colorless mana ({C}) is tracked separately from generic mana ({1}–{N}) in ManaCost and feature engineering
- Forge card scripts are the card data source, not AllIdentifiers.json
- AllPrintings.json provides name→UUID mapping only
- AllPricesToday.json provides Cardmarket EUR prices only
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
