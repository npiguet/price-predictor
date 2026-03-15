# Research: 008 Model Harmonization

## R1: Converted Card Text Parser for sklearn

**Decision**: Create a new `converted_card_parser.py` in `infrastructure/` that reads the converted text format and produces `Card` entities.

**Rationale**: The sklearn pipeline requires structured `Card` entities for `FeatureEngineering.transform()` which extracts 77+ dense features (mana cost breakdown, type multi-hot, keywords, P/T, etc.) plus TF-IDF. The transformer pipeline does not need this — it passes raw text directly to the BERT tokenizer. A dedicated parser keeps the two pipelines independent while sharing the same input format.

**Alternatives considered**:
- Reuse `forge_parser.py` — rejected because the converted format has different syntax (lowercase `name:` lines vs uppercase `Name:` Forge format). Extending forge_parser would conflate two distinct formats.
- Make sklearn work on raw text — rejected because it would mean replacing the entire feature engineering pipeline (GradientBoosting needs structured numeric features, not text).

**Key mapping** (converted text field → Card entity field):
| Converted line | Card field | Notes |
|---|---|---|
| `name:` | `name` | Direct |
| `mana cost:` | `mana_cost` | Parse via `ManaCost.parse()` |
| `types:` | `types`, `supertypes`, `subtypes` | Split using existing `_classify_types()` logic |
| `power toughness:` | `power`, `toughness` | Split on `/` |
| `loyalty:` | `loyalty` | Direct |
| `colors:` | (not currently in Card) | Ignored — derived from mana cost |
| Ability lines (`spell[N]:`, `triggered:`, `activated:`, etc.) | `oracle_text`, `keywords`, `ability_count` | Concatenate for oracle_text; count for ability_count |

## R2: sklearn Training Input Change

**Decision**: Replace `parse_forge_cards(forge_cards_path)` call in `train.py` with a new function that reads converted card text files from `./output/` and produces `Card` entities + matches them to prices.

**Rationale**: The current sklearn training pipeline reads Forge scripts → Card → join prices → features → train. The only change is the input parser; the rest of the pipeline (FeatureEngineering, GradientBoosting, model_store) stays the same.

**Alternatives considered**:
- Keep Forge scripts for sklearn, converted text only for transformer — rejected because FR-002 requires a single canonical input format.

## R3: Unified CLI Architecture

**Decision**: Replace the flat subcommand structure with hierarchical subcommands: `train {model}`, `predict {model}`, `evaluate {model}`. Use argparse subparsers two levels deep (command → model).

**Rationale**: argparse supports nested subparsers natively. Each model registers its specific optional arguments under its sub-subparser, while shared arguments (like `--output` paths) live on the parent command parser.

**Alternatives considered**:
- Single level with `--model` flag — rejected because positional model arg reads more naturally (`train sklearn` vs `train --model sklearn`) and matches the spec.
- Click or Typer — rejected because the project already uses argparse; adding a new dependency violates Simplicity First.

## R4: predict Command — Local Execution

**Decision**: The `predict` command calls the model's inference logic directly (same code paths as the REST server) without HTTP. For sklearn: load model → parse converted text → FeatureEngineering.transform → model.predict → exp(). For transformer: load model → tokenize raw text → model forward → exp() - 2.

**Rationale**: The inference logic already exists in `predict.py` (sklearn) and inline in `server.py` (transformer). Extracting the transformer inference into a reusable function in `application/` keeps it callable from both CLI and server.

**Alternatives considered**:
- Keep calling REST service — rejected because the spec explicitly requires local execution (FR-008).

## R5: REST API Input Format Change

**Decision**: The endpoint is renamed from `/api/v1/evaluate` to `/api/v1/predict` for consistency with the CLI `predict` command. It switches from parsing Forge script text via `parse_forge_text()` to: (1) passing raw text to transformer, and (2) parsing via the new `parse_converted_text()` for sklearn features. The response format stays the same (both model predictions).

**Rationale**: Minimal change — just swap the parser call and adjust tokenizer input. The response contract is preserved.

## R6: Model Artifact Directory Structure and Versioning

**Decision**: sklearn artifacts move from `models/` (root) to `models/sklearn/`. Transformer stays at `models/transformer/` (already there). `model_store.py` and `transformer_store.py` get updated default paths.

Additionally, `transformer_store.py` must adopt the same versioning convention as `model_store.py`: timestamped filenames (`<version>.pt`) plus a `latest.pt` copy. Currently the transformer store always overwrites a single `model.pt` with no versioning.

**Rationale**: Consistent `models/<model_id>/` structure per FR-003/FR-004. Consistent versioning lets users keep previous model versions and roll back, regardless of which model type they're working with.

**Alternatives considered**:
- Keep transformer as single `model.pt` — rejected because it loses previous versions on retrain, inconsistent with sklearn convention.

## R7: Standardized Evaluation Metrics

**Decision**: Both sklearn and transformer evaluations return the same set of metrics: `model_version`, `mean_absolute_error_eur`, `median_percentage_error`, `median_abs_error_log`, `top_20_overlap`, `sample_count`.

**Rationale**: Previously sklearn reported `median_percentage_error` and `top_20_overlap` while transformer reported `median_abs_error_log`. Users comparing models need identical metrics. The union of all fields is small and cheap to compute, so both models now report all of them.

**Changes**:
- Transformer gains: `median_percentage_error`, `top_20_overlap`, `model_version` (replacing `model_path`)
- Sklearn gains: `median_abs_error_log`
