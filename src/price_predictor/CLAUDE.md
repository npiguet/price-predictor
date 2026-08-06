# `price_predictor` — the price prediction system


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

The two model types have **different input contracts**:
- `sklearn` takes a parsed `Card` object and runs it through `FeatureEngineering` to produce a numeric vector.
- `transformer` takes raw converted card text, tokenizes it with the custom vocab, and receives `PrintingData` as a side-channel (not embedded in the text).

Prices are trained on `log(price + offset)` and exp-transformed back on inference — the skew of the EUR distribution would otherwise let a handful of expensive cards dominate the loss.
