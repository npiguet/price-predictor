# `price_predictor` — the price prediction system

Entry point: `python -m price_predictor <subcommand>` (see `src/price_predictor/__main__.py` and `infrastructure/cli.py`). The terms this file uses are defined in `GLOSSARY.md` beside it.

Besides its two price models, the package owns what every other package reads cards through: the converted card format, the tokenizer and the vocabulary. `sealed`, `draft` and `effects` import from it; it imports from none of them.

Subcommands:
- **`convert`** — runs the Java `ConvertMain` to turn Forge card scripts (`../forge/forge-gui/res/cardsfolder/`) into converted cards under `output/cardsfolder/`, one file per card: header lines, then ability lines, one block per face. Forge's token scripts (`../forge/forge-gui/res/tokenscripts/`) are converted by the same run into `output/tokenscripts/`. Beside each converted card it writes a `<name>.provenance.json` provenance sidecar, which the effects package reads; the format is in `src/effects/CLAUDE.md` § File formats.
- **`check-convert`** — the conversion check: scores each converted card's converted ability text against the Oracle text in its Forge card script by word-bag similarity, and lists the cards below `--threshold`.
- **`vocabulary`** — builds the vocabulary from the converted-card corpus into `models/price-predictor/transformer/vocab.txt`. **Required before training or running the price transformer**, and read by every package that tokenizes converted cards. Also seeds set-code fragments from `AllPrintings.json` when available.
- **`train sklearn`** — trains the sklearn model (a `GradientBoostingRegressor`) on the log of the card price over eight feature groups: mana cost, card types, supertypes, keywords, power and toughness, layout, printing data, and TF-IDF word frequencies of the converted ability text. Saves to `models/price-predictor/sklearn/{timestamp}.joblib` + `latest.joblib`.
- **`train transformer`** — trains the price transformer (tunable `--d-model/--n-layers/--n-heads/--ff-dim`) on tokenized converted cards, with printing data beside the tokens. It targets the shifted-log price (`--log-offset`), oversamples by sampling bucket (`--sampler-exponent`), and keeps the epoch with the lowest loss on the held-out split. Saves to `models/price-predictor/transformer/`.
- **`predict {sklearn,transformer}`** — reads a converted card (from `--file` or inline `--card`) and prints JSON with `predicted_price_eur` + `model_version`. Looks up the card's printing data by canonical name. For a hypothetical card, the price transformer uses default printing data, while the sklearn model gets no printing data at all, so its printing-data columns are zero.
- **`evaluate {sklearn,transformer}`** — scores a model on the held-out split: mean absolute error, median percentage error, median log error and top-20 overlap. The transformer adds a breakdown by reporting bucket, with the signed log error.
- **`serve`** — FastAPI + uvicorn REST service exposing `POST /api/v1/predict`. Loads the sklearn model (required), the price transformer (optional, graceful degradation), the tokenizer and the MTGJSON printing data. Accepts a converted card as `text/plain` and returns a prediction from every model it loaded.

The two price models have **different input contracts**:
- the sklearn model takes a parsed `Card` and runs it through `FeatureEngineering` to produce its feature vector;
- the price transformer takes the converted card's text, tokenizes it with the vocabulary, and receives `PrintingData` beside the tokens, never inside the text.

Prices are learned in log space and turned back into euros at inference, because the skew of the euro price distribution would otherwise let a handful of expensive cards dominate the loss. The sklearn model uses the plain log of the card price; the price transformer uses the shifted-log price.

The price transformer's price encoder is the alternate card encoder the sealed pipeline accepts through `--encoder-checkpoint`.

## Shared infrastructure

Other packages also import this package's general-purpose infrastructure: `forge_jvm` (classpath assembly, JVM launch, the Forge worker pool), `torch_checkpoint`, `torch_training`, `append_only` (readers that tolerate a trailing partial line), `ridge_probes` (the linear-probe harness over card embeddings) and `dataclass_defaults.field_default`.

## Inputs on disk

- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps, the source of card prices and printing data.
- `../forge/forge-gui/res/cardsfolder/` — Forge card scripts (source for `convert`).
- `../forge/forge-gui/res/tokenscripts/` — Forge token scripts, converted to `output/tokenscripts/` by the same `convert` run. Kept out of the card tree because converted token and card filenames collide and the sealed pipeline reads that tree as its card corpus.
- `output/cardsfolder/` — converted cards, each paired with a `.npz` after `sealed encode-cards` runs and a `.provenance.json` written by `convert`.
