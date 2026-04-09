# price-predictor Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-01

## Active Technologies
- Local JSON files (resources/), Forge card scripts (001-card-price-predictor)
- Python 3.14+ + scikit-learn, pandas, numpy, joblib, ijson (001-card-price-predictor)
- Local JSON files (resources/), joblib model files (models/) (001-card-price-predictor)
- Python 3.14+ (service), Java 17+ (connector) + FastAPI, uvicorn (Python); no external deps (Java — uses java.net.http) (002-forge-api-integration)
- Same as feature 001 — joblib model files in `models/` (002-forge-api-integration)
- Python 3.14+ + scikit-learn, pandas, numpy, joblib (no new dependencies) (003-cheapest-printing-price)
- Local JSON files (AllPrintings.json, AllPricesToday.json), joblib model files (003-cheapest-printing-price)
- Python 3.14+ + No new dependencies (existing: scikit-learn, pandas, numpy, joblib) (004-cardmarket-eur-pricing)
- Python 3.14+ + FastAPI, uvicorn (existing) — no new dependencies added (005-card-eval-endpoints)
- N/A (no storage changes) (005-card-eval-endpoints)
- Python 3.14+ (existing project stack) + None new — pure text parsing with stdlib only (006-card-script-parsing)
- Local text files (input: Forge card scripts at `../forge/forge-gui/res/cardsfolder/`, output: `./output/`) (006-card-script-parsing)
- Java 17+ (converter module), Python 3.14+ (CLI integration) + forge-game 2.0.10-SNAPSHOT (transitively includes forge-core), JUnit 5 (006-card-script-parsing)
- Local text files (input: Forge card scripts, output: converted text files in `./output/`) (006-card-script-parsing)
- Python 3.14+ + PyTorch (`torch`), Hugging Face `transformers` (BERT tokenizer), existing: scikit-learn, FastAPI, uvicorn, numpy, pandas, joblib, ijson (007-transformer-model-arch)
- `.pt` model artifact in `models/transformer/`, existing `.joblib` models in `models/` (007-transformer-model-arch)
- Python 3.14+ + scikit-learn, pandas, numpy, joblib, PyTorch, transformers (BERT tokenizer), FastAPI, uvicorn (008-model-harmonization)
- `.joblib` model files in `models/sklearn/`, `.pt` model files in `models/transformer/` (008-model-harmonization)
- Python 3.14+ + scikit-learn, pandas, numpy, joblib, PyTorch, transformers (BERT tokenizer), FastAPI, uvicorn, ijson (009-printing-metadata)
- Local JSON files (AllPrintings.json, AllPricesToday.json), joblib model files (models/sklearn/), .pt model files (models/transformer/) (009-printing-metadata)
- Python 3.14+ + Standard library only — no new Python packages required (010-mtg-custom-tokenizer)
- `models/transformer/vocab.txt` (plain text, UTF-8, one token per line) (010-mtg-custom-tokenizer)
- Python 3.14+ (sealed module), Java 17+ (forge-connector extension) + PyTorch + existing MtgTokenizer (Python); forge-game 2.0.10-SNAPSHOT (Java — already in pom.xml) (011-sealed-dataset)
- `.npz` embedding files in cards-path folder; `pools.txt` flat text file in output/sealed/pools/{set}/ (011-sealed-dataset)
- Python 3.14+ (supervisor), Java 17+ (forge-connector worker) + Python stdlib only (subprocess, signal, time, pathlib); Java: forge-game 2.0.10-SNAPSHOT (already in forge-connector pom.xml) (012-sealed-training-data)
- Append-only flat text file at `./output/sealed/match-outcomes.txt` (012-sealed-training-data)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.14+: Follow standard conventions

## Recent Changes
- 012-sealed-training-data: Added Python 3.14+ (supervisor), Java 17+ (forge-connector worker) + Python stdlib only (subprocess, signal, time, pathlib); Java: forge-game 2.0.10-SNAPSHOT (already in forge-connector pom.xml)
- 011-sealed-dataset: Added Python 3.14+ (sealed module), Java 17+ (forge-connector extension) + PyTorch + existing MtgTokenizer (Python); forge-game 2.0.10-SNAPSHOT (Java — already in pom.xml)
- 010-mtg-custom-tokenizer: Added Python 3.14+ + Standard library only — no new Python packages required

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
