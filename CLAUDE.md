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
- 006-card-script-parsing: Added Java 17+ (converter module), Python 3.14+ (CLI integration) + forge-game 2.0.10-SNAPSHOT (transitively includes forge-core), JUnit 5
- 006-card-script-parsing: Added Python 3.14+ (existing project stack) + None new — pure text parsing with stdlib only
- 005-card-eval-endpoints: Added Python 3.14+ + FastAPI, uvicorn (existing) — no new dependencies added

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
