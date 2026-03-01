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
- 004-cardmarket-eur-pricing: Added Python 3.14+ + No new dependencies (existing: scikit-learn, pandas, numpy, joblib)
- 003-cheapest-printing-price: Added Python 3.14+ + scikit-learn, pandas, numpy, joblib (no new dependencies)
- 002-forge-api-integration: Added Python 3.14+ (service), Java 17+ (connector) + FastAPI, uvicorn (Python); no external deps (Java — uses java.net.http)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
