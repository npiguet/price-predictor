# price-predictor Development Guidelines

## Stack

- **Python**: 3.14+, managed with venv + pip (`python` / `pip`)
- **ML**: PyTorch, numpy, scikit-learn, pandas, joblib, ijson
- **API**: FastAPI, uvicorn
- **Java**: 17+ (forge-connector module), forge-game 2.0.10-SNAPSHOT
- **Custom tokenizer**: `MtgTokenizer` (spec 010), produces 512-dim card embeddings stored as `.npz`

## Project Structure

```
src/
tests/
specs/
```

## Commands

```bash
cd src
pytest
ruff check .
```

## Code Style

Python 3.14+: follow standard conventions.
