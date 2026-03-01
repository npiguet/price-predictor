# CLI Contract: Serve Command

**Date**: 2026-03-01
**Feature**: 002-forge-api-integration
**Extends**: `specs/001-card-price-predictor/contracts/cli.md`

## Overview

This feature adds a `serve` subcommand to the existing CLI. The
command starts the prediction service (REST API) and keeps it
running until interrupted. All existing commands (`train`, `predict`,
`evaluate`) are unchanged.

## New Command

### `serve` — Start the prediction service

**Usage**:
```
python -m price_predictor serve [OPTIONS]
```

**Input options**:
```
--model-path PATH     Path to trained model
                      (default: models/latest.joblib)
--host TEXT           Bind address (default: 0.0.0.0)
--port INT            Listen port (default: 8000)
```

**Behavior**:
1. Loads the trained model from `--model-path`
2. If model file not found → prints error to stderr, exits with
   code 2
3. Starts uvicorn HTTP server on `--host`:`--port`
4. Logs startup message to stderr: `Prediction service started on
   http://{host}:{port}`
5. Serves requests until interrupted (Ctrl+C)

**Output** (stderr only — no stdout JSON):
```
Loading model from models/latest.joblib...
Prediction service started on http://0.0.0.0:8000
```

**Error output** (stderr):
```
Error: Model file not found at models/latest.joblib
```

**Exit codes**:
- 0: Clean shutdown (Ctrl+C)
- 2: Model file not found or not loadable

**Signals**:
- SIGINT / Ctrl+C: Graceful shutdown (uvicorn handles this natively)
