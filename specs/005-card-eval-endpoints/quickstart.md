# Quick Start: Card Evaluation Endpoints

**Feature**: 005-card-eval-endpoints

## Prerequisites

- Python 3.14+ with project dependencies installed
- A trained model at `models/latest.joblib` (run `python -m price_predictor train` first)

## 1. Start the Prediction Service

```bash
cd src
python -m price_predictor serve
```

The service starts on `http://localhost:8000` by default.

## 2. Evaluate a Card via the Endpoint

Send a Forge card script directly:

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: text/plain" \
  -d "Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target."
```

Response:
```json
{
  "predicted_price_eur": 2.35,
  "model_version": "latest"
}
```

## 3. Evaluate a Card via the CLI Tool

Given a Forge card script file (e.g., `card.txt`):

```bash
cd src
python -m price_predictor eval card.txt
```

Output:
```
Predicted price: €2.35
Model version:   latest
```

### Custom Endpoint

If the service runs on a different host/port:

```bash
python -m price_predictor eval card.txt --endpoint http://myhost:9000/api/v1/evaluate
```

## 4. Error Handling

**File not found**:
```bash
python -m price_predictor eval nonexistent.txt
# Error: File not found: nonexistent.txt
```

**Service not running**:
```bash
python -m price_predictor eval card.txt
# Error: Could not connect to prediction service at http://localhost:8000/api/v1/evaluate
```

**Invalid card script**:
```bash
echo "not a card" > bad.txt
python -m price_predictor eval bad.txt
# Error: Prediction service returned error (400): Failed to parse card script: No Types field found in card script
```

## 5. Structured Request Logging

Every request to the endpoint is logged to stderr in JSON format:

```json
{"event": "evaluate_request", "timestamp": "2026-03-01T14:30:00.123456", "status_code": 200, "latency_ms": 42.5, "card_name": "Lightning Bolt", "card_types": ["Instant"], "card_mana_cost": "R", "predicted_price_eur": 2.35, "model_version": "latest"}
```

## Running Tests

```bash
cd src
pytest                           # All unit tests
pytest -m integration            # Integration tests (requires trained model fixture)
```
