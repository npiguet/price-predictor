# Quickstart: Forge API Integration

## Prerequisites

- Python 3.14+ with pip
- Java 17+ with Maven (for the connector)
- A trained model (run `python -m price_predictor train` first —
  see `specs/001-card-price-predictor/quickstart.md`)

## Setup (Python service)

```bash
# Install new dependencies (adds fastapi, uvicorn)
cd price-predictor
pip install -e ".[dev]"
```

## Start the prediction service

```bash
python -m price_predictor serve
```

You should see:
```
Loading model from models/latest.joblib...
Prediction service started on http://0.0.0.0:8000
```

Optional flags:
```bash
python -m price_predictor serve --port 9000 --model-path models/20260301-143000.joblib
```

## Test the endpoint manually

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: text/plain" \
  -d "Name:Lightning Bolt
ManaCost:R
Types:Instant
Oracle:Lightning Bolt deals 3 damage to any target."
```

Expected response:
```json
{"predicted_price_eur": 2.35, "model_version": "20260301-143000"}
```

## Build the Java connector

```bash
cd forge-connector
mvn package
```

The JAR is at `forge-connector/target/forge-connector-1.0.0-SNAPSHOT.jar`.

## Use the connector from Java

```java
import com.pricepredictor.connector.*;

var client = new PricePredictorClient();
var estimate = client.predict(CardAttributes.builder()
    .types("Creature")
    .manaCost("1 G G")
    .power("2").toughness("2")
    .build());
System.out.println(estimate.predictedPriceEur());
```

## Run tests

```bash
# Python tests (from repo root)
cd src; pytest; cd ..

# Java tests
cd forge-connector
mvn test
```

## Validation checklist

- [ ] `python -m price_predictor serve` starts without errors
- [ ] Service fails fast with error if no trained model exists
- [ ] `curl` to `/api/v1/evaluate` returns a JSON price estimate
- [ ] Malformed input returns a 400 error with descriptive message
- [ ] `mvn package` in forge-connector/ produces a JAR under 1 MB
- [ ] Java connector `predict()` returns a price estimate
- [ ] Java connector throws `ServiceUnavailableException` when service is down
- [ ] Connector returns error within 5 seconds when service is unreachable
- [ ] `pytest` passes all tests (existing + new)
- [ ] `mvn test` passes all connector tests
