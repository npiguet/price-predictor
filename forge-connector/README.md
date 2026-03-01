# Forge Price Predictor Connector

Lightweight Java 17+ library that lets MTG Forge get card price predictions
from the price prediction service. Zero external dependencies.

## Maven Coordinates

```xml
<dependency>
    <groupId>com.pricepredictor</groupId>
    <artifactId>forge-connector</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

## Usage (5 lines)

```java
var client = new PricePredictorClient();
var estimate = client.predict(CardAttributes.builder()
    .types("Creature")
    .manaCost("1 G G")
    .power("2").toughness("2")
    .build());
System.out.println(estimate.predictedPriceEur()); // e.g. 0.15
```

## Build

```bash
cd forge-connector
mvn package
```

The JAR is at `target/forge-connector-1.0.0-SNAPSHOT.jar`.

## Requirements

- Java 17+
- The price prediction service must be running (`python -m price_predictor serve`)

## API

### PricePredictorClient

```java
// Default: http://localhost:8000, 5s timeout
new PricePredictorClient();

// Custom endpoint
new PricePredictorClient("http://myserver:9000");

// Custom endpoint + timeout (ms)
new PricePredictorClient("http://myserver:9000", 3000);

// Predict
PriceEstimate estimate = client.predict(cardAttributes);
```

### CardAttributes (builder)

```java
CardAttributes card = CardAttributes.builder()
    .name("Lightning Bolt")           // optional
    .manaCost("R")                    // optional
    .types("Instant")                 // required (at least one)
    .supertypes("Legendary")          // optional
    .subtypes("Human", "Wizard")      // optional
    .oracleText("Deals 3 damage.")    // optional
    .keywords("Flying", "Haste")      // optional
    .power("3").toughness("4")        // optional
    .loyalty("3")                     // optional
    .build();
```

### PriceEstimate (record)

```java
estimate.predictedPriceEur()  // double
estimate.modelVersion()       // String
```

## Error Handling

| Scenario | Exception | Message pattern |
|----------|-----------|-----------------|
| Service not running | `ServiceUnavailableException` | "Connection refused: {url}" |
| Request timeout (default 5s) | `ServiceUnavailableException` | "Request timed out after {ms}ms" |
| Server returns 400/500 | `InvalidResponseException` | Server's error message |
| Unparseable response | `InvalidResponseException` | "Failed to parse response: {detail}" |

All exceptions extend `PricePredictorException` (checked).

## Thread Safety

`PricePredictorClient` is thread-safe and reusable across requests. Each
`predict()` call creates a fresh HTTP request. The client automatically
recovers when the service restarts.
