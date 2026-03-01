# Java Connector API Contract

**Date**: 2026-03-01
**Feature**: 002-forge-api-integration

## Overview

The connector is a Java 17+ library that Forge (or any Java
application) can use to get card price predictions from the
prediction service. It handles HTTP communication, Forge script
serialization, and error handling internally.

## Maven Coordinates

```xml
<dependency>
    <groupId>com.pricepredictor</groupId>
    <artifactId>forge-connector</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Public API

### PricePredictorClient

Main entry point. Thread-safe, reusable across requests.

```java
package com.pricepredictor.connector;

public class PricePredictorClient {

    // Default endpoint: http://localhost:8000
    public PricePredictorClient();

    // Custom endpoint URL
    public PricePredictorClient(String endpointUrl);

    // Custom endpoint URL and timeout in milliseconds
    public PricePredictorClient(String endpointUrl, int timeoutMs);

    // Predict price for a card
    // Throws PricePredictorException on any failure
    public PriceEstimate predict(CardAttributes card)
        throws PricePredictorException;
}
```

### CardAttributes

Input data for a prediction. Uses builder pattern.

```java
package com.pricepredictor.connector;

public class CardAttributes {

    public static Builder builder() { ... }

    public static class Builder {
        public Builder name(String name);
        public Builder manaCost(String manaCost);
        public Builder types(String... types);        // required
        public Builder types(List<String> types);     // required
        public Builder supertypes(String... types);
        public Builder supertypes(List<String> types);
        public Builder subtypes(String... types);
        public Builder subtypes(List<String> types);
        public Builder oracleText(String text);
        public Builder keywords(String... keywords);
        public Builder keywords(List<String> keywords);
        public Builder power(String power);
        public Builder toughness(String toughness);
        public Builder loyalty(String loyalty);
        public CardAttributes build();  // throws if types is empty
    }
}
```

### PriceEstimate

Prediction result. Immutable.

```java
package com.pricepredictor.connector;

public record PriceEstimate(
    double predictedPriceEur,
    String modelVersion
) {}
```

### PricePredictorException

Base exception for all connector errors.

```java
package com.pricepredictor.connector;

public class PricePredictorException extends Exception {
    public PricePredictorException(String message);
    public PricePredictorException(String message, Throwable cause);
}

// Service unreachable (connection refused, timeout)
public class ServiceUnavailableException
    extends PricePredictorException { ... }

// Server returned error or unparseable response
public class InvalidResponseException
    extends PricePredictorException { ... }
```

## Usage Example (5 lines)

```java
var client = new PricePredictorClient();
var estimate = client.predict(CardAttributes.builder()
    .types("Creature")
    .manaCost("1 G G")
    .power("2").toughness("2")
    .build());
System.out.println(estimate.predictedPriceEur()); // e.g. 0.15
```

## Error Handling Contract

| Scenario | Exception Type | Message Pattern |
|----------|---------------|-----------------|
| Service not running | ServiceUnavailableException | "Connection refused: {url}" |
| Request timeout (>5s default) | ServiceUnavailableException | "Request timed out after {ms}ms" |
| Server returns 400 | InvalidResponseException | Server's error message |
| Server returns 500 | InvalidResponseException | Server's error message |
| Unparseable JSON response | InvalidResponseException | "Failed to parse response: {detail}" |
| Network I/O error | ServiceUnavailableException | "Network error: {detail}" |

## Thread Safety

`PricePredictorClient` is thread-safe. It creates a new
`HttpClient` instance internally and can be shared across threads.
Each `predict()` call is independent.

## Dependencies

**None.** The connector uses only:
- `java.net.http.HttpClient` (JDK 11+)
- `java.net.URI` (JDK)
- String processing for JSON parsing and Forge script serialization

No external JSON library — the response JSON is simple enough to
parse with basic string operations (two fields: a number and a
string).
