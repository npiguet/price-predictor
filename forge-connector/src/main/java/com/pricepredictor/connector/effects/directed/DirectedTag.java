package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.json.Json;
import java.util.Objects;

/**
 * What a directed fork changed: the scenario variant it applied and the
 * operation that applied it. Written as a record's {@code directed} member,
 * collection metadata the reader keeps out of the model.
 */
public record DirectedTag(String variant, ForkOperation operation) {

    public DirectedTag {
        Objects.requireNonNull(variant, "variant");
        Objects.requireNonNull(operation, "operation");
    }

    public String toJson() {
        return Json.object()
                .value("variant", variant)
                .value("operation", operation)
                .render();
    }
}
