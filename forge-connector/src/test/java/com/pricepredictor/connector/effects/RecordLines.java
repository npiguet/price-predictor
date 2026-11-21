package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.eclipse.jetty.util.ajax.JSON;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reading the record lines a shard run wrote, parsed: a record line as a map, and the maps inside it.
 */
final class RecordLines {

    private RecordLines() {
    }

    /**
     * A JSON object, parsed: objects as maps, arrays as lists, whole numbers
     * as {@code Long}, {@code null} as null.
     */
    static Map<String, Object> parse(String json) {
        Object parsed;
        try {
            parsed = JSON.parse(json);
        } catch (RuntimeException e) {
            throw new AssertionError("not JSON: " + json, e);
        }
        if (!(parsed instanceof Map)) {
            throw new AssertionError("not a JSON object: " + json);
        }
        return asMap(normalized(parsed));
    }

    private static Object normalized(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            map.forEach((key, entry) -> copy.put((String) key, normalized(entry)));
            return copy;
        }
        if (value instanceof Object[] array) {
            List<Object> copy = new ArrayList<>();
            for (Object entry : array) {
                copy.add(normalized(entry));
            }
            return copy;
        }
        if (value instanceof List<?> list) {
            return list.stream().map(RecordLines::normalized).toList();
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }

    /** An event's {@code params} object. */
    static Map<String, Object> params(Map<String, Object> event) {
        Map<String, Object> params = asMap(event.get("params"));
        assertTrue(params != null, "no params in " + event);
        return params;
    }
}
