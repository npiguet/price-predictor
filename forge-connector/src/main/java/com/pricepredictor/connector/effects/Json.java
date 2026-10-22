package com.pricepredictor.connector.effects;

import java.util.Collection;
import java.util.Map;
import java.util.StringJoiner;

/**
 * Minimal JSON writing for the sidecar and the record shards.
 *
 * <p>The connector's own code is stdlib-only apart from minlog, and the two
 * formats it emits are small and fully under our control, so a JSON library
 * would be a new bundled dependency for no benefit. Reading is the Python
 * side's job.
 */
public final class Json {

    private Json() {
    }

    /** A JSON string literal, or {@code null} for a null input. */
    public static String string(String value) {
        if (value == null) return "null";
        StringBuilder sb = new StringBuilder(value.length() + 2);
        sb.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.append('"').toString();
    }

    /** A JSON array of strings. */
    public static String stringArray(Collection<String> values) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String value : values) {
            joiner.add(string(value));
        }
        return joiner.toString();
    }

    /**
     * Any value as JSON: a boolean or number bare, a map as an object in its
     * own iteration order, an iterable as an array, and anything else as the
     * string literal of its {@code String.valueOf}.
     *
     * <p>A null therefore becomes the string {@code "null"}, not JSON null.
     * Event params nested a null that way into corpora that cannot be
     * rewritten, and dedup hashes the rendered bytes, so a different spelling
     * here would split one event into two.
     */
    public static String value(Object value) {
        if (value instanceof Boolean || value instanceof Number) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            StringJoiner joiner = new StringJoiner(",", "{", "}");
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                joiner.add(string(String.valueOf(entry.getKey())) + ":" + value(entry.getValue()));
            }
            return joiner.toString();
        }
        if (value instanceof Iterable<?> items) {
            StringJoiner joiner = new StringJoiner(",", "[", "]");
            for (Object item : items) {
                joiner.add(value(item));
            }
            return joiner.toString();
        }
        return string(String.valueOf(value));
    }

    /** An empty object, for fields added one at a time. */
    static ObjectWriter object() {
        return new ObjectWriter();
    }

    /**
     * A JSON object whose fields render in the order they are added.
     *
     * <p>The order is part of the record contract, so the writer never sorts
     * and never drops a field: a null reaches the output however the value
     * passed for it spells one.
     */
    static final class ObjectWriter {

        private final StringJoiner fields = new StringJoiner(",", "{", "}");

        private ObjectWriter() {
        }

        /** A field whose value is already rendered JSON. */
        ObjectWriter raw(String name, String json) {
            fields.add(string(name) + ":" + json);
            return this;
        }

        /** A field rendered by {@link Json#value}. */
        ObjectWriter value(String name, Object value) {
            return raw(name, Json.value(value));
        }

        /** Every entry of an ordered map, as {@link #value} fields. */
        ObjectWriter values(Map<String, ?> entries) {
            for (Map.Entry<String, ?> entry : entries.entrySet()) {
                value(entry.getKey(), entry.getValue());
            }
            return this;
        }

        String render() {
            return fields.toString();
        }
    }
}
