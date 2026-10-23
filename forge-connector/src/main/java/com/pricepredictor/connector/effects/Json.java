package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

import java.util.Map;
import java.util.StringJoiner;
import java.util.function.Function;

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
    public static String stringArray(Iterable<String> values) {
        return array(values, Json::string);
    }

    /**
     * A JSON array of {@code items}, in iteration order, each rendered to
     * JSON by {@code render}.
     */
    public static <T> String array(Iterable<T> items, Function<? super T, String> render) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (T item : items) {
            joiner.add(render.apply(item));
        }
        return joiner.toString();
    }

    /**
     * Any value as JSON: null as JSON {@code null}, a boolean or number bare,
     * a {@link WireNamed} member as the string literal of its wire name, a map
     * as an object in its own iteration order, an iterable as an array, and
     * anything else as the string literal of its {@code String.valueOf}.
     *
     * <p>Dedup hashes the rendered bytes, so this one spelling is what makes
     * two reports of the same value the same event. A vocabulary member is
     * spelled by its wire name here rather than at each call site because the
     * fallback would otherwise write its Java constant name, a spelling every
     * reader refuses, wherever a caller forgot to convert it.
     */
    public static String value(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Boolean || value instanceof Number) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            StringJoiner joiner = new StringJoiner(",", "{", "}");
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                joiner.add(string(text(entry.getKey())) + ":" + value(entry.getValue()));
            }
            return joiner.toString();
        }
        if (value instanceof Iterable<?> items) {
            return array(items, Json::value);
        }
        return string(text(value));
    }

    /** A non-null scalar's spelling: its wire name if it has one. */
    private static String text(Object value) {
        return value instanceof WireNamed named ? named.wireName() : String.valueOf(value);
    }

    /** An empty object, for fields added one at a time. */
    public static ObjectWriter object() {
        return new ObjectWriter();
    }

    /**
     * A JSON object whose fields render in the order they are added.
     *
     * <p>The order is part of the record contract, so the writer never sorts
     * and never drops a field: a null passed to {@link #value} is written as
     * JSON {@code null}.
     */
    public static final class ObjectWriter {

        private final StringJoiner fields = new StringJoiner(",", "{", "}");

        private ObjectWriter() {
        }

        /** A field whose value is already rendered JSON. */
        public ObjectWriter raw(String name, String json) {
            fields.add(string(name) + ":" + json);
            return this;
        }

        /** A field rendered by {@link Json#value}. */
        public ObjectWriter value(String name, Object value) {
            return raw(name, Json.value(value));
        }

        /** Every entry of an ordered map, as {@link #value} fields. */
        public ObjectWriter values(Map<String, ?> entries) {
            for (Map.Entry<String, ?> entry : entries.entrySet()) {
                value(entry.getKey(), entry.getValue());
            }
            return this;
        }

        public String render() {
            return fields.toString();
        }
    }
}
