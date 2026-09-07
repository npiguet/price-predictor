package com.pricepredictor.connector.effects;

import java.util.Collection;
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

    /** A JSON array of already-rendered element fragments. */
    public static String array(Collection<String> rendered) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String element : rendered) {
            joiner.add(element);
        }
        return joiner.toString();
    }

    /** A JSON array of strings. */
    public static String stringArray(Collection<String> values) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String value : values) {
            joiner.add(string(value));
        }
        return joiner.toString();
    }
}
