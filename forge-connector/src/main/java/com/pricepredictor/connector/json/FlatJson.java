package com.pricepredictor.connector.json;

import java.util.HashMap;
import java.util.Map;

/**
 * Minimal reader for a flat JSON object whose values are strings, integers,
 * booleans or null — all the supervisor's pick response ever carries. Integers
 * come back as {@link Long}.
 *
 * <p>The reading half of {@link Json}, and as narrow as the one document it
 * reads: anything richer than a flat object is refused rather than half-read.
 */
public final class FlatJson {

    private FlatJson() {
    }

    /** @throws IllegalArgumentException on anything that is not such an object */
    public static Map<String, Object> parse(String json) {
        Map<String, Object> map = new HashMap<>();
        int n = json.length();
        int i = skipWs(json, 0);
        if (i >= n || json.charAt(i) != '{') {
            throw new IllegalArgumentException("expected '{'");
        }
        i = skipWs(json, i + 1);
        if (i < n && json.charAt(i) == '}') {
            return map;
        }
        while (i < n) {
            i = skipWs(json, i);
            if (json.charAt(i) != '"') {
                throw new IllegalArgumentException("expected key string");
            }
            StringBuilder key = new StringBuilder();
            i = parseString(json, i, key);
            i = skipWs(json, i);
            if (i >= n || json.charAt(i) != ':') {
                throw new IllegalArgumentException("expected ':'");
            }
            i = skipWs(json, i + 1);
            char c = json.charAt(i);
            if (c == '"') {
                StringBuilder val = new StringBuilder();
                i = parseString(json, i, val);
                map.put(key.toString(), val.toString());
            } else if (c == 't') {
                i = expectLiteral(json, i, "true");
                map.put(key.toString(), true);
            } else if (c == 'f') {
                i = expectLiteral(json, i, "false");
                map.put(key.toString(), false);
            } else if (c == 'n') {
                i = expectLiteral(json, i, "null");
                map.put(key.toString(), null);
            } else {
                int start = i;
                while (i < n && (Character.isDigit(json.charAt(i))
                        || json.charAt(i) == '-' || json.charAt(i) == '+')) {
                    i++;
                }
                map.put(key.toString(), Long.parseLong(json.substring(start, i)));
            }
            i = skipWs(json, i);
            if (i < n && json.charAt(i) == ',') {
                i++;
                continue;
            }
            if (i < n && json.charAt(i) == '}') {
                break;
            }
            throw new IllegalArgumentException("expected ',' or '}'");
        }
        return map;
    }

    /** The index past {@code literal}, which must start at {@code i}. */
    private static int expectLiteral(String s, int i, String literal) {
        if (!s.regionMatches(i, literal, 0, literal.length())) {
            throw new IllegalArgumentException("expected '" + literal + "'");
        }
        return i + literal.length();
    }

    private static int skipWs(String s, int i) {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
        return i;
    }

    /** Parse a JSON string starting at the opening quote; returns the index past it. */
    private static int parseString(String s, int i, StringBuilder out) {
        i++;  // opening quote
        int n = s.length();
        while (i < n) {
            char c = s.charAt(i++);
            if (c == '"') {
                return i;
            }
            if (c == '\\') {
                char e = s.charAt(i++);
                switch (e) {
                    case '"' -> out.append('"');
                    case '\\' -> out.append('\\');
                    case '/' -> out.append('/');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'u' -> {
                        out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> out.append(e);
                }
            } else {
                out.append(c);
            }
        }
        throw new IllegalArgumentException("unterminated string");
    }
}
