package com.pricepredictor.connector.json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Written JSON split back into its parts, each kept as the text it was
 * written as: an array's elements, an object's fields. Nested values are not
 * parsed, so two parts compare as text, which is what comparing two renders
 * of the same writer needs.
 */
public final class JsonParts {

    private JsonParts() {
    }

    /** The text of each element of {@code array}, in order. */
    public static List<String> elements(String array) {
        String body = inside(array, '[', ']');
        List<String> parts = new ArrayList<>();
        for (String part : topLevel(body)) {
            parts.add(part);
        }
        return parts;
    }

    /** The text of each field of {@code object}, by name, in the order written. */
    public static Map<String, String> fields(String object) {
        String body = inside(object, '{', '}');
        Map<String, String> fields = new LinkedHashMap<>();
        for (String member : topLevel(body)) {
            int colon = nameEnd(member);
            String name = member.substring(member.indexOf('"') + 1, colon - 1);
            fields.put(name, member.substring(colon + 1).strip());
        }
        return fields;
    }

    /** What lies between the outer brackets of {@code json}. */
    private static String inside(String json, char open, char close) {
        String trimmed = json.strip();
        if (trimmed.length() < 2 || trimmed.charAt(0) != open
                || trimmed.charAt(trimmed.length() - 1) != close) {
            throw new IllegalArgumentException("not a JSON " + (open == '[' ? "array" : "object")
                    + ": " + abbreviate(json));
        }
        return trimmed.substring(1, trimmed.length() - 1);
    }

    /** {@code body} cut at every comma outside a string and outside any nesting. */
    private static List<String> topLevel(String body) {
        List<String> parts = new ArrayList<>();
        int depth = 0;
        boolean inString = false;
        int start = 0;
        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (inString) {
                if (c == '\\') {
                    i++;
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '[' || c == '{') {
                depth++;
            } else if (c == ']' || c == '}') {
                depth--;
            } else if (c == ',' && depth == 0) {
                parts.add(body.substring(start, i).strip());
                start = i + 1;
            }
        }
        String last = body.substring(start).strip();
        if (!last.isEmpty() || !parts.isEmpty()) {
            parts.add(last);
        }
        return parts;
    }

    /** The index of the colon after a member's quoted name. */
    private static int nameEnd(String member) {
        int i = member.indexOf('"') + 1;
        while (member.charAt(i) != '"') {
            i += member.charAt(i) == '\\' ? 2 : 1;
        }
        return member.indexOf(':', i);
    }

    private static String abbreviate(String text) {
        return text.length() <= 80 ? text : text.substring(0, 80) + "...";
    }
}
