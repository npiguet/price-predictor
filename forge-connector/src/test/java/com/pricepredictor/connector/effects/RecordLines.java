package com.pricepredictor.connector.effects;

import org.eclipse.jetty.util.ajax.JSON;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reading the record lines a shard run wrote, for tests that pick records
 * out by what they act through and pin their events.
 *
 * <p>Fields are read off the parsed record, so a lookup cannot land on a key
 * of the same name somewhere else in the line. A test pinning how a value is
 * <em>spelled</em> — the dedup hashes rendered bytes — compares the rendered
 * text instead, through {@link #stateOf} and {@link #payloadOf}.
 */
final class RecordLines {

    private RecordLines() {
    }

    // ── parsing ─────────────────────────────────────────────────────────

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

    /** The record's {@code payload} object. */
    static Map<String, Object> payload(String record) {
        Map<String, Object> payload = asMap(parse(record).get("payload"));
        assertTrue(payload != null, "no payload in " + record);
        return payload;
    }

    /**
     * The record's events, in order: a resolution's or a combat record's
     * {@code events}, or the one {@code event} a trigger record carries.
     * None for a record of any other payload.
     */
    static List<Map<String, Object>> events(String record) {
        Map<String, Object> payload = payload(record);
        List<Map<String, Object>> events = new ArrayList<>();
        if (payload.get("events") instanceof List<?> list) {
            list.forEach(event -> events.add(asMap(event)));
        } else if (payload.get("event") instanceof Map<?, ?> event) {
            events.add(asMap(event));
        }
        return events;
    }

    /** The one event a record carries. */
    static Map<String, Object> onlyEvent(String record) {
        List<Map<String, Object>> events = events(record);
        assertEquals(1, events.size(), record);
        return events.get(0);
    }

    /** An event's {@code params} object. */
    static Map<String, Object> params(Map<String, Object> event) {
        Map<String, Object> params = asMap(event.get("params"));
        assertTrue(params != null, "no params in " + event);
        return params;
    }

    /** The types of a record's events, in order. */
    static List<String> eventTypes(String record) {
        return events(record).stream().map(event -> (String) event.get("type")).toList();
    }

    /**
     * The record's events whose {@code subjects} name {@code ref}, in order:
     * a check that no event acts on something holds whatever else the event
     * carries and wherever the ref sits among its subjects.
     */
    static List<Map<String, Object>> eventsNaming(String record, String ref) {
        return events(record).stream()
                .filter(event -> event.get("subjects") instanceof List<?> subjects
                        && subjects.contains(ref))
                .toList();
    }

    /** The records carrying an event of {@code type}, in the order written. */
    static List<String> withEventOfType(List<String> all, String type) {
        return all.stream().filter(record -> eventTypes(record).contains(type)).toList();
    }

    // ── the rendered text ───────────────────────────────────────────────

    /** A record's {@code state} object as written, key included. */
    static String stateOf(String record) {
        int from = record.indexOf("\"state\":");
        int to = record.indexOf(",\"payload\":");
        assertTrue(from >= 0 && to > from, record);
        return record.substring(from, to);
    }

    /** A record's {@code payload} object as written, key included. */
    static String payloadOf(String record) {
        int from = record.indexOf("\"payload\":");
        assertTrue(from >= 0, record);
        return record.substring(from, record.length() - 1);
    }

    /** How many times {@code needle} occurs in {@code haystack}, overlaps counted. */
    static int count(String haystack, String needle) {
        int found = 0;
        for (int at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + 1)) {
            found++;
        }
        return found;
    }

    // ── picking records out ─────────────────────────────────────────────

    /** Every resolution record of {@code moment}, in the order written. */
    static List<String> records(List<String> all, String moment) {
        List<String> found = new ArrayList<>();
        for (String record : all) {
            if (moment.equals(parse(record).get("moment"))) {
                found.add(record);
            }
        }
        return found;
    }

    /** The acting keys of a record's {@code ability}. */
    static List<Map<String, Object>> abilityKeys(String record) {
        Object ability = parse(record).get("ability");
        List<Map<String, Object>> keys = new ArrayList<>();
        if (ability instanceof List<?> list) {
            list.forEach(key -> keys.add(asMap(key)));
        }
        return keys;
    }

    /** The script file of the record's first acting key, or null when it has none. */
    static String scriptOf(String record) {
        List<Map<String, Object>> keys = abilityKeys(record);
        return keys.isEmpty() ? null : (String) keys.get(0).get("script_file");
    }

    /**
     * The {@code Choices$} position of the mode the record's first acting key
     * names, or null when it acts through no mode.
     */
    static Long optionOf(String record) {
        List<Map<String, Object>> keys = abilityKeys(record);
        return keys.isEmpty() ? null : (Long) keys.get(0).get("option");
    }

    /** The record's {@code link_id}, or null when it has none. */
    static String linkOf(String record) {
        return (String) parse(record).get("link_id");
    }

    /** The one record of {@code moment} acting through {@code script}. */
    static String only(List<String> all, String moment, String script) {
        return only(all, moment, script, "");
    }

    /**
     * The same, through the line of one trait kind of the script: a kind
     * whose name starts with {@code traitKind}.
     */
    static String only(List<String> all, String moment, String script, String traitKind) {
        List<String> found = new ArrayList<>();
        for (String record : records(all, moment)) {
            if (script.equals(scriptOf(record)) && abilityKeys(record).stream()
                    .anyMatch(key -> ((String) key.get("trait_kind")).startsWith(traitKind))) {
                found.add(record);
            }
        }
        assertEquals(1, found.size(), moment + " through " + script + ": " + all);
        return found.get(0);
    }

    /** The outcome a cost half was stamped with. */
    static String outcomeOf(String record) {
        Object outcome = payload(record).get("outcome");
        assertTrue(outcome instanceof String, "no outcome in " + record);
        return (String) outcome;
    }
}
