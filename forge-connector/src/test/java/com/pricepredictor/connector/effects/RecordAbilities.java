package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordEvents.payload;
import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The acting keys, links and outcome a record carries.
 */
final class RecordAbilities {

    private RecordAbilities() {
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

    /**
     * The ends of a record's {@code targets} links in {@code index} order,
     * each index from 0 held once: what a record whose one stack object
     * targets names as that object's targets.
     */
    static List<Object> targetEnds(String record) {
        List<?> links = (List<?>) asMap(parse(record).get("state")).get("links");
        Map<Long, Object> byIndex = new java.util.TreeMap<>();
        for (Object each : links) {
            Map<String, Object> link = asMap(each);
            if ("targets".equals(link.get("type"))) {
                assertEquals(null, byIndex.put((Long) link.get("index"), link.get("to")), record);
            }
        }
        assertEquals(new ArrayList<>(byIndex.keySet()),
                java.util.stream.LongStream.range(0, byIndex.size()).boxed().toList(), record);
        return new ArrayList<>(byIndex.values());
    }

    /** The outcome an effect half carries (FR-042b). */
    static String outcomeOf(String record) {
        Object outcome = payload(record).get("outcome");
        assertTrue(outcome instanceof String, "no outcome in " + record);
        return (String) outcome;
    }
}
