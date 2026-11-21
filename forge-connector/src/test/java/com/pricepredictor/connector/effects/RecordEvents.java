package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A record's payload and the events it carries.
 */
final class RecordEvents {

    private RecordEvents() {
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
}
