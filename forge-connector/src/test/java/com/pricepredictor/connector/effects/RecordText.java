package com.pricepredictor.connector.effects;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A record line's rendered text: what a test pinning how a value is spelled compares.
 */
final class RecordText {

    private RecordText() {
    }

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
}
