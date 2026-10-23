package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reading the record lines a shard run wrote, for tests that pick records
 * out by what they act through and pin their events.
 */
final class RecordLines {

    private RecordLines() {
    }

    /** Every resolution record of {@code moment}, in the order written. */
    static List<String> records(List<String> all, String moment) {
        List<String> found = new ArrayList<>();
        for (String record : all) {
            if (record.contains("\"moment\":\"" + moment + "\"")) {
                found.add(record);
            }
        }
        return found;
    }

    /** The script file of the record's first acting key. */
    static String scriptOf(String record) {
        int at = record.indexOf("\"script_file\":\"");
        assertTrue(at >= 0, record);
        int from = at + "\"script_file\":\"".length();
        return record.substring(from, record.indexOf('"', from));
    }

    /** The record's {@code link_id}, or null when it has none. */
    static String linkOf(String record) {
        int at = record.indexOf("\"link_id\":\"");
        if (at < 0) {
            return null;
        }
        int from = at + "\"link_id\":\"".length();
        return record.substring(from, record.indexOf('"', from));
    }

    /** The one record of {@code moment} acting through {@code script}. */
    static String only(List<String> all, String moment, String script) {
        return only(all, moment, script, "");
    }

    /**
     * The same, through the line of one trait kind of the script. The kind is
     * read inside {@code ability} only, since the snapshot names acting
     * traits too.
     */
    static String only(List<String> all, String moment, String script, String traitKind) {
        List<String> found = new ArrayList<>();
        for (String record : records(all, moment)) {
            String ability = record.substring(
                    record.indexOf("\"ability\":"), record.indexOf("\"ability_unresolved\":"));
            if (scriptOf(record).equals(script)
                    && ability.contains("\"trait_kind\":\"" + traitKind)) {
                found.add(record);
            }
        }
        assertEquals(1, found.size(), moment + " through " + script + ": " + all);
        return found.get(0);
    }

    /** The types of a record's events, in order. */
    static List<String> eventTypes(String record) {
        String payload = record.substring(record.indexOf("\"payload\":"));
        List<String> types = new ArrayList<>();
        Matcher type = Pattern.compile("\"type\":\"(\\w+)\"").matcher(payload);
        while (type.find()) {
            types.add(type.group(1));
        }
        return types;
    }

    /** The outcome a cost half was stamped with. */
    static String outcomeOf(String record) {
        int at = record.indexOf("\"outcome\":\"");
        assertTrue(at >= 0, record);
        int from = at + "\"outcome\":\"".length();
        return record.substring(from, record.indexOf('"', from));
    }
}
