package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.effects.RecordAbilities.abilityKeys;
import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Picking records out of a shard's lines by moment, script and event type.
 */
final class RecordPicks {

    private RecordPicks() {
    }

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

    /** The records carrying an event of {@code type}, in the order written. */
    static List<String> withEventOfType(List<String> all, String type) {
        return all.stream().filter(record -> eventTypes(record).contains(type)).toList();
    }
}
