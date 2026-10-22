package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The closed vocabularies a record is written in, spelled out literally.
 *
 * <p>Each one has a Python twin that refuses a value it does not have, and the
 * corpus it guards is append-only, so a rename on this side alone loses every
 * record collected after it. The literals here are the Python spellings: a
 * change to an enum that does not change this test is a change to the wire.
 */
class WireVocabularyTest {

    private static <E extends Enum<E>> List<String> wireNames(
            E[] values, Function<E, String> wireName) {
        return Arrays.stream(values).map(wireName).toList();
    }

    /** {@code RecordKind} in {@code records.py}. */
    @Test
    void recordKinds() {
        assertEquals(
                List.of("resolution", "rewrite", "continuous", "combat", "trigger", "playability"),
                wireNames(RecordKind.values(), RecordKind::wireName));
    }

    /** {@code Moment} in {@code records.py}. */
    @Test
    void moments() {
        assertEquals(List.of("activation", "resolution"),
                wireNames(Moment.values(), Moment::wireName));
    }

    /** {@code PlayabilitySubkind} in {@code records.py}. */
    @Test
    void playabilitySubkinds() {
        assertEquals(List.of("decision", "attackers", "blockers"),
                wireNames(PlayabilitySubkind.values(), PlayabilitySubkind::wireName));
    }
}
