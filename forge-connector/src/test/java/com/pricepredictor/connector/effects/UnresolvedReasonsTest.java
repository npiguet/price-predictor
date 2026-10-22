package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The reasons an {@code ability} can be empty are written into a corpus that
 * cannot be rebuilt, and Python's {@code UNRESOLVED_REASONS} in
 * {@code records.py} rejects any record whose reason is not in its set.
 */
class UnresolvedReasonsTest {

    @Test
    void theReasonsAreSpelledAsPythonSpellsThem() {
        assertEquals(
                List.of("no_card_state", "unknown_kind", "engine_effect",
                        "granted_keyword", "granted_trait", "unindexable"),
                Arrays.stream(UnresolvedReason.values()).map(UnresolvedReason::wireName).toList());
    }
}
