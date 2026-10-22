package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The reasons an {@code ability} can be empty are written into a corpus that
 * cannot be rebuilt, and Python's {@code UNRESOLVED_REASONS} in
 * {@code records.py} rejects any record whose reason is not in its set.
 */
class UnresolvedReasonsTest {

    @Test
    void theReasonsAreSpelledAsPythonSpellsThem() {
        assertEquals("no_card_state", ProvenanceKey.UNRESOLVED_NO_CARD_STATE);
        assertEquals("unknown_kind", ProvenanceKey.UNRESOLVED_UNKNOWN_KIND);
        assertEquals("engine_effect", ProvenanceKey.UNRESOLVED_ENGINE_EFFECT);
        assertEquals("granted_keyword", ProvenanceKey.UNRESOLVED_GRANTED_KEYWORD);
        assertEquals("granted_trait", ProvenanceKey.UNRESOLVED_GRANTED_TRAIT);
        assertEquals("unindexable", ProvenanceKey.UNRESOLVED_UNINDEXABLE);
    }
}
