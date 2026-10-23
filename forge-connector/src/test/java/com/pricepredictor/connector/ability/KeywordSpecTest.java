package com.pricepredictor.connector.ability;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class KeywordSpecTest {

    @Test
    void aPartIsOneFieldAndTheRestKeepsLaterColons() {
        KeywordSpec spec = new KeywordSpec("Protection:Card.MultiColor:multicolored:x");

        assertEquals(4, spec.size());
        assertEquals("Card.MultiColor", spec.part(1));
        assertEquals("Card.MultiColor:multicolored:x", spec.rest(1));
        assertEquals("multicolored:x", spec.rest(2));
        assertNull(spec.part(4));
        assertNull(spec.rest(4));
    }

    @Test
    void textWithoutColonsHasOneFieldAndNoRest() {
        KeywordSpec spec = new KeywordSpec("Protection from white");

        assertEquals(1, spec.size());
        assertEquals("Protection from white", spec.part(0));
        assertNull(spec.rest(1));
    }

    @Test
    void emptyFieldsCountIncludingTrailingOnes() {
        KeywordSpec spec = new KeywordSpec("Specialize:2::");

        assertEquals(4, spec.size());
        assertEquals("", spec.part(2));
        assertEquals("", spec.part(3));
        assertEquals(":", spec.rest(2));
    }
}
