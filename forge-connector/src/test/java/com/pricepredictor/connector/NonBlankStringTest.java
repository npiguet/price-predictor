package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class NonBlankStringTest {

    @Test
    void of_returnsEmptyForNull() {
        assertEquals(Optional.empty(), NonBlankString.of(null));
    }

    @Test
    void of_returnsEmptyForEmptyString() {
        assertEquals(Optional.empty(), NonBlankString.of(""));
    }

    @Test
    void of_returnsEmptyForBlankString() {
        assertEquals(Optional.empty(), NonBlankString.of("   "));
        assertEquals(Optional.empty(), NonBlankString.of("\t\n"));
    }

    @Test
    void of_returnsTrimmedValueForValidString() {
        NonBlankString result = NonBlankString.of("  hello  ").orElseThrow();
        assertEquals("hello", result.value());
        assertEquals("hello", result.toString());
    }

    @Test
    void of_returnsUnchangedWhenAlreadyTrimmed() {
        NonBlankString result = NonBlankString.of("hello").orElseThrow();
        assertEquals("hello", result.value());
    }

    @Test
    void require_wrapsNonBlankString() {
        NonBlankString result = NonBlankString.require("flying");
        assertEquals("flying", result.value());
    }

    @Test
    void require_trimsInput() {
        assertEquals("flying", NonBlankString.require("  flying  ").value());
    }

    @Test
    void require_throwsForNull() {
        assertThrows(IllegalArgumentException.class, () -> NonBlankString.require(null));
    }

    @Test
    void require_throwsForBlank() {
        assertThrows(IllegalArgumentException.class, () -> NonBlankString.require(""));
        assertThrows(IllegalArgumentException.class, () -> NonBlankString.require("   "));
    }

    @Test
    void equality_basedOnTrimmedValue() {
        NonBlankString a = NonBlankString.require("flying");
        NonBlankString b = NonBlankString.require("flying");
        NonBlankString c = NonBlankString.require("  flying  ");
        assertEquals(a, b);
        assertEquals(a, c);
    }

    @Test
    void hashCode_consistentWithEquals() {
        assertEquals(
                NonBlankString.require("flying").hashCode(),
                NonBlankString.require("flying").hashCode());
    }

    @Test
    void toString_returnsValue() {
        assertEquals("flying", NonBlankString.require("flying").toString());
    }

    @Test
    void contentEquals_comparesWithString() {
        NonBlankString nbs = NonBlankString.require("flying");
        assertTrue(nbs.contentEquals("flying"));
        assertFalse(nbs.contentEquals("vigilance"));
    }

    @Test
    void contains_delegatesToValue() {
        NonBlankString nbs = NonBlankString.require("destroy target creature");
        assertTrue(nbs.contains("target"));
        assertFalse(nbs.contains("exile"));
    }

    @Test
    void startsWith_delegatesToValue() {
        NonBlankString nbs = NonBlankString.require("flying");
        assertTrue(nbs.startsWith("fly"));
        assertFalse(nbs.startsWith("swim"));
    }

    @Test
    void compareTo_delegatesToValue() {
        NonBlankString a = NonBlankString.require("abc");
        NonBlankString b = NonBlankString.require("xyz");
        assertTrue(a.compareTo(b) < 0);
        assertTrue(b.compareTo(a) > 0);
        assertEquals(0, a.compareTo(NonBlankString.require("abc")));
    }
}
