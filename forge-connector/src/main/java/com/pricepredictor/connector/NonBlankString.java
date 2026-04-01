package com.pricepredictor.connector;

import java.util.Optional;

/**
 * A non-null, non-blank string value type.
 *
 * <p>Instances are always trimmed and guaranteed to contain at least one non-whitespace character.
 * Use {@link #of(String)} at system boundaries (Forge API calls, user input) where null or blank
 * input must be handled gracefully. Use {@link #require(String)} for values that are known to be
 * non-blank and should fail loudly if not.
 */
public record NonBlankString(String value) implements Comparable<NonBlankString> {

    public NonBlankString {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("value must not be null or blank");
        }
        value = value.trim();
    }

    /**
     * Parse a nullable or potentially blank string. Returns empty if the input is null, empty,
     * or whitespace-only. The primary entry point at system boundaries.
     */
    public static Optional<NonBlankString> of(String s) {
        if (s == null || s.isBlank()) return Optional.empty();
        return Optional.of(new NonBlankString(s));
    }

    /**
     * Wrap a string known to be non-blank. Throws {@link IllegalArgumentException} if blank.
     * Use for values already validated by the surrounding logic.
     */
    public static NonBlankString require(String s) {
        return new NonBlankString(s);
    }

    // String-like convenience methods — allow callers to use NonBlankString without
    // repeatedly unwrapping via .value(), especially in test assertions.
    public boolean contains(CharSequence s) {
        return value.contains(s);
    }

    public boolean startsWith(String prefix) {
        return value.startsWith(prefix);
    }

    public boolean endsWith(String suffix) {
        return value.endsWith(suffix);
    }

    /**
     * Compare content equality with a raw String. Use instead of .equals() when comparing to String literals.
     */
    public boolean contentEquals(String other) {
        return value.equals(other);
    }

    public int indexOf(int ch) {
        return value.indexOf(ch);
    }

    public int length() {
        return value.length();
    }

    public NonBlankString substring(int beginIndex, int endIndex) {
        return NonBlankString.require(value.substring(beginIndex, endIndex).trim());
    }

    /**
     * Returns the raw string value. Enables string concatenation and StringBuilder.append() without .value().
     */
    @Override
    public String toString() {
        return value;
    }

    @Override
    public int compareTo(NonBlankString o) {
        return value.compareTo(o.value);
    }
}
