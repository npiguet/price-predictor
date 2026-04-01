package com.pricepredictor.connector.ability;

/**
 * Parsed fields from a Forge keyword string in the format {@code "Keyword:field1:field2:..."}.
 * Provides safe indexed access without risk of {@link ArrayIndexOutOfBoundsException}.
 *
 * <p>Index 0 is always the keyword name; caller-meaningful data starts at index 1.
 */
record KeywordFields(String[] parts) {

    /**
     * Parse a Forge keyword original string, splitting on {@code ':'} up to {@code maxFields} parts.
     * Pass {@code maxFields} equal to the highest field index you need plus one to avoid splitting
     * colons that are part of a field value (e.g. mana cost strings).
     */
    static KeywordFields parse(String original, int maxFields) {
        return new KeywordFields(original.split(":", maxFields));
    }

    /** Return the field at {@code index}, or an empty string if the field is absent. */
    String field(int index) {
        return index < parts.length ? parts[index] : "";
    }

    /** Return true when the field at {@code index} is present and non-empty. */
    boolean hasField(int index) {
        return index < parts.length && !parts[index].isEmpty();
    }
}
