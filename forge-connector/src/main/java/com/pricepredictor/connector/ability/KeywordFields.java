package com.pricepredictor.connector.ability;

import forge.game.keyword.KeywordInterface;

import java.util.Arrays;

/**
 * Parsed fields from a Forge keyword string in the format {@code "Keyword:field1:field2:..."}.
 * Provides safe indexed access without risk of {@link ArrayIndexOutOfBoundsException}.
 *
 * <p>Index 0 is always the keyword name; caller-meaningful data starts at index 1.
 */
public record KeywordFields(String[] parts) {

    /**
     * Parse a Forge keyword original string, splitting on {@code ':'} up to {@code maxFields} parts.
     * Pass {@code maxFields} equal to the highest field index you need plus one to avoid splitting
     * colons that are part of a field value (e.g. mana cost strings).
     */
    public static KeywordFields parse(String original, int maxFields) {
        return new KeywordFields(original.split(":", maxFields));
    }

    /**
     * Parse all fields of a Forge keyword original string with no field limit.
     * Use this when you need to iterate over all fields and there are no embedded colons
     * to protect (e.g. {@code AlternateAdditionalCost:cost1:cost2:...}).
     */
    public static KeywordFields parseAll(String original) {
        return new KeywordFields(original.split(":"));
    }

    /**
     * Parse a keyword's fields directly from a {@link KeywordInterface}.
     * Shorthand for {@code parse(ki.getOriginal(), maxFields)}.
     */
    public static KeywordFields from(KeywordInterface ki, int maxFields) {
        return parse(ki.getOriginal(), maxFields);
    }

    /** Return the keyword name (field 0). */
    public String keywordName() {
        return field(0);
    }

    /** Return the field at {@code index}, or an empty string if the field is absent. */
    public String field(int index) {
        return index < parts.length ? parts[index] : "";
    }

    /** Return true when the field at {@code index} is present and non-empty. */
    public boolean hasField(int index) {
        return index < parts.length && !parts[index].isEmpty();
    }

    /**
     * Return the last non-empty field after the keyword name (index &gt;= 1).
     * Useful when a keyword encodes both a filter and a human-readable description as separate
     * fields and the most specific (last) one should be used (e.g. Protection).
     * Returns an empty string if no such field exists.
     */
    public String lastNonEmptyField() {
        for (int i = parts.length - 1; i >= 1; i--) {
            if (!parts[i].isEmpty()) return parts[i];
        }
        return "";
    }

    /**
     * Return all fields after the keyword name (indices 1..N) as an array.
     * Returns an empty array when there are no fields beyond the keyword name.
     */
    public String[] tailFields() {
        return parts.length <= 1 ? new String[0] : Arrays.copyOfRange(parts, 1, parts.length);
    }
}
