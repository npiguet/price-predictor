package com.pricepredictor.connector.ability;

/**
 * A keyword's original script text read as its colon-separated fields, as in
 * {@code Kicker:1 G:Sac<1/Creature>} or {@code Class:2:…}.
 *
 * <p>What each field means is the keyword's own business; this only answers
 * the two questions the readers ask of it: one field on its own, and
 * everything from a field on, colons included, for a keyword whose last field
 * may itself hold colons.
 */
public final class KeywordSpec {

    private final String original;
    private final String[] fields;

    public KeywordSpec(String original) {
        this.original = original;
        this.fields = original.split(":", -1);
    }

    /** The number of fields, one more than the number of colons. */
    public int size() {
        return fields.length;
    }

    /** Field {@code i} on its own, or null when the text has fewer fields. */
    public String part(int i) {
        return i < fields.length ? fields[i] : null;
    }

    /**
     * The text after the {@code i}-th colon, later colons included, or null
     * when the text has fewer than {@code i} colons.
     */
    public String rest(int i) {
        return i < fields.length ? original.split(":", i + 1)[i] : null;
    }
}
