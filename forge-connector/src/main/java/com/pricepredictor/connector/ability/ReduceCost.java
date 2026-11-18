package com.pricepredictor.connector.ability;

import forge.game.keyword.Keyword;

/**
 * The cost-reduction sentence a keyword's original script text carries, if any.
 *
 * <p>Pattern A (Equip): {@code Equip:cost:::ReduceCost$ SVar:full sentence}.
 * Pattern B (Specialize): {@code Specialize:cost::full sentence.:ReduceCost$ SVar},
 * the full sentence being field 3. (Adapt and Monstrosity are not members of
 * the {@link Keyword} enum, so no pattern covers them.)
 */
record ReduceCost(String original) {

    private static final String MARKER = "ReduceCost$ ";

    /** The sentence, or null if {@code kw}'s original text has none. */
    String sentence(Keyword kw) {
        if (original == null) return null;
        String equip = equip();
        if (equip != null) return equip;
        return kw == Keyword.SPECIALIZE ? specialize() : null;
    }

    /** Pattern A: the human text after the SVar name that follows the marker, or null. */
    private String equip() {
        int idx = original.indexOf(MARKER);
        if (idx < 0) return null;
        String after = original.substring(idx + MARKER.length());
        int colon = after.indexOf(':');
        return colon < 0 ? null : after.substring(colon + 1);
    }

    /** Pattern B: the full description in field 3, with the marker in field 4, or null. */
    private String specialize() {
        KeywordSpec spec = new KeywordSpec(original);
        if (spec.size() < 5 || !original.contains("ReduceCost$")) return null;
        return spec.part(3).isEmpty() ? null : spec.part(3);
    }
}
