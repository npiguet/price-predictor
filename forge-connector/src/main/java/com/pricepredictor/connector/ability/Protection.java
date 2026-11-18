package com.pricepredictor.connector.ability;

import java.util.Locale;

/**
 * A Protection keyword's script text, read as "Protection from &lt;target&gt;".
 *
 * <p>{@code Protection:Creature} names its target directly, and
 * {@code Protection:Card.MultiColor:multicolored} gives it after a filter,
 * as the human-readable text after the second colon. The colon-free form
 * ({@code Protection from white}) is already oracle text, so the original
 * is the answer. The keyword's title is not: Forge splits the script line
 * on its first space, leaving "from white" as the type description, and
 * Protection.getTitle() then prepends "Protection from " to produce
 * "Protection from from white".
 */
record Protection(String original) {

    String title() {
        KeywordSpec spec = new KeywordSpec(original);
        String target = spec.size() > 2 ? spec.rest(2) : spec.rest(1);
        return target == null ? original : "Protection from " + target.toLowerCase(Locale.ROOT);
    }
}
