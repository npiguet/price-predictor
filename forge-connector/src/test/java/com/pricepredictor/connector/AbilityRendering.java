package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;

/**
 * An ability's line as the converter writes it, for tests that assert on one
 * ability's text without building a card around it.
 */
final class AbilityRendering {

    private AbilityRendering() {
    }

    /**
     * The first line the converter writes for {@code ability}: rendered through
     * {@link Ability#appendBlock}, which numbers an actionable ability from a
     * fresh counter (so {@code spell[1]: }) and folds line breaks as the
     * converted file does.
     */
    static String lineOf(Ability ability) {
        List<RenderedLine> lines = new ArrayList<>();
        ability.appendBlock(new ActionCounter(0), 0, lines);
        return lines.get(0).text();
    }
}
