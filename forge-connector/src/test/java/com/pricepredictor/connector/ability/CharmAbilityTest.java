package com.pricepredictor.connector.ability;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A mode's additional cost, as the card prints it.
 *
 * <p>Forge writes a cost space-separated ({@code ModeCost$ 3 W W}) and the
 * converted text is read by a tokenizer for which braces delimit one atom. One
 * pair of braces around the whole cost therefore produced a distinct vocabulary
 * entry per multi-shard mode cost — {@code {3 W W}}, {@code {B B}},
 * {@code {5 U}} — each unrelated to the {@code {3}} and {@code {W}} every other
 * cost in the corpus is spelled with, and each seen a handful of times.
 */
class CharmAbilityTest {

    @Test
    void aMultiShardCostBecomesOneSymbolPerShard() {
        assertEquals("{3}{W}{W}", CharmAbility.manaCostSymbols("3 W W"));
    }

    @Test
    void aSingleShardCostIsUnchanged() {
        // The common case, and the reason this went unnoticed: 60 of the 68
        // cards using ModeCost have a one-shard cost, which was already right.
        assertEquals("{1}", CharmAbility.manaCostSymbols("1"));
        assertEquals("{0}", CharmAbility.manaCostSymbols("0"));
    }

    @Test
    void aHybridShardStaysOneSymbol() {
        // The slash is inside a shard, not between two of them.
        assertEquals("{2}{W/U}", CharmAbility.manaCostSymbols("2 W/U"));
    }

    @Test
    void surroundingSpaceIsNotAShard() {
        assertEquals("{3}{W}", CharmAbility.manaCostSymbols("  3   W  "));
    }
}
