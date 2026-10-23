package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(ForgeExtension.class)
class PoolGeneratorTest {

    @Test
    void generateReturnsCorrectPoolCount() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 5, HoldoutCards.NONE);
        assertEquals(5, pools.size());
    }

    @Test
    void generatePoolContainsNoBasicLands() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 3, HoldoutCards.NONE);
        for (List<String> pool : pools) {
            for (String cardName : pool) {
                assertFalse(BasicLands.isBasic(cardName),
                        "Basic land found in pool: " + cardName);
            }
        }
    }

    @Test
    void generatePoolHasExpectedSize() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 1, HoldoutCards.NONE);
        List<String> pool = pools.get(0);
        // 6 boosters × ~14–15 non-land cards = 84–90 cards
        assertTrue(pool.size() >= 70,
                "Pool too small: " + pool.size());
        assertTrue(pool.size() <= 100,
                "Pool too large: " + pool.size());
    }

    @Test
    void generatePoolOmitsExcludedCardsAndKeepsItsSize() {
        // A depleted training pool must never contain a held-out card: the whole
        // point is that the training corpus can be collected without discarding
        // the games those cards appear in. Its size holds because a card is
        // redrawn within the slot rather than dropped: a pool short of cards
        // would build a different deck, and the depleted corpus is supposed to
        // differ from the full-strength one only in which cards exist.
        PoolGenerator generator = new PoolGenerator();
        List<String> reference = generator.generate("RVR", 1, HoldoutCards.NONE).get(0);
        Set<String> excluded = Set.copyOf(reference.subList(0, 10));

        List<String> pool = generator.generate("RVR", 1, HoldoutCards.of(excluded)).get(0);

        for (String cardName : pool) {
            assertFalse(excluded.contains(cardName),
                    "Excluded card found in depleted pool: " + cardName);
        }
        assertTrue(pool.size() >= 70, "Depleted pool too small: " + pool.size());
        assertTrue(pool.size() <= 100, "Depleted pool too large: " + pool.size());
    }

    @Test
    void generateZeroPoolsReturnsEmptyList() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 0, HoldoutCards.NONE);
        assertEquals(0, pools.size());
    }

    @Test
    void generateThrowsForNullSetCode() {
        PoolGenerator generator = new PoolGenerator();
        assertThrows(IllegalArgumentException.class, () -> generator.generate(null, 1, HoldoutCards.NONE));
    }

    @Test
    void generateThrowsForUnknownSetCode() {
        PoolGenerator generator = new PoolGenerator();
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> generator.generate("XXXXUNKNOWN", 1, HoldoutCards.NONE));
        assertEquals("Unknown or unsupported set code: XXXXUNKNOWN", thrown.getMessage());
    }
}
