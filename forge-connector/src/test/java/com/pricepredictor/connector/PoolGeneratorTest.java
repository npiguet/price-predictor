package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(ForgeExtension.class)
class PoolGeneratorTest {

    private static final Set<String> BASIC_LAND_NAMES = Set.of(
            "Plains", "Island", "Swamp", "Mountain", "Forest",
            "Wastes", "Snow-Covered Plains", "Snow-Covered Island",
            "Snow-Covered Swamp", "Snow-Covered Mountain", "Snow-Covered Forest"
    );

    @Test
    void generateReturnsCorrectPoolCount() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 5);
        assertEquals(5, pools.size());
    }

    @Test
    void generateSinglePool() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 1);
        assertEquals(1, pools.size());
    }

    @Test
    void generatePoolContainsNoBasicLands() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 3);
        for (List<String> pool : pools) {
            for (String cardName : pool) {
                assertFalse(BASIC_LAND_NAMES.contains(cardName),
                        "Basic land found in pool: " + cardName);
            }
        }
    }

    @Test
    void generatePoolHasExpectedSize() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 1);
        List<String> pool = pools.get(0);
        // 6 boosters × ~14–15 non-land cards = 84–90 cards
        assertTrue(pool.size() >= 70,
                "Pool too small: " + pool.size());
        assertTrue(pool.size() <= 100,
                "Pool too large: " + pool.size());
    }

    @Test
    void generateZeroPoolsReturnsEmptyList() {
        PoolGenerator generator = new PoolGenerator();
        List<List<String>> pools = generator.generate("RVR", 0);
        assertEquals(0, pools.size());
    }

    @Test
    void generateThrowsForNullSetCode() {
        PoolGenerator generator = new PoolGenerator();
        assertThrows(IllegalArgumentException.class, () -> generator.generate(null, 1));
    }

    @Test
    void generateThrowsForUnknownSetCode() {
        PoolGenerator generator = new PoolGenerator();
        assertThrows(IllegalArgumentException.class, () -> generator.generate("XXXXUNKNOWN", 1));
    }
}
