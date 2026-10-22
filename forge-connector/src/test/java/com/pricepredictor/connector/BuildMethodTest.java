package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@link BuildMethod}'s weighted roll and {@link DeckBuilder#selectMethod()}.
 * Neither needs Forge's card database, so this runs in the default build.
 */
class BuildMethodTest {

    @Test
    void theRollThresholdsAreTheWeightsCumulated() {
        // 4:3:2:1 over a total of 10: the same cuts a literal {0.4, 0.7, 0.9}
        // table drew, so the corpus's method mix is unchanged.
        assertEquals(10, BuildMethod.TOTAL_WEIGHT);
        assertEquals(BuildMethod.FORGE_BEST, BuildMethod.forRoll(Math.nextDown(0.4)));
        assertEquals(BuildMethod.FORGE_3SUB, BuildMethod.forRoll(0.4));
        assertEquals(BuildMethod.FORGE_3SUB, BuildMethod.forRoll(Math.nextDown(0.7)));
        assertEquals(BuildMethod.FORGE_8SUB, BuildMethod.forRoll(0.7));
        assertEquals(BuildMethod.FORGE_8SUB, BuildMethod.forRoll(Math.nextDown(0.9)));
        assertEquals(BuildMethod.RANDOM, BuildMethod.forRoll(0.9));
        assertEquals(BuildMethod.RANDOM, BuildMethod.forRoll(Math.nextDown(1.0)));
    }

    @Test
    void selectMethodRollsOnTheBuildersRandomAndReachesEveryMethod() {
        DeckBuilder builder = new DeckBuilder(new Random(42));
        Random mirror = new Random(42);
        Set<BuildMethod> seen = new HashSet<>();

        for (int i = 0; i < 500; i++) {
            BuildMethod selected = builder.selectMethod();
            assertEquals(BuildMethod.forRoll(mirror.nextDouble()), selected);
            seen.add(selected);
        }

        assertEquals(Set.of(BuildMethod.values()), seen,
                "All 4 methods should appear within 500 draws");
    }
}
