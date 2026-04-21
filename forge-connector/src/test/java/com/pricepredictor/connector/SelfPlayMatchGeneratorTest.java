package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure unit tests for {@link SelfPlayMatchGenerator}.
 *
 * <p>The Forge-dependent paths in {@link SelfPlayMatchGenerator#generateMatch()}
 * (pool generation, deck materialization, game play) are exercised by integration
 * tests. These unit tests cover the routing logic only:
 * <ul>
 *   <li>method 5 weighting (4/14)</li>
 *   <li>same-set constraint for method 5 picks</li>
 *   <li>exclude-self for method 5 picks</li>
 * </ul>
 * The methods 1-4 distribution itself is owned by {@link DeckBuilder} and
 * verified in {@link DeckBuilderTest}.
 */
class SelfPlayMatchGeneratorTest {

    private static GeneratedDeck deck(String setCode, String prefix) {
        // 40 fake card names — content doesn't matter for routing tests.
        String[] names = new String[40];
        for (int i = 0; i < 40; i++) {
            names[i] = prefix + "_" + i;
        }
        return new GeneratedDeck(setCode, List.of(names));
    }

    private static GeneratedDecksIndex indexOf(GeneratedDeck... decks) {
        return new GeneratedDecksIndex(List.of(decks));
    }

    private static SelfPlayMatchGenerator generatorWith(GeneratedDecksIndex index, Random random) {
        // Real DeckBuilder/PoolGenerator/GamePlayer — never invoked in these tests.
        return new SelfPlayMatchGenerator(
                index,
                new DeckBuilder(),
                new PoolGenerator(),
                new GamePlayer(),
                List.of("MH3", "BLB", "RVR"),
                random
        );
    }

    // ── rollIsMethod5 distribution ────────────────────────────────────────────

    @Test
    void rollIsMethod5WeightsApproximate4Over14() {
        SelfPlayMatchGenerator gen = generatorWith(
                indexOf(deck("MH3", "a")), new Random(123));

        int trials = 14_000;
        int method5Count = 0;
        for (int i = 0; i < trials; i++) {
            if (gen.rollIsMethod5() == 5) method5Count++;
        }

        // Expected ~4000 / 14000 = 28.57%. Allow ±3% tolerance.
        double frac = method5Count / (double) trials;
        assertTrue(frac >= 0.255 && frac <= 0.315,
                "method 5 fraction should be ~4/14 (≈0.286), got " + frac);
    }

    @Test
    void rollIsMethod5ReturnsOnly0Or5() {
        SelfPlayMatchGenerator gen = generatorWith(
                indexOf(deck("MH3", "a")), new Random(0));

        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            seen.add(gen.rollIsMethod5());
        }

        for (int v : seen) {
            assertTrue(v == 0 || v == 5, "rollIsMethod5 must return 0 or 5, got " + v);
        }
    }

    // ── pickDeckA ─────────────────────────────────────────────────────────────

    @Test
    void pickDeckAReturnsDeckFromIndex() {
        GeneratedDeck a = deck("MH3", "a");
        GeneratedDeck b = deck("BLB", "b");
        GeneratedDecksIndex index = indexOf(a, b);
        SelfPlayMatchGenerator gen = generatorWith(index, new Random(0));

        Set<String> seenSets = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            seenSets.add(gen.pickDeckA().setCode());
        }

        assertEquals(Set.of("MH3", "BLB"), seenSets);
    }

    // ── pickMethod5DeckB: same-set constraint ─────────────────────────────────

    @Test
    void pickMethod5DeckBReturnsSameSetAsDeckA() {
        GeneratedDeck a = deck("MH3", "a");
        GeneratedDeck b = deck("MH3", "b");
        GeneratedDeck c = deck("BLB", "c");
        GeneratedDeck d = deck("RVR", "d");
        SelfPlayMatchGenerator gen = generatorWith(indexOf(a, b, c, d), new Random(7));

        for (int i = 0; i < 100; i++) {
            GeneratedDeck pick = gen.pickMethod5DeckB(a);
            assertNotNull(pick);
            assertEquals("MH3", pick.setCode(),
                    "Method 5 deck B must share set code with deck A");
        }
    }

    // ── pickMethod5DeckB: exclude-self ───────────────────────────────────────

    @Test
    void pickMethod5DeckBNeverReturnsDeckA() {
        GeneratedDeck a = deck("MH3", "a");
        GeneratedDeck b = deck("MH3", "b");
        SelfPlayMatchGenerator gen = generatorWith(indexOf(a, b), new Random(42));

        for (int i = 0; i < 100; i++) {
            GeneratedDeck pick = gen.pickMethod5DeckB(a);
            assertNotNull(pick);
            assertNotSame(a, pick, "Deck A must never be returned for method 5");
            assertSame(b, pick, "Only other MH3 deck is b, so pick must be b");
        }
    }

    @Test
    void pickMethod5DeckBReturnsNullWhenOnlyOneDeckOfSet() {
        GeneratedDeck a = deck("MH3", "a");
        GeneratedDeck c = deck("BLB", "c");
        SelfPlayMatchGenerator gen = generatorWith(indexOf(a, c), new Random(0));

        // Only one MH3 deck — exclude=a leaves zero candidates.
        assertNull(gen.pickMethod5DeckB(a));
    }
}
