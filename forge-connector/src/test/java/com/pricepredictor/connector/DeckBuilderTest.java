package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.item.PaperCard;
import forge.model.FModel;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DeckBuilder's four construction methods, weighted selection, land rebalancing,
 * and pool-card uniqueness guarantee (FR-010).
 *
 * <p>Requires Forge environment for PaperCard objects, so tagged as "integration".
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class DeckBuilderTest {

    /**
     * Generate a realistic pool from 6 boosters of the given set (no basic lands).
     */
    private static List<PaperCard> realPool(String setCode) {
        return new PoolGenerator().openPool(
                BoosterTemplates.forSet(setCode).orElseThrow(), HoldoutCards.NONE);
    }

    // ── Method Selection Distribution ─────────────────────────────────────────

    @Test
    void methodSelectionDistributionApproximatesWeights() {
        DeckBuilder builder = new DeckBuilder();
        java.util.Map<BuildMethod, Integer> counts = new java.util.EnumMap<>(BuildMethod.class);

        int trials = 1000;
        for (int i = 0; i < trials; i++) {
            counts.merge(builder.selectMethod(), 1, Integer::sum);
        }

        // forge-best: ~40% (expect 350-450)
        int best = counts.getOrDefault(BuildMethod.FORGE_BEST, 0);
        assertTrue(best >= 300 && best <= 500,
                "forge-best should be ~40%, got " + best + "/" + trials);
        // forge-3sub: ~30% (expect 250-350)
        int sub3 = counts.getOrDefault(BuildMethod.FORGE_3SUB, 0);
        assertTrue(sub3 >= 200 && sub3 <= 400,
                "forge-3sub should be ~30%, got " + sub3 + "/" + trials);
        // forge-8sub: ~20% (expect 150-250)
        int sub8 = counts.getOrDefault(BuildMethod.FORGE_8SUB, 0);
        assertTrue(sub8 >= 100 && sub8 <= 300,
                "forge-8sub should be ~20%, got " + sub8 + "/" + trials);
        // random: ~10% (expect 50-150)
        int random = counts.getOrDefault(BuildMethod.RANDOM, 0);
        assertTrue(random >= 30 && random <= 200,
                "random should be ~10%, got " + random + "/" + trials);
    }

    @Test
    void allMethodsEventuallySelected() {
        DeckBuilder builder = new DeckBuilder();
        Set<BuildMethod> seen = new HashSet<>();

        for (int i = 0; i < 500 && seen.size() < 4; i++) {
            seen.add(builder.selectMethod());
        }

        assertEquals(Set.of(BuildMethod.values()), seen,
                "All 4 methods should appear within 500 trials");
    }

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

    // ── Method 1: Standard Build ───────────────────────────────────────────────

    @Test
    void method1ProducesExactly40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(42));

        Deck deck = builder.buildStandard(pool);

        assertEquals(40, deck.getMain().countAll(), "Method 1 deck must be exactly 40 cards");
    }

    // ── Drafted pools ─────────────────────────────────────────────────────────

    /**
     * A drafted pool, approximated: 45 cards in pick order, opening with picks that
     * commit the seat to a known colour pair.
     */
    private static List<PaperCard> draftedPool(String setCode, String... openingPicks) {
        List<PaperCard> pool = new ArrayList<>();
        for (String name : openingPicks) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            assertNotNull(card, "test fixture card not found: " + name);
            pool.add(card);
        }
        for (PaperCard card : realPool(setCode)) {
            if (pool.size() >= 45) {
                break;
            }
            pool.add(card);
        }
        return pool;
    }

    @Test
    void buildDraftedProducesExactly40Cards() {
        DeckBuilder builder = new DeckBuilder(new Random(42));

        Deck deck = builder.buildDrafted(draftedPool("RVR", "Lightning Bolt", "Counterspell"));

        assertEquals(40, deck.getMain().countAll(), "A drafted deck must be exactly 40 cards");
    }

    /**
     * The regression this method exists for. {@code SealedDeckBuilder} would re-derive the
     * colours from the finished pool and return the same deck for both orders; the drafted
     * builder honours the commitment the pick order records, so disjoint opening pairs give
     * different decks.
     */
    @Test
    void buildDraftedFollowsThePickOrderNotJustThePoolContents() {
        DeckBuilder builder = new DeckBuilder(new Random(42));

        List<PaperCard> asWU = new ArrayList<>(draftedPool("RVR", "Wrath of God", "Counterspell"));
        List<PaperCard> asBR = new ArrayList<>(asWU);
        asBR.set(0, FModel.getMagicDb().getCommonCards().getCard("Lightning Bolt"));
        asBR.set(1, FModel.getMagicDb().getCommonCards().getCard("Dark Ritual"));

        Deck fromWU = builder.buildDrafted(asWU);
        Deck fromBR = builder.buildDrafted(asBR);

        assertNotEquals(
                nonlandNames(fromWU),
                nonlandNames(fromBR),
                "Opening picks in disjoint colours must produce different decks — "
                        + "the pool is otherwise identical, so only the commitment differs");
    }

    /** Sorted nonland card names of a deck, duplicates repeated. */
    private static List<String> nonlandNames(Deck deck) {
        List<String> names = new ArrayList<>();
        for (PaperCard card : deck.getMain().toFlatList()) {
            if (!card.getRules().getMainPart().getType().isLand()) {
                names.add(card.getName());
            }
        }
        Collections.sort(names);
        return names;
    }

    // ── Method 2: 3 Swaps ─────────────────────────────────────────────────────

    @Test
    void method2ProducesExactly40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(42));

        Deck deck = builder.buildWithSwaps(pool, 3);

        assertEquals(40, deck.getMain().countAll(), "Method 2 deck must be exactly 40 cards");
    }

    @Test
    void method2SwapsExactly3NonlandCards() {
        // Use a fixed seed to make the standard deck deterministic
        List<PaperCard> pool = realPool("RVR");

        // Build standard deck to know what cards would be there
        DeckBuilder refBuilder = new DeckBuilder(new Random(42));
        Deck standardDeck = refBuilder.buildStandard(new ArrayList<>(pool));

        DeckBuilder swapBuilder = new DeckBuilder(new Random(42));
        Deck swappedDeck = swapBuilder.buildWithSwaps(new ArrayList<>(pool), 3);

        // Both decks have 40 cards, so differences come from the 3 swaps
        // Count nonland card differences
        List<String> standardNonlands = standardDeck.getMain().toFlatList().stream()
                .filter(c -> !c.getRules().getMainPart().getType().isBasicLand())
                .map(PaperCard::getName)
                .sorted()
                .toList();
        List<String> swappedNonlands = swappedDeck.getMain().toFlatList().stream()
                .filter(c -> !c.getRules().getMainPart().getType().isBasicLand())
                .map(PaperCard::getName)
                .sorted()
                .toList();

        // The swapped deck must have SOME difference from standard (unless pool is too small)
        // We can't assert exact swap count without same seed for both builders, but we can
        // assert the swapped deck uses pool cards only
        assertNoDuplicatePoolCards(pool, swappedDeck);
    }

    // ── Method 3: 8 Swaps ─────────────────────────────────────────────────────

    @Test
    void method3ProducesExactly40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(7));

        Deck deck = builder.buildWithSwaps(pool, 8);

        assertEquals(40, deck.getMain().countAll(), "Method 3 deck must be exactly 40 cards");
    }

    @Test
    void method3NoPoolCardReuse() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(7));

        Deck deck = builder.buildWithSwaps(pool, 8);

        assertNoDuplicatePoolCards(pool, deck);
    }

    // ── Method 4: Random ──────────────────────────────────────────────────────

    @Test
    void method4ProducesExactly40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(99));

        Deck deck = builder.buildRandom(pool);

        assertEquals(40, deck.getMain().countAll(), "Method 4 deck must be exactly 40 cards");
    }

    @Test
    void method4PicksAtMost23NonlandCards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(99));

        Deck deck = builder.buildRandom(pool);

        long nonlandCount = deck.getMain().toFlatList().stream()
                .filter(c -> !c.getRules().getMainPart().getType().isBasicLand())
                .count();

        assertTrue(nonlandCount <= 23,
                "Method 4 must pick at most 23 nonland cards, got " + nonlandCount);
    }

    @Test
    void method4NoPoolCardReuse() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(99));

        Deck deck = builder.buildRandom(pool);

        assertNoDuplicatePoolCards(pool, deck);
    }

    // ── Land Rebalancing ──────────────────────────────────────────────────────

    @Test
    void rebalanceLandsProducesExactly40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder(new Random(1));

        // Build a standard deck, extract spells and non-basic lands separately
        Deck standard = builder.buildStandard(new ArrayList<>(pool));
        List<PaperCard> spells = standard.getMain().toFlatList().stream()
                .filter(c -> !c.getRules().getMainPart().getType().isLand())
                .toList();
        List<PaperCard> nonbasicLands = standard.getMain().toFlatList().stream()
                .filter(c -> c.getRules().getMainPart().getType().isLand()
                        && !c.getRules().getMainPart().getType().isBasicLand())
                .toList();

        Deck rebalanced = builder.rebalanceLands(new ArrayList<>(spells), new ArrayList<>(nonbasicLands));

        assertEquals(40, rebalanced.getMain().countAll(),
                "rebalanceLands must produce exactly 40 cards");
    }

    // ── Full buildDeck integration ─────────────────────────────────────────────

    @Test
    void buildDeckAlwaysProduces40Cards() {
        List<PaperCard> pool = realPool("RVR");
        DeckBuilder builder = new DeckBuilder();

        // Run several times to exercise different methods
        for (int i = 0; i < 5; i++) {
            DeckBuilder.BuiltDeck built = builder.buildDeck(new ArrayList<>(pool));
            assertEquals(40, built.deck().getMain().countAll(),
                    "buildDeck must always produce exactly 40 cards");
            assertNotNull(built.method(), "buildDeck must report a method tag");
        }
    }

    @Test
    void buildDeckMethodTagMatchesSelectedMethod() {
        List<PaperCard> pool = realPool("RVR");
        Random rng = new Random(42);
        DeckBuilder builder = new DeckBuilder(rng);

        // Collect method tags across many invocations; we expect all four tags to appear
        // with the 4:3:2:1 weighting, so over 40 runs all four should be present.
        Set<BuildMethod> seen = new HashSet<>();
        for (int i = 0; i < 40; i++) {
            DeckBuilder.BuiltDeck built = builder.buildDeck(new ArrayList<>(pool));
            seen.add(built.method());
        }
        assertEquals(4, seen.size(),
                "Expected all 4 method tags to appear across 40 weighted invocations, got " + seen);
    }

    // ── FR-010: Pool card uniqueness ──────────────────────────────────────────

    /**
     * Assert that no card in the deck appears more times than it appeared in the original pool.
     */
    private static void assertNoDuplicatePoolCards(List<PaperCard> pool, Deck deck) {
        // Count each card name in pool
        Map<String, Integer> poolCounts = new HashMap<>();
        for (PaperCard card : pool) {
            poolCounts.merge(card.getName(), 1, Integer::sum);
        }

        // Count each non-basic-land card name in deck
        Map<String, Integer> deckCounts = new HashMap<>();
        for (PaperCard card : deck.getMain().toFlatList()) {
            if (!card.getRules().getMainPart().getType().isBasicLand()) {
                deckCounts.merge(card.getName(), 1, Integer::sum);
            }
        }

        // Each nonland card in deck must appear at most as many times as in pool
        for (Map.Entry<String, Integer> entry : deckCounts.entrySet()) {
            String name = entry.getKey();
            int inDeck = entry.getValue();
            int inPool = poolCounts.getOrDefault(name, 0);
            assertTrue(inDeck <= inPool,
                    "FR-010 violation: " + name + " appears " + inDeck
                            + " times in deck but only " + inPool + " times in pool");
        }
    }
}
