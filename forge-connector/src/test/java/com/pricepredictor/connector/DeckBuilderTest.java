package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.item.PaperCard;
import forge.model.FModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DeckBuilder's four construction methods, land rebalancing,
 * and pool-card uniqueness guarantee (FR-010).
 */
@ExtendWith(ForgeExtension.class)
class DeckBuilderTest {

    /**
     * Generate a realistic pool from 6 boosters of the given set (no basic lands).
     */
    private static List<PaperCard> realPool(String setCode) {
        return new PoolGenerator().openSinglePool(
                BoosterTemplates.forSet(setCode).orElseThrow(), HoldoutCards.NONE);
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
     * What this method exists for. {@code SealedDeckBuilder} would re-derive the
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

    // ── The four construction methods ─────────────────────────────────────────

    private record Method(String name, long seed, BiFunction<DeckBuilder, List<PaperCard>, Deck> build) {
        @Override
        public String toString() {
            return name;
        }
    }

    static Stream<Method> methods() {
        return Stream.of(
                new Method("method 1: standard build", 42, DeckBuilder::buildStandard),
                new Method("method 2: 3 swaps", 42, (b, pool) -> b.buildWithSwaps(pool, 3)),
                new Method("method 3: 8 swaps", 7, (b, pool) -> b.buildWithSwaps(pool, 8)),
                new Method("method 4: random", 99, DeckBuilder::buildRandom));
    }

    @ParameterizedTest
    @MethodSource("methods")
    void everyMethodProduces40CardsWithoutReusingPoolCards(Method method) {
        List<PaperCard> pool = realPool("RVR");

        Deck deck = method.build().apply(new DeckBuilder(new Random(method.seed())), pool);

        assertEquals(40, deck.getMain().countAll(), method.name() + " deck must be exactly 40 cards");
        assertNoDuplicatePoolCards(pool, deck);
    }

    @Test
    void method2SwapsExactly3NonlandCards() {
        // Forge's standard build is not repeatable, so it is fixed here to leave the seeded
        // swaps as the only difference between the two decks. The pool holds one copy of each
        // spell so a swap always has a differing replacement, and no land, which could have
        // none left to swap in.
        Map<String, PaperCard> distinctSpells = new LinkedHashMap<>();
        for (PaperCard card : realPool("RVR")) {
            if (!card.getRules().getMainPart().getType().isLand()) {
                distinctSpells.putIfAbsent(card.getName(), card);
            }
        }
        List<PaperCard> pool = new ArrayList<>(distinctSpells.values());
        Deck standardDeck = new DeckBuilder(new Random(42)).buildStandard(new ArrayList<>(pool));
        DeckBuilder builder = new DeckBuilder(new Random(42)) {
            @Override
            Deck buildStandard(List<PaperCard> ignored) {
                return standardDeck;
            }
        };

        Deck swappedDeck = builder.buildWithSwaps(new ArrayList<>(pool), 3);

        List<PaperCard> removed = nonbasics(standardDeck);
        nonbasics(swappedDeck).forEach(removed::remove);
        assertEquals(3, removed.size(),
                "3 swaps replace exactly 3 of the standard deck's nonbasic cards, got " + removed);
        // The other direction too: a nonbasic swapped for a basic land would
        // leave the count above unchanged and bring in one card too few here.
        // Basics are left out because the builder re-balances them for the
        // swapped deck's colours.
        List<PaperCard> added = nonbasics(swappedDeck);
        nonbasics(standardDeck).forEach(added::remove);
        assertEquals(3, added.size(), "3 swaps bring in exactly 3 nonbasic cards, got " + added);
        assertTrue(added.stream().noneMatch(c -> c.getRules().getMainPart().getType().isLand()),
                "every card a swap brings in is a nonland card, got " + added);
    }

    /** The deck's nonbasic cards as a mutable list, duplicates repeated. */
    private static List<PaperCard> nonbasics(Deck deck) {
        return deck.getMain().toFlatList().stream()
                .filter(c -> !c.getRules().getMainPart().getType().isBasicLand())
                .collect(Collectors.toCollection(ArrayList::new));
    }

    // ── Full buildDeck integration ─────────────────────────────────────────────

    @Test
    void buildDeckMakes40CardsAndItsMethodTagMatchesTheRolledMethodAndEveryMethodOccurs() {
        List<PaperCard> pool = realPool("RVR");

        // The builder's first draw is the method roll, so a mirror Random predicts the tag.
        // Seeds come from a master Random because consecutive seeds give near-identical first
        // doubles; 40 of them at weights 4:3:2:1 reach all four methods.
        Random seeds = new Random(42);
        Set<BuildMethod> seen = new HashSet<>();
        for (int i = 0; i < 40; i++) {
            long seed = seeds.nextLong();
            DeckBuilder.BuiltDeck built =
                    new DeckBuilder(new Random(seed)).buildDeck(new ArrayList<>(pool));
            BuildMethod rolled = BuildMethod.forRoll(new Random(seed).nextDouble());
            assertEquals(40, built.deck().getMain().countAll(),
                    "buildDeck must always produce exactly 40 cards, seed " + seed);
            assertEquals(rolled, built.method(), "seed " + seed);
            if (rolled == BuildMethod.RANDOM) {
                assertTrue(nonbasics(built.deck()).size() <= DeckBuilder.RANDOM_DECK_SPELLS,
                        "a random-tagged deck holds at most 23 nonbasic cards");
            }
            seen.add(built.method());
        }
        assertEquals(Set.of(BuildMethod.values()), seen);
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
