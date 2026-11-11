package com.pricepredictor.connector;

import forge.item.PaperCard;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Random;
import java.util.Set;

import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Where {@link MatchGenerator} takes a match's decks from — its own pool, or a
 * decks file — asked directly, without playing a match.
 */
@ExtendWith(ForgeExtension.class)
class MatchGeneratorDeckSourcingTest {

    @Test
    void generatePoolOmitsExcludedCardsInTheCollectionPath() {
        // The pool every collected effect record is played from. `generate-pools
        // --exclude-cards` never reaches here: MatchGenerator makes its own pool
        // per match, so a depleted collection run needs the exclusion on the
        // worker, not on the pool file.
        List<String> reference = new PoolGenerator().generate("RVR", 1, HoldoutCards.NONE).get(0);
        Set<String> excluded = Set.copyOf(reference.subList(0, 10));

        MatchGenerator generator = MatchGenerator.builder(List.of("RVR"), RUN_ID)
                .random(new Random(42)).excludedCards(HoldoutCards.of(excluded)).build();

        List<PaperCard> pool = generator.generatePool("RVR");

        assertTrue(pool.size() >= 70, "Depleted pool too small: " + pool.size());
        for (PaperCard card : pool) {
            assertFalse(excluded.contains(card.getName()),
                    "Excluded card reached a collected game: " + card.getName());
        }
    }

    @Test
    void decksOnlyAcceptsAMirrorInsteadOfReachingForge() {
        // What this guards against: a coverage round's live-card pool
        // shrinks monotonically as cards are satisfied (deck_weights in
        // collect_coverage.py drops every satisfied card), so a round with
        // exactly one card left samples 23 nonlands with replacement from a
        // single candidate -- every deck that round builds is byte-identical,
        // deterministically. Under decksOnly, pickDeckB's non-mirror search
        // then always comes up empty. Falling through to forgeBuilt("COVERAGE")
        // -- a sentinel set code that resolves against no real Forge edition --
        // would throw on every single match attempt, since decksOnly forces
        // every attempt down this branch: no match would ever complete.
        GeneratedDecksIndex.GeneratedDeck deck = new GeneratedDecksIndex.GeneratedDeck(
                "coverage", "COVERAGE", List.of("Llanowar Elves"));
        GeneratedDecksIndex mirrorOnly = new GeneratedDecksIndex(List.of(deck, deck));
        // Both sides point at the same index, which is what
        // CollectorSupervisor does (one decks file, both --side-*-decks
        // paths) and what decksOnly requires: a Forge-built side does not
        // fail a decks-only match, it quietly turns it back into sealed
        // self-play. pickDeckB never reads side A -- only pickDeckA does --
        // so this changes nothing about the branch under test, which is
        // called directly below with an explicit set code and deck-A card list.
        MatchGenerator.DeckSource decks = new MatchGenerator.DeckSource.FileSampled(mirrorOnly);
        MatchGenerator generator = MatchGenerator.builder(List.of("RVR"), RUN_ID)
                .sideA(decks).sideB(decks, 1).random(new Random(7)).decksOnly(true).build();

        MatchGenerator.DeckSelection b = assertDoesNotThrow(
                () -> generator.pickDeckB("COVERAGE", List.of("Llanowar Elves")),
                "pickDeckB must accept a mirror rather than reach Forge's "
                        + "set-based builder for a decksOnly round");

        assertEquals("COVERAGE", b.setCode());
        assertEquals(List.of("Llanowar Elves"), b.cardNames());
        assertNotNull(b.deck(), "the mirror deck must still materialize");
    }

    /**
     * A steered deck's rule reaches the game (FR-058d): the match is played
     * with both decks' rules, so its records can carry them as
     * {@code aimed_at}.
     */
    @Test
    void bothDecksRulesReachThePlayedMatch() {
        GeneratedDecksIndex.GeneratedDeck steered = new GeneratedDecksIndex.GeneratedDeck(
                "variant", "COVERAGE", List.of("Llanowar Elves", "Forest"), "704.5f");
        GeneratedDecksIndex.GeneratedDeck plain = new GeneratedDecksIndex.GeneratedDeck(
                "variant", "COVERAGE", List.of("Llanowar Elves", "Plains"));
        MatchGenerator.DeckSource decks = new MatchGenerator.DeckSource.FileSampled(
                new GeneratedDecksIndex(List.of(steered, plain)));
        List<com.pricepredictor.connector.effects.DeckRules> played = new java.util.ArrayList<>();
        GamePlayer capturing = new GamePlayer(1, null,
                com.pricepredictor.connector.effects.CollectionCaps.defaults()) {
            @Override
            public PlayedMatch playMatch(forge.deck.Deck deckA, forge.deck.Deck deckB,
                    com.pricepredictor.connector.effects.DeckRules rules) {
                played.add(rules);
                return new PlayedMatch(
                        List.of(new GameOutcome(Side.A, Side.A, List.of(), List.of())), 0, false);
            }
        };
        MatchGenerator generator = MatchGenerator.builder(List.of("RVR"), RUN_ID)
                .sideA(decks).sideB(decks, 1).random(new Random(3)).decksOnly(true)
                .gamePlayer(capturing).build();

        generator.generateMatch();

        assertEquals(1, played.size());
        // Two decks of which exactly one is steered, never a mirror of each other.
        assertEquals(List.of("704.5f"),
                played.get(0).aimedAt(java.util.Optional.of(List.of())));
    }
}
