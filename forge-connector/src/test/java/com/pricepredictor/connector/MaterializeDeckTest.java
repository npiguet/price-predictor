package com.pricepredictor.connector;

import forge.deck.Deck;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link MatchGenerator#materializeDeck} against Forge's real card database.
 *
 * <p>Tagged {@code integration} because resolving a card name needs
 * {@code FModel}; the policy half — what a miss does once counted — is a pure
 * unit test in {@link MatchGeneratorRoutingTest}. Kept in its own class rather
 * than in {@link MatchGeneratorTest} so it can be run on its own without
 * playing whole Forge matches first.
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class MaterializeDeckTest {

    /** The exact spelling a coverage deck carries: converted text is lowercase. */
    private static final String CONVERTED_SPELLING = "llanowar elves";
    private static final String NOT_A_CARD = "Definitely Not A Real Card 0";

    @Test
    void aConvertedLowercaseNameResolves() {
        // `collect-coverage` builds its decks from `output/cardsfolder/`,
        // whose `name:` lines are lowercase. If Forge's lookup were
        // case-sensitive, every coverage deck would be 17 basics -- so this
        // is the assertion that says strict mode is a guard rather than a
        // landmine on the ordinary path.
        Deck deck = MatchGenerator.materializeDeck(List.of(CONVERTED_SPELLING), true);

        assertEquals(1, deck.getMain().countAll(),
                "Forge did not resolve the converted tree's lowercase spelling");
    }

    @Test
    void aDecksOnlyDeckThrowsOnAnUnresolvableName() {
        // The finding: `if (card != null) main.add(card)` dropped a miss in
        // silence. A coverage or variant deck then materialized as 17 basics,
        // a game played, a progress line was appended, records were written --
        // and the card the round exists to reach never entered a game.
        List<String> names = new ArrayList<>(List.of(CONVERTED_SPELLING, NOT_A_CARD));

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> MatchGenerator.materializeDeck(names, true));

        assertTrue(ex.getMessage().contains(NOT_A_CARD),
                "the exception must name what did not resolve, got: " + ex.getMessage());
    }

    @Test
    void anOrdinarySealedDeckStillDropsTheMiss() {
        // Sealed self-play's names came out of Forge in the first place, and
        // killing an overnight run over one surprise is worse than logging it.
        Deck deck = MatchGenerator.materializeDeck(
                List.of(CONVERTED_SPELLING, NOT_A_CARD), false);

        assertEquals(1, deck.getMain().countAll());
    }

    @Test
    void afullyResolvableDeckIsUnchangedByTheCheck() {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < 17; i++) {
            names.add("Forest");
        }
        names.add(CONVERTED_SPELLING);

        Deck deck = MatchGenerator.materializeDeck(names, true);

        assertEquals(18, deck.getMain().countAll());
    }
}
