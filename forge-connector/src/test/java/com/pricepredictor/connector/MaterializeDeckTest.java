package com.pricepredictor.connector;

import forge.deck.Deck;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.DeckMaterializer.UnresolvedPolicy.FAIL;
import static com.pricepredictor.connector.DeckMaterializer.UnresolvedPolicy.WARN;
import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link DeckMaterializer#lookUp} and {@link DeckMaterializer#materialize}
 * against Forge's real card database.
 *
 * <p>The policy half — what a miss does once counted — is a pure unit test
 * in {@link DeckMaterializerTest}. Kept out of {@link MatchGeneratorTest},
 * which plays a whole Forge match first.
 */
@ExtendWith(ForgeExtension.class)
class MaterializeDeckTest {

    /** The exact spelling a coverage deck carries: converted text is lowercase. */
    private static final String CONVERTED_SPELLING = "llanowar elves";
    private static final String NOT_A_CARD = "Definitely Not A Real Card 0";
    private static final String WHY = MatchGenerator.DECKS_ONLY_UNRESOLVED;

    @Test
    void lookUpFindsAKnownNameAndNothingForAnUnknownOne() {
        assertEquals("Llanowar Elves",
                DeckMaterializer.lookUp(CONVERTED_SPELLING).orElseThrow().getName());
        assertTrue(DeckMaterializer.lookUp(NOT_A_CARD).isEmpty());
    }

    @Test
    void aConvertedLowercaseNameResolves() {
        // `collect-coverage` builds its decks from `output/cardsfolder/`,
        // whose `name:` lines are lowercase. If Forge's lookup were
        // case-sensitive, every coverage deck would be 17 basics -- so this
        // is the assertion that says strict mode is a guard rather than a
        // landmine on the ordinary path.
        Deck deck = DeckMaterializer.materialize(List.of(CONVERTED_SPELLING), FAIL, WHY);

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
                () -> DeckMaterializer.materialize(names, FAIL, WHY));

        assertTrue(ex.getMessage().contains(NOT_A_CARD),
                "the exception must name what did not resolve, got: " + ex.getMessage());
    }

    @Test
    void anOrdinarySealedDeckStillDropsTheMiss() {
        // Sealed self-play's names came out of Forge in the first place, and
        // killing an overnight run over one surprise is worse than logging it.
        Deck deck = DeckMaterializer.materialize(
                List.of(CONVERTED_SPELLING, NOT_A_CARD), WARN, WHY);

        assertEquals(1, deck.getMain().countAll());
    }

    @Test
    void afullyResolvableDeckIsUnchangedByTheCheck() {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < 17; i++) {
            names.add("Forest");
        }
        names.add(CONVERTED_SPELLING);

        Deck deck = DeckMaterializer.materialize(names, FAIL, WHY);

        assertEquals(18, deck.getMain().countAll());
    }
}
