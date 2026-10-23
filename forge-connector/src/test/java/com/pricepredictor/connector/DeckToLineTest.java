package com.pricepredictor.connector;

import forge.deck.Deck;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Arrays;
import java.util.List;

import static com.pricepredictor.connector.DeckMaterializer.UnresolvedPolicy.FAIL;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link DeckBuilderMain#deckToLine} against Forge's real card database: the
 * line Python reads back as one deck per pool.
 */
@ExtendWith(ForgeExtension.class)
class DeckToLineTest {

    @Test
    void aDeckIsOnePipeSeparatedLineWithDuplicatesRepeated() {
        Deck deck = DeckMaterializer.materialize(
                List.of("Mountain", "Lightning Bolt", "Mountain"), FAIL, "test");

        String line = DeckBuilderMain.deckToLine(deck);

        // Forge's card pool does not promise insertion order, so compare the
        // multiset of names.
        List<String> names = Arrays.asList(line.split("\\|", -1));
        assertEquals(List.of("Lightning Bolt", "Mountain", "Mountain"),
                names.stream().sorted().toList());
    }
}
