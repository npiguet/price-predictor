package com.pricepredictor.connector;

import forge.item.PaperCard;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic-land sourcing and the size guard of {@link DeckBuilder#rebalanceLands}.
 *
 * <p>Tagged {@code integration} because resolving a basic land needs Forge's card database.
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class DeckBuilderLandsTest {

    private static final String SET_WITHOUT_PRINTINGS = "ZZZ-NO-SUCH-SET";

    @Test
    void basicLandFallsBackToDefaultPrintingWhenSetHasNone() {
        PaperCard land = DeckBuilder.basicLand("Plains", SET_WITHOUT_PRINTINGS);

        assertEquals("Plains", land.getName());
    }

    @Test
    void basicLandThatDoesNotExistNamesLandAndSet() {
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> DeckBuilder.basicLand("Not A Basic Land", "GRN"));

        assertTrue(e.getMessage().contains("Not A Basic Land"), e.getMessage());
        assertTrue(e.getMessage().contains("GRN"), e.getMessage());
    }

    @Test
    void moreSpellsThanDeckSizeIsRejectedWithCounts() {
        List<PaperCard> spells = nonlandSpells(41);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new DeckBuilder().rebalanceLands(spells, List.of()));

        assertTrue(e.getMessage().contains("41"), e.getMessage());
    }

    private static List<PaperCard> nonlandSpells(int count) {
        List<PaperCard> spells = new ArrayList<>();
        while (spells.size() < count) {
            for (PaperCard card : new PoolGenerator().openPool(
                    BoosterTemplates.forSet("RVR").orElseThrow(), HoldoutCards.NONE)) {
                if (!card.getRules().getMainPart().getType().isLand()) {
                    spells.add(card);
                }
            }
        }
        return spells.subList(0, count);
    }
}
