package com.pricepredictor.connector;

import forge.card.MagicColor;
import forge.gamemodes.limited.DeckColors;
import forge.gamemodes.limited.DraftedColorCommitment;
import forge.item.PaperCard;
import forge.model.FModel;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests that replaying a seat's picks recovers the colour pair it committed to.
 *
 * <p>This is the rule {@code forge-native} decks are built against, so it has to match
 * {@code LimitedPlayerAI}'s: commit as you pick, at most two colours, colourless picks
 * never consuming a slot.
 *
 * <p>Needs real {@link PaperCard} objects, so tagged as "integration".
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class DraftedColorCommitmentTest {

    private static List<PaperCard> picks(String... names) {
        List<PaperCard> cards = new ArrayList<>(names.length);
        for (String name : names) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            assertTrue(card != null, "test fixture card not found: " + name);
            cards.add(card);
        }
        return cards;
    }

    private static String colorsOf(List<PaperCard> picks) {
        DeckColors colors = DraftedColorCommitment.replay(picks);
        StringBuilder sb = new StringBuilder();
        for (byte color : MagicColor.WUBRG) {
            if (colors.getChosenColors().hasAnyColor(color)) {
                sb.append(MagicColor.toShortString(color));
            }
        }
        return sb.toString();
    }

    @Test
    void commitsToTheColoursOfTheFirstTwoColouredPicks() {
        assertEquals("UR", colorsOf(picks("Lightning Bolt", "Counterspell")));
    }

    @Test
    void oneColourIsCommittedWhenEveryPickSharesIt() {
        assertEquals("R", colorsOf(picks("Lightning Bolt", "Shock")));
    }

    @Test
    void stopsAtTwoColours() {
        assertEquals(
                "UR",
                colorsOf(picks("Lightning Bolt", "Counterspell", "Llanowar Elves")));
    }

    @Test
    void aGoldPickCommitsBothOfItsColoursAtOnce() {
        assertEquals("WR", colorsOf(picks("Lightning Helix")));
    }

    @Test
    void colourlessPicksDoNotConsumeASlot() {
        assertEquals(
                "UR",
                colorsOf(picks("Ornithopter", "Lightning Bolt", "Counterspell")));
    }

    /**
     * The property the fix turns on: the pool alone does not determine the colours, so
     * {@code buildDrafted} must be handed its picks in pick order.
     */
    @Test
    void pickOrderDecidesWhichTwoColoursWin() {
        assertEquals("UR", colorsOf(picks("Lightning Bolt", "Counterspell", "Llanowar Elves")));
        assertEquals("UG", colorsOf(picks("Llanowar Elves", "Counterspell", "Lightning Bolt")));
    }

    @Test
    void anAllColourlessPoolCommitsToNothing() {
        assertEquals("", colorsOf(picks("Ornithopter")));
    }
}
