package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.DeckMaterializer.UnresolvedPolicy.FAIL;
import static com.pricepredictor.connector.DeckMaterializer.UnresolvedPolicy.WARN;
import static org.junit.jupiter.api.Assertions.*;

/**
 * What {@link DeckMaterializer} does with names that missed, and the
 * {@link CardNameList} field format — the halves that need no Forge card
 * database. Resolution itself is in {@link MaterializeDeckTest}.
 */
class DeckMaterializerTest {

    private static final String WHY = "Playing it short would corrupt the test.";

    // ── unresolved card names ─────────────────────────────────────────────

    @Test
    void nothingUnresolvedSaysNothing() throws Throwable {
        String printed = StdStreams.captureStderr(() -> {
            DeckMaterializer.reportUnresolved(List.of(), 40, FAIL, WHY);
            DeckMaterializer.reportUnresolved(List.of(), 40, WARN, WHY);
        });

        assertEquals("", printed);
    }

    @Test
    void failThrowsNamingWhatMissedAndWhy() {
        // A coverage or variant deck's names come from output/cardsfolder/
        // and from VariantRegistry, not from Forge. A name Forge misses is
        // dropped, the deck materializes as 17 basics, a game plays, a
        // progress line is appended, records are written -- and the card the
        // round exists to reach never entered a game.
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> DeckMaterializer.reportUnresolved(
                        List.of("Bolt Variant 0", "Bolt Variant 1"), 40, FAIL, WHY));

        assertTrue(ex.getMessage().contains("Bolt Variant 0"),
                "the exception must name the cards that did not resolve, got: "
                        + ex.getMessage());
        assertTrue(ex.getMessage().contains("Bolt Variant 1"));
        assertTrue(ex.getMessage().contains("2 of 40"),
                "the exception must say how many of how many missed, got: "
                        + ex.getMessage());
        assertTrue(ex.getMessage().contains(WHY),
                "the exception must say what the short deck would corrupt, got: "
                        + ex.getMessage());
    }

    @Test
    void warnOnlyLogsTheMiss() throws Throwable {
        // Sealed self-play's names came out of Forge in the first place, so a
        // miss there is a surprise rather than a broken round -- and killing
        // a long self-play run over one is worse than saying so.
        String printed = StdStreams.captureStderr(() -> DeckMaterializer.reportUnresolved(
                List.of("Whatever"), 40, WARN, WHY));

        assertTrue(printed.startsWith("WARNING: "), printed);
        assertTrue(printed.contains("1 of 40"), printed);
        assertTrue(printed.contains("Whatever"),
                "the warning must name the card that did not resolve, got: " + printed);
    }

    @Test
    void aLongUnresolvedListIsCappedRatherThanDumped() {
        List<String> many = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            many.add("Missing " + i);
        }

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> DeckMaterializer.reportUnresolved(many, 40, FAIL, WHY));

        assertTrue(ex.getMessage().contains("40 of 40"));
        assertFalse(ex.getMessage().contains("Missing 39"),
                "a 40-name dump is not a message anyone reads");
        assertTrue(ex.getMessage().contains("more"),
                "a capped list must say it was capped, got: " + ex.getMessage());
    }

    // ── the pipe-separated field ──────────────────────────────────────────

    @Test
    void anEmptyFieldIsNoCards() {
        // cards-played.txt writes an empty field for a side that played
        // nothing; reading it back as one empty name would invent a card.
        assertEquals(List.of(), CardNameList.parse(""));
        assertEquals("", CardNameList.format(List.of()));
    }

    @Test
    void namesRoundTripInOrderWithDuplicates() {
        List<String> names = List.of("Forest", "Ach! Hans, Run!", "Forest");

        assertEquals("Forest|Ach! Hans, Run!|Forest", CardNameList.format(names));
        assertEquals(names, CardNameList.parse(CardNameList.format(names)));
    }
}
