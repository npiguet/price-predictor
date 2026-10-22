package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The worker's rewrite tally, and the summary it prints. */
@ExtendWith(ForgeExtension.class)
class RewriteTallyTest {

    /**
     * A record with no outgoing names the parameters that moved under it.
     *
     * <p>The count alone says the channel filled; this says whether the rows in
     * it are as thin as they look or whether the normaliser cannot express what
     * the replacement changed, and it hands over the key that would close the
     * gap.
     */
    @Test
    void aRecordWithNoOutgoingNamesTheParametersTheNormaliserDidNotRead() {
        RewriteTally tally = new RewriteTally();
        Card card = TestCards.build("Mountain");
        tally.record("Moved", "replaced", false, false,
                Map.of("Card", card, "LibraryPosition", 0),
                Map.of("Card", card, "LibraryPosition", -1));

        String summary = tally.summary();
        assertTrue(summary.contains("Moved.LibraryPosition=1"), summary);
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
    }

    /** And one that moved nothing at all is not counted against the reader. */
    @Test
    void aRecordThatMovedNothingIsNoNormaliserGap() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Untap", "skipped", false, false, same, same);

        String summary = tally.summary();
        assertTrue(summary.contains(
                "0 carried no outgoing while a raw parameter moved: none"), summary);
    }

    /**
     * The headline is the volume and the results, because that is the question.
     *
     * <p>The channel held 34 records in 1.88M against a 7% share of the
     * mixture, and the operator needs to know in the first hour whether it
     * filled. A breakdown that is all {@code ?} is a Forge jar whose listener
     * predates the result argument, which reads identically to a wired one on
     * every other line of the summary.
     */
    @Test
    void theSummaryLeadsWithTheVolumeAndTheResults() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", "updated", true, true, same, same);
        for (int i = 0; i < 9; i++) {
            tally.record("Moved", "not_replaced", false, false, same, same);
        }

        String summary = tally.summary();
        assertTrue(summary.contains(
                "10 records, 1 rewrote a parameter in place (10.0%), "
                        + "1 named the ability that ran instead"), summary);
        assertTrue(summary.contains("by result: not_replaced=9, updated=1"), summary);
        assertTrue(summary.contains("by mode and result: Moved.not_replaced=9"),
                summary);
    }

    /** A listener that never learnt the result says so rather than guessing. */
    @Test
    void aResultlessRecordIsItsOwnRowInTheBreakdown() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", null, false, false, same, same);

        assertTrue(tally.summary().contains("by result: ?=1"), tally.summary());
    }

    /**
     * A worker that saw no replacement says nothing.
     *
     * <p>Whether the hook is installed at all is {@code PatchHooks.report()}'s
     * line, and repeating it once a game would be noise in every worker log.
     */
    @Test
    void aGameWithNoReplacementPrintsNothing() {
        RewriteTally tally = new RewriteTally();
        java.io.PrintStream out = System.out;
        java.io.ByteArrayOutputStream captured = new java.io.ByteArrayOutputStream();
        System.setOut(new java.io.PrintStream(captured));
        try {
            tally.endOfGame();
        } finally {
            System.setOut(out);
        }
        assertEquals("", captured.toString());
    }

    /**
     * The tally separates an unreadable change from no change at all.
     *
     * <p>The smoke run reported 110 drops, {@code Moved=91} among them, and
     * {@code 69 dropped with no raw parameter difference at all} — three lines
     * needing arithmetic across them to reach the one fact that decides what to
     * do next: 62 of Moved's 91 changed nothing whatever, while 29 changed only
     * the entry-counter table, which {@code zone_change} has no slot for. Those
     * rows are all written now, so the question is no longer which to drop but
     * which of them a reader is being short-changed on.
     */
    @Test
    void theTallySeparatesAnUnreadableChangeFromNoChange() {
        RewriteTally tally = new RewriteTally();

        tally.record("Moved", "replaced", false, true,
                Map.of("Destination", "Battlefield"),
                Map.of("Destination", "Battlefield"));
        tally.record("Moved", "replaced", false, true,
                Map.of("CounterMap", "{}"),
                Map.of("CounterMap", "{P1P1=1}"));

        String summary = tally.summary();
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
        assertTrue(summary.contains("of those, by mode: Moved=1"), summary);
        assertTrue(summary.contains("Moved.CounterMap=1"), summary);
    }
}
