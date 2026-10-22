package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which records one game writes once: those that say the same thing on the same board. */
class RecordDedupTest {

    /**
     * A decision is keyed by what it says and the board it was asked on: the
     * same answer on an unchanged board is written once, and a board that
     * moved gets its own record.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a decision on an unchanged board is written once, '{life:20}', false",
            "a board that moved gets its own decision record, '{life:19}', true"})
    void aDecisionIsKeyedByItsPayloadAndBoard(String why, String board, boolean allowed) {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct("decision", "{payload}", "{life:20}"));
        assertEquals(allowed, dedup.allowDistinct("decision", "{payload}", board), why);
    }

    /**
     * The legality cap keys on the snapshot as well as the payload (FR-028).
     *
     * <p>The AI's re-asks within one combat still collapse -- the board does
     * not move while it evaluates, so their snapshots render identically --
     * while the same answer on a board that moved is kept, which the
     * payload-only key of gen-1 dropped as a repeat (spec Story 3 scenario 10).
     */
    @Test
    void aLegalityAnswerCoalescesOnPayloadAndSnapshotTogether() {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct("blockers", "{payload}", "{board-1}"));
        assertFalse(dedup.allowDistinct("blockers", "{payload}", "{board-1}"));
        assertTrue(dedup.allowDistinct("blockers", "{payload}", "{board-2}"));
        assertTrue(dedup.allowDistinct("attackers", "{payload}", "{board-1}"));
    }
}
