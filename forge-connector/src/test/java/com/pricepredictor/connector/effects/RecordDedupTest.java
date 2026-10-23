package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which records one game writes once: those that say the same thing -- the
 * same kind, about the same line, on the same board -- whatever their ids.
 */
class RecordDedupTest {

    private static int nextId;

    /** A fresh id and timestamp every time, so only what the record says can match. */
    static EffectRecord playability(PlayabilitySubkind subkind, String payload, String state) {
        nextId++;
        return EffectRecord.playability(
                        "r" + nextId, "run", "t" + nextId, "game-1", subkind,
                        AttributionMode.PATCHED)
                .state(state).payload(payload);
    }

    /** The same, for a kind that names its acting line. */
    static EffectRecord ofKind(
            RecordKind kind, ProvenanceKey line, String payload, String state) {
        nextId++;
        return new EffectRecord(
                        "r" + nextId, "run", "t" + nextId, "game-1", kind,
                        AttributionMode.PATCHED)
                .ability(new ProvenanceKey.Resolved(line, null))
                .state(state).payload(payload);
    }

    /** One of Garruk's Uprising's two {@code ChangesZone} triggers. */
    static ProvenanceKey uprisingTrigger(int index) {
        return new ProvenanceKey(
                "cardsfolder/g/garruks_uprising.txt", 0, TraitKind.TRIGGER, index, null);
    }

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
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.DECISION, "{payload}", "{life:20}")));
        assertEquals(allowed, dedup.allowDistinct(
                playability(PlayabilitySubkind.DECISION, "{payload}", board)), why);
    }

    /**
     * The legality cap keys on the snapshot as well as the payload (FR-028).
     *
     * <p>The AI's re-asks within one combat still collapse -- the board does
     * not move while it evaluates, so their snapshots render identically --
     * while the same answer on a board that moved is kept, which the
     * payload-only key of gen-1 dropped as a repeat (spec Story 3 scenario 10).
     * And one subkind's answer never coalesces with another's.
     */
    @Test
    void aLegalityAnswerCoalescesOnPayloadAndSnapshotTogether() {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.BLOCKERS, "{payload}", "{board-1}")));
        assertFalse(dedup.allowDistinct(
                playability(PlayabilitySubkind.BLOCKERS, "{payload}", "{board-1}")));
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.BLOCKERS, "{payload}", "{board-2}")));
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.ATTACKERS, "{payload}", "{board-1}")));
    }

    /**
     * Two lines on one host that say the same about one event are two records.
     *
     * <p>A trigger's snapshot names its host and the event, not which of the
     * host's triggers is evaluating, so the acting line is the only part of
     * the record that tells "when this enters" from "whenever a creature
     * enters" not firing on the same entry.
     */
    @Test
    void theActingLineIsPartOfWhatARecordSays() {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct(
                ofKind(RecordKind.TRIGGER, uprisingTrigger(0), "{not-fired}", "{board}")));
        assertTrue(dedup.allowDistinct(
                ofKind(RecordKind.TRIGGER, uprisingTrigger(1), "{not-fired}", "{board}")));
        assertFalse(dedup.allowDistinct(
                ofKind(RecordKind.TRIGGER, uprisingTrigger(1), "{not-fired}", "{board}")));
    }

    /** One kind's record never coalesces with another kind's that renders the same. */
    @Test
    void theKindIsPartOfWhatARecordSays() {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct(
                ofKind(RecordKind.TRIGGER, uprisingTrigger(0), "{payload}", "{board}")));
        assertTrue(dedup.allowDistinct(
                ofKind(RecordKind.REWRITE, uprisingTrigger(0), "{payload}", "{board}")));
    }

    /**
     * The parts are compared as parts: moving characters from the payload to
     * the snapshot makes a different record, though the two concatenate to
     * the same text.
     */
    @Test
    void partsAreNotConcatenatedIntoOneAnother() {
        RecordDedup dedup = new RecordDedup();
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.DECISION, "{ab", "c}")));
        assertTrue(dedup.allowDistinct(
                playability(PlayabilitySubkind.DECISION, "{a", "bc}")));
    }
}
