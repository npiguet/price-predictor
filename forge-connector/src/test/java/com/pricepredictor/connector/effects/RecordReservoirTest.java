package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which of a game's mana activations survive to be written. */
class RecordReservoirTest {

    @Test
    void theFirstActivationsFillTheReservoir() {
        RecordReservoir reservoir = new RecordReservoir(3, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(1, reservoir.slot("Mana$ G"));
        assertEquals(2, reservoir.slot("Mana$ G"));
    }

    @Test
    void theReservoirIsPerUniqueManaAbilityText() {
        // A basic land taps a dozen times a game; a rare mana ability must not
        // be starved by it.
        RecordReservoir reservoir = new RecordReservoir(1, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(0, reservoir.slot("Mana$ Any | Amount$ 2"));
    }

    @Test
    void aCapOfZeroRecordsNothing() {
        assertEquals(-1, new RecordReservoir(0, new Random(0)).slot("Mana$ G"));
    }

    /**
     * Every activation is equally likely to survive.
     *
     * <p>The property the reservoir exists for. Taking the first would make
     * every mana record describe turn one against an empty board, because that
     * is when a land's first tap happens — a bias introduced before collection
     * has even started, into the one input the model conditions on.
     */
    @Test
    void everyActivationIsEquallyLikelyToSurvive() {
        int activations = 8;
        int trials = 20000;
        int[] survivors = new int[activations];
        for (int trial = 0; trial < trials; trial++) {
            RecordReservoir reservoir = new RecordReservoir(1, new Random(GameRecording.scramble(trial)));
            int held = -1;
            for (int i = 0; i < activations; i++) {
                if (reservoir.slot("Mana$ G") >= 0) {
                    held = i;
                }
            }
            survivors[held]++;
        }
        double expected = (double) trials / activations;
        for (int i = 0; i < activations; i++) {
            assertTrue(
                    Math.abs(survivors[i] - expected) < expected * 0.15,
                    "activation " + i + " survived " + survivors[i]
                            + " times, expected about " + expected);
        }
    }

    private static EffectRecord record() {
        return RecordDedupTest.ofKind(RecordKind.TRIGGER,
                new ProvenanceKey("cardsfolder/p/paralyze.txt", 0, TraitKind.TRIGGER, 0, null),
                "{}", "{}");
    }

    /** Held records are drained once, in slot order, and the reservoir starts over. */
    @Test
    void heldRecordsAreDrainedOnce() {
        RecordReservoir reservoir = new RecordReservoir(2, new Random(0));
        EffectRecord first = record();
        EffectRecord second = record();
        reservoir.hold("line", reservoir.slot("line"), first);
        reservoir.hold("line", reservoir.slot("line"), second);
        assertEquals(2, reservoir.held("line"));

        assertEquals(java.util.List.of(first, second), reservoir.drain());
        assertEquals(java.util.List.of(), reservoir.drain());
        assertEquals(0, reservoir.slot("line"), "the next game's first activation fills again");
    }

    /**
     * A slot whose record was never built (its builder threw) leaves the
     * reservoir usable: the next record is held, not an IndexOutOfBounds
     * for the rest of the game (C2 #13, V2 #17).
     */
    @Test
    void aSlotWhoseRecordNeverCameLeavesTheReservoirUsable() {
        RecordReservoir reservoir = new RecordReservoir(3, new Random(0));
        assertEquals(0, reservoir.slot("line"));
        // The record for slot 0 threw while it was built: never held.
        EffectRecord next = record();
        reservoir.hold("line", reservoir.slot("line"), next);

        assertEquals(java.util.List.of(next), reservoir.drain());
    }

    /** A draw is made against the count the activation would make, and counts nothing itself. */
    @Test
    void aDrawCountsNothingUntilTheActivationIsCounted() {
        RecordReservoir reservoir = new RecordReservoir(2, new Random(0));
        for (int i = 0; i < 10; i++) {
            assertEquals(0, reservoir.drawFor("line"));
        }
        reservoir.count("line");
        assertEquals(1, reservoir.drawFor("line"));
        reservoir.count("line");
        assertTrue(reservoir.drawFor("line") < 2, "a third activation draws one of two slots or none");
    }
}
