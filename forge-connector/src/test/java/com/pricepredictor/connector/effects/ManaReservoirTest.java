package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which of a game's mana activations survive to be written. */
class ManaReservoirTest {

    @Test
    void theFirstActivationsFillTheReservoir() {
        ManaReservoir reservoir = new ManaReservoir(3, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(1, reservoir.slot("Mana$ G"));
        assertEquals(2, reservoir.slot("Mana$ G"));
    }

    @Test
    void theReservoirIsPerUniqueManaAbilityText() {
        // A basic land taps a dozen times a game; a rare mana ability must not
        // be starved by it.
        ManaReservoir reservoir = new ManaReservoir(1, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(0, reservoir.slot("Mana$ Any | Amount$ 2"));
    }

    @Test
    void aCapOfZeroRecordsNothing() {
        assertEquals(-1, new ManaReservoir(0, new Random(0)).slot("Mana$ G"));
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
            ManaReservoir reservoir = new ManaReservoir(1, new Random(GameRecording.scramble(trial)));
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
}
