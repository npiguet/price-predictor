package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link AgentMix}; pure, so it runs in the default build. */
class AgentMixTest {

    @Test
    void samplesByWeight() {
        AgentMix mix = AgentMix.parse("forge-full:9,forge-r100:1");
        Random random = new Random(42);
        int full = 0;
        int trials = 5000;
        for (int i = 0; i < trials; i++) {
            if (mix.sample(random).equals("forge-full")) {
                full++;
            }
        }
        double share = (double) full / trials;
        assertTrue(share > 0.85 && share < 0.95, "forge-full share ~0.9, got " + share);
    }

    @Test
    void rejectsMalformed() {
        // Every malformed mix is an IllegalArgumentException, the one type
        // DraftWorkerMain turns into a bad-configuration exit.
        assertThrows(IllegalArgumentException.class, () -> AgentMix.parse("forge-full"));
        assertThrows(IllegalArgumentException.class, () -> AgentMix.parse("forge-full:0"));
        assertThrows(IllegalArgumentException.class, () -> AgentMix.parse("forge-full:x"));
    }
}
