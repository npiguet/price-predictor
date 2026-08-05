package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the {@code -Ddraft.required.agent} rule (spec 021 FR-003).
 *
 * <p>Deliberately <em>not</em> tagged "integration": the rule is pure — it needs
 * no Forge — and {@code mvn test} excludes the integration group, so a test
 * placed alongside the Forge-driven cases in {@link DraftWorkerMainTest} would
 * never run in the default build.
 */
class DraftWorkerRequiredAgentTest {

    private static String[] draw(String... agents) {
        return agents.clone();
    }

    @Test
    void overwritesExactlyOneSeatWhenTheDrawContainsNone() {
        String[] agents = draw(
                "forge-full", "forge-full", "gen-1", "forge-r30",
                "gen-1", "forge-full", "forge-r100", "gen-1");
        String[] before = agents.clone();

        DraftWorkerMain.forceRequiredAgent(agents, "gen-3", new Random(7));

        long carrying = Arrays.stream(agents).filter("gen-3"::equals).count();
        assertEquals(1, carrying, "exactly one seat should be forced to the learner");

        // Every other seat is untouched — the rest of the draw is preserved.
        int changed = 0;
        for (int s = 0; s < agents.length; s++) {
            if (!agents[s].equals(before[s])) {
                changed++;
                assertEquals("gen-3", agents[s]);
            }
        }
        assertEquals(1, changed, "only the forced seat should differ");
    }

    @Test
    void leavesADrawThatAlreadyCarriesTheLabelUnchanged() {
        String[] agents = draw(
                "forge-full", "gen-3", "gen-1", "forge-r30",
                "gen-1", "forge-full", "forge-r100", "gen-1");
        String[] before = agents.clone();

        DraftWorkerMain.forceRequiredAgent(agents, "gen-3", new Random(7));

        assertArrayEquals(before, agents);
    }

    @Test
    void aDrawAlreadyFullOfTheLabelIsUnchanged() {
        String[] agents = draw("gen-3", "gen-3", "gen-3", "gen-3");
        DraftWorkerMain.forceRequiredAgent(agents, "gen-3", new Random(1));
        assertArrayEquals(draw("gen-3", "gen-3", "gen-3", "gen-3"), agents);
    }

    @Test
    void absentOrBlankLabelLeavesSamplingByteForByteAsBefore() {
        String[] agents = draw("forge-full", "gen-1", "forge-r30", "forge-full");
        String[] before = agents.clone();

        for (String label : new String[] {null, "", "   "}) {
            // A fresh Random with the same seed must be left un-advanced, so a
            // run without the property draws exactly the sequence it always did.
            Random random = new Random(99);
            DraftWorkerMain.forceRequiredAgent(agents, label, random);
            assertArrayEquals(before, agents, "seats unchanged for label " + label);
            assertEquals(new Random(99).nextInt(1000), random.nextInt(1000),
                    "the random stream must not be advanced for label " + label);
        }
    }

    @Test
    void theForcedSeatIsChosenUniformly() {
        Set<Integer> seatsHit = new HashSet<>();
        Random random = new Random(3);
        for (int trial = 0; trial < 400; trial++) {
            String[] agents = draw(
                    "forge-full", "forge-full", "forge-full", "forge-full",
                    "forge-full", "forge-full", "forge-full", "forge-full");
            DraftWorkerMain.forceRequiredAgent(agents, "gen-3", random);
            for (int s = 0; s < agents.length; s++) {
                if ("gen-3".equals(agents[s])) {
                    seatsHit.add(s);
                }
            }
        }
        assertEquals(8, seatsHit.size(), "every seat index should be reachable");
    }
}
