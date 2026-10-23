package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
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

    @Test
    void overwritesExactlyOneSeatWhenTheDrawContainsNone() {
        List<String> before = List.of(
                "forge-full", "forge-full", "gen-1", "forge-r30",
                "gen-1", "forge-full", "forge-r100", "gen-1");

        List<String> agents = DraftWorkerMain.withRequiredAgent(before, "gen-3", new Random(7));

        assertEquals(1, agents.stream().filter("gen-3"::equals).count(),
                "exactly one seat should be forced to the learner");
        // Every other seat is untouched — the rest of the draw is preserved.
        int changed = 0;
        for (int s = 0; s < agents.size(); s++) {
            if (!agents.get(s).equals(before.get(s))) {
                changed++;
                assertEquals("gen-3", agents.get(s));
            }
        }
        assertEquals(1, changed, "only the forced seat should differ");
    }

    @Test
    void leavesADrawThatAlreadyCarriesTheLabelUnchanged() {
        List<String> before = List.of(
                "forge-full", "gen-3", "gen-1", "forge-r30",
                "gen-1", "forge-full", "forge-r100", "gen-1");

        assertEquals(before, DraftWorkerMain.withRequiredAgent(before, "gen-3", new Random(7)));
    }

    @Test
    void absentOrBlankLabelLeavesSamplingByteForByteAsBefore() {
        List<String> before = List.of("forge-full", "gen-1", "forge-r30", "forge-full");

        for (String label : new String[] {null, "", "   "}) {
            // A fresh Random with the same seed must be left un-advanced, so a
            // run without the property draws exactly the sequence it always did.
            Random random = new Random(99);
            assertEquals(before, DraftWorkerMain.withRequiredAgent(before, label, random),
                    "seats unchanged for label " + label);
            assertEquals(new Random(99).nextInt(1000), random.nextInt(1000),
                    "the random stream must not be advanced for label " + label);
        }
    }

    @Test
    void theForcedSeatIsChosenUniformly() {
        Set<Integer> seatsHit = new HashSet<>();
        Random random = new Random(3);
        List<String> pod = List.of(
                "forge-full", "forge-full", "forge-full", "forge-full",
                "forge-full", "forge-full", "forge-full", "forge-full");
        for (int trial = 0; trial < 400; trial++) {
            seatsHit.add(DraftWorkerMain.withRequiredAgent(pod, "gen-3", random).indexOf("gen-3"));
        }
        assertEquals(8, seatsHit.size(), "every seat index should be reachable");
    }
}
