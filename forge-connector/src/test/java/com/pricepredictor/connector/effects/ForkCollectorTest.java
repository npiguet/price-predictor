package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork budgets and the record shapes they produce.
 *
 * <p>Forking a live game needs one, which is the integration test's job. What is
 * testable here is the budget arithmetic — and it is worth testing precisely,
 * because the failure mode is a run that spends its whole simulation budget on
 * forks of one board.
 */
class ForkCollectorTest {

    @TempDir
    Path tempDir;

    private static CollectionCaps caps(
            int interventions, int probes, List<String> keywords) {
        return new CollectionCaps(2000, 0.1, interventions, probes, keywords);
    }

    private ForkCollector collector(CollectionCaps caps) {
        return new ForkCollector(
                null, new RecordShardWriter(tempDir, "run", 0), "run.0.0",
                caps, 42L);
    }

    // ── the per-game budget ─────────────────────────────────────────────

    @Test
    void aFreshGameMayIntervene() {
        assertTrue(collector(caps(2, 2, List.of())).mayIntervene(1));
    }

    @Test
    void anExhaustedInterventionBudgetStopsFurtherForks() {
        ForkCollector collector = collector(caps(0, 2, List.of()));
        assertFalse(collector.mayIntervene(1));
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void atMostTwoForksMayTargetOneRealResolution() {
        assertEquals(2, ForkCollector.MAX_FORKS_PER_RESOLUTION);
    }

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        ForkCollector collector = collector(caps(100, 2, List.of()));
        collector.intervene(null, 7);
        collector.intervene(null, 7);
        assertFalse(collector.mayIntervene(7));
        assertTrue(collector.mayIntervene(8));
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. A null game makes every copy fail.
        ForkCollector collector = collector(caps(1, 2, List.of()));
        assertFalse(collector.intervene(null, 1));
        assertFalse(collector.mayIntervene(1));
        assertEquals(1, collector.discardedForks());
    }

    @Test
    void aFailedForkWritesNoRecord() {
        ForkCollector collector = collector(caps(2, 2, List.of()));
        collector.intervene(null, 1);
        assertEquals(0L, collector.recordsWritten());
    }

    // ── the probe switch ────────────────────────────────────────────────

    @Test
    void noProbeIsTakenWithTheFlagUnset() {
        // Whatever the build state: the flag is the runtime switch, and it is
        // deliberately separate from the build decision gate 2 drives.
        ForkCollector collector = collector(caps(2, 2, List.of()));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void onlyTheNamedKeywordsAreProbed() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertTrue(collector.mayProbe("wither"));
        assertFalse(collector.mayProbe("lifelink"));
    }

    @Test
    void theProbeBudgetIsSpentByTakingProbes() {
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertTrue(collector.recordProbeBranch(
                "wither", "run.0.1", "{}", "{}", "P0"));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void aProbeBranchIsRecordedAsAnOrdinaryCombatRecord() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertTrue(collector.recordProbeBranch(
                "wither", "run.0.1", "{}", "{}", "P0"));
        assertEquals(1L, collector.recordsWritten());
    }

    @Test
    void aProbeForAnUnnamedKeywordWritesNothing() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertFalse(collector.recordProbeBranch(
                "lifelink", "run.0.1", "{}", "{}", "P0"));
        assertEquals(0L, collector.recordsWritten());
    }

    // ── the seeded source ───────────────────────────────────────────────

    @Test
    void bothBranchesDrawFromOneSeededSource() {
        // A difference between the branches has to be the keyword, not the
        // draw, so the source is seeded per fork rather than per branch.
        ForkCollector first = collector(caps(2, 2, List.of()));
        ForkCollector second = collector(caps(2, 2, List.of()));
        assertEquals(
                first.seededRandom().nextInt(1000),
                second.seededRandom().nextInt(1000));
    }
}
