package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.DirectedForkTally;
import com.pricepredictor.connector.effects.HookFailures;
import com.pricepredictor.connector.effects.PatchHooks;
import com.pricepredictor.connector.effects.TestHookFailures;
import com.pricepredictor.connector.json.FlatJson;
import com.pricepredictor.connector.json.JsonParts;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The worker's failure report (V3 #14): the connector's counts and Forge's
 * per slot on one line, printed after a match however it ended, a match
 * that threw included.
 */
class HookFailureReportTest {

    private static String stderrOf(Runnable body) {
        PrintStream err = System.err;
        ByteArrayOutputStream printed = new ByteArrayOutputStream();
        System.setErr(new PrintStream(printed, true, StandardCharsets.UTF_8));
        try {
            body.run();
        } finally {
            System.setErr(err);
        }
        return printed.toString(StandardCharsets.UTF_8);
    }

    @Test
    void aMatchThatThrowsStillPrintsTheCountsItChanged() {
        MatchWorkerMain.HookFailureReport report = new MatchWorkerMain.HookFailureReport();
        stderrOf(report::afterMatch);

        String printed = stderrOf(() -> assertThrows(IllegalStateException.class, () ->
                MatchWorkerMain.oneMatch(() -> {
                    TestHookFailures.count("report under test");
                    throw new IllegalStateException("the match threw");
                }, Optional.empty(), Optional.empty(), report)));

        assertEquals("effect-record hook failures so far, listener calls that threw and lost"
                        + " their records: " + HookFailures.counts()
                        + "; contained by Forge, per slot: " + PatchHooks.forgeFailureCounts(),
                printed.trim());
        assertTrue(printed.contains("report under test="), printed);
    }

    @Test
    void unchangedCountsPrintNothing() {
        MatchWorkerMain.HookFailureReport report = new MatchWorkerMain.HookFailureReport();
        stderrOf(report::afterMatch);

        assertEquals("", stderrOf(report::afterMatch));
    }

    /**
     * A fork that refused is no failure: the tally is a line of its own,
     * printed when it changed, and the failure line does not hold it.
     */
    @Test
    void forkRefusalsArePrintedApartFromTheFailures() {
        MatchWorkerMain.HookFailureReport report = new MatchWorkerMain.HookFailureReport();
        stderrOf(report::afterMatch);

        TestHookFailures.refuse("report-test", "no choices");
        String printed = stderrOf(report::afterMatch);

        assertEquals(MatchWorkerMain.HookFailureReport.REFUSALS_LINE
                + com.pricepredictor.connector.effects.ForkRefusals.counts(), printed.trim());
        assertTrue(printed.contains("report-test no choices=1"), printed);
        assertTrue(!HookFailures.counts().containsKey("report-test no choices"));
        assertEquals("", stderrOf(report::afterMatch));
    }

    /**
     * The directed fork tally is a line of its own, printed when it changed:
     * {@code DirectedForkTally.LINE} and a JSON object from each rule to its
     * six counts, which parses back to the tally whatever the rule id holds
     * (here a comma, a space and an equals sign, which a map's toString
     * would let a reader split on).
     */
    @Test
    void theDirectedForkTallyLineParsesBackToTheTally() {
        MatchWorkerMain.HookFailureReport report = new MatchWorkerMain.HookFailureReport();
        stderrOf(report::afterMatch);
        String rule = "704.5f, report=test";

        TestHookFailures.tallyDirected(rule, "MATCHED");
        TestHookFailures.tallyDirected(rule, "MATCHED");
        TestHookFailures.tallyDirected(rule, "DRAWN");
        TestHookFailures.tallyDirected(rule, "TAKEN");
        TestHookFailures.tallyDirected(rule, "PRECONDITION_THREW");
        String printed = stderrOf(report::afterMatch).trim();

        assertTrue(printed.startsWith(DirectedForkTally.LINE), printed);
        Map<String, String> rules = JsonParts.fields(printed.substring(DirectedForkTally.LINE.length()));
        Map<String, Object> parsed = FlatJson.parse(rules.get(rule));
        assertEquals(Map.of("matched", 2L, "drawn", 1L, "budget_blocked", 0L, "taken", 1L,
                "kept", 0L, "precondition_threw", 1L), parsed);
        assertEquals(TestHookFailures.directedCounts(rule), parsed);
        assertEquals("", stderrOf(report::afterMatch), "an unchanged tally prints nothing");
    }
}
