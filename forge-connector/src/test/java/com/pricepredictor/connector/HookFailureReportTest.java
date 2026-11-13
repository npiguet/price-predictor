package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.HookFailures;
import com.pricepredictor.connector.effects.PatchHooks;
import com.pricepredictor.connector.effects.TestHookFailures;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
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
}
