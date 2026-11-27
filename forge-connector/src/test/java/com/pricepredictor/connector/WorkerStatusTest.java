package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.AttributionMode;
import com.pricepredictor.connector.effects.TestHookFailures;
import com.pricepredictor.connector.json.FlatJson;
import com.pricepredictor.connector.json.JsonParts;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.jar.JarFile;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The worker's status line: one JSON object after
 * {@link WorkerStatus#LINE}, printed at start-up and after a match whose
 * counts changed, however it ended, which reads back whole whatever a reason,
 * a gap or a rule id holds.
 */
class WorkerStatusTest {

    /** A name holding what a printed Java map or a split would cut: a comma, quotes, an equals sign, non-ASCII. */
    private static final String AWKWARD = "remembered, \"imprinted\" or exiled=Lim-Dûl's Vault";

    private static final String ENGINE_COMMIT = "0123456789abcdef0123456789abcdef01234567";
    private static final String CONNECTOR_COMMIT = "89abcdef0123456789abcdef0123456789abcdef";

    private static WorkerStatus status(List<String> gaps) {
        return new WorkerStatus("lifetime-1", new WorkerStatus.Engine(AttributionMode.PATCHED, gaps,
                new LoadedBuilds(ENGINE_COMMIT, false, CONNECTOR_COMMIT)),
                PinnedForgePreferences.PINNED);
    }

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

    /** The members of the one status line {@code printed} holds. */
    private static Map<String, String> members(String printed) {
        String line = printed.strip();
        assertTrue(line.startsWith(WorkerStatus.LINE), printed);
        assertFalse(line.contains("\n"), "one line: " + printed);
        return JsonParts.fields(line.substring(WorkerStatus.LINE.length()));
    }

    /** A JSON string literal's value. */
    private static Object decoded(String literal) {
        return FlatJson.parse("{\"v\":" + literal + "}").get("v");
    }

    @Test
    void theStartupLineCarriesTheLifetimeTheHooksAndThePinnedPreferences() {
        String printed = stderrOf(status(List.of())::atStartup);

        Map<String, String> members = members(printed);
        assertEquals(List.of("lifetime", "hooks", "pinned_preferences", "hook_failures",
                "fork_refusals", "directed"), List.copyOf(members.keySet()));
        assertEquals("lifetime-1", decoded(members.get("lifetime")));
        assertEquals(Map.of("mode", "\"patched\"", "gaps", "[]",
                "forge_build_commit", "\"" + ENGINE_COMMIT + "\"", "forge_build_dirty", "false",
                "connector_forge_commit", "\"" + CONNECTOR_COMMIT + "\""),
                JsonParts.fields(members.get("hooks")));
        assertEquals(Map.of("mulligan_rule", "London", "filtered_hands", false,
                "performance_mode", false, "ui_language", "en-US"),
                FlatJson.parse(members.get("pinned_preferences")));
    }

    @Test
    void theStartupLineIsPrintedWhateverTheCounts() {
        WorkerStatus status = status(List.of());
        stderrOf(status::atStartup);

        assertTrue(stderrOf(status::atStartup).startsWith(WorkerStatus.LINE));
    }

    /**
     * A gap and a refusal reason holding a comma, quotes, an equals sign and a
     * card name past ASCII read back whole, and the line itself is ASCII, so
     * the encoding the JVM writes stderr in cannot change it.
     */
    @Test
    void aGapAndAReasonHoldingACommaQuotesAndNonAsciiRoundTrip() {
        WorkerStatus status = status(List.of(AWKWARD, "plain gap"));
        TestHookFailures.refuse("status-test", AWKWARD);

        String printed = stderrOf(status::atStartup);

        assertTrue(printed.chars().allMatch(c -> c < 0x80), printed);
        Map<String, String> members = members(printed);
        List<Object> gaps = JsonParts.elements(JsonParts.fields(members.get("hooks")).get("gaps"))
                .stream().map(WorkerStatusTest::decoded).toList();
        assertEquals(List.of(AWKWARD, "plain gap"), gaps);
        assertEquals(1L, FlatJson.parse(members.get("fork_refusals")).get("status-test " + AWKWARD));
    }

    @Test
    void aMatchThatThrowsStillPrintsTheCountsItChanged() {
        WorkerStatus status = status(List.of());
        stderrOf(status::atStartup);

        String printed = stderrOf(() -> assertThrows(IllegalStateException.class, () ->
                MatchWorkerMain.oneMatch(() -> {
                    TestHookFailures.count("status under test");
                    throw new IllegalStateException("the match threw");
                }, Optional.empty(), Optional.empty(), status)));

        Map<String, String> failures = JsonParts.fields(members(printed).get("hook_failures"));
        assertEquals(List.of("connector", "forge"), List.copyOf(failures.keySet()));
        assertEquals(1L, FlatJson.parse(failures.get("connector")).get("status under test"));
    }

    @Test
    void unchangedCountsPrintNothingAfterAMatch() {
        WorkerStatus status = status(List.of());
        stderrOf(status::atStartup);

        assertEquals("", stderrOf(status::afterMatch));
    }

    /** A fork that refused is no failure: it is in its own member, not among the hook failures. */
    @Test
    void forkRefusalsAreCountedApartFromTheFailures() {
        WorkerStatus status = status(List.of());
        stderrOf(status::atStartup);

        TestHookFailures.refuse("status-test", "no choices");
        Map<String, String> members = members(stderrOf(status::afterMatch));

        assertEquals(1L, FlatJson.parse(members.get("fork_refusals")).get("status-test no choices"));
        assertFalse(members.get("hook_failures").contains("no choices"), members.toString());
        assertEquals("", stderrOf(status::afterMatch));
    }

    /**
     * The directed fork tally is the {@code directed} member: each rule to its
     * six counts, which parse back to the tally whatever the rule id holds.
     */
    @Test
    void theDirectedForkTallyParsesBackToTheTally() {
        WorkerStatus status = status(List.of());
        stderrOf(status::atStartup);
        String rule = "704.5f, status=test";

        TestHookFailures.tallyDirected(rule, "MATCHED");
        TestHookFailures.tallyDirected(rule, "MATCHED");
        TestHookFailures.tallyDirected(rule, "DRAWN");
        TestHookFailures.tallyDirected(rule, "TAKEN");
        TestHookFailures.tallyDirected(rule, "PRECONDITION_THREW");
        Map<String, String> rules = JsonParts.fields(members(stderrOf(status::afterMatch)).get("directed"));

        Map<String, Object> parsed = FlatJson.parse(rules.get(rule));
        assertEquals(Map.of("matched", 2L, "drawn", 1L, "budget_blocked", 0L, "taken", 1L,
                "kept", 0L, "precondition_threw", 1L), parsed);
        assertEquals(TestHookFailures.directedCounts(rule), parsed);
        assertEquals("", stderrOf(status::afterMatch), "an unchanged tally prints nothing");
    }

    /**
     * The status line names the commit the Forge engine this JVM loaded was
     * built at: the {@code Forge-Build-Commit} of the jar {@code forge.game.Game}
     * comes from, the installed {@code forge-game} jar on this classpath, read
     * here from that jar's manifest.
     */
    @Test
    void theLineNamesTheCommitTheLoadedForgeGameJarWasBuiltAt() throws IOException {
        String installed;
        try (JarFile jar = new JarFile(forgeGameJar())) {
            installed = jar.getManifest().getMainAttributes().getValue("Forge-Build-Commit");
        }
        assertTrue(installed != null && installed.matches("[0-9a-f]{40}"),
                "the installed forge-game jar names no build commit: rebuild ../forge with install");

        WorkerStatus status = new WorkerStatus("lifetime-1", new WorkerStatus.Engine(
                AttributionMode.PATCHED, List.of(), LoadedBuilds.ofThisJvm()),
                PinnedForgePreferences.PINNED);
        Map<String, String> hooks = JsonParts.fields(members(stderrOf(status::atStartup)).get("hooks"));

        assertEquals(installed, decoded(hooks.get("forge_build_commit")));
        assertEquals("false", hooks.get("forge_build_dirty"), "../forge was built with uncommitted changes");
    }

    /** The forge-game jar on this JVM's classpath. */
    private static File forgeGameJar() {
        return Arrays.stream(System.getProperty("java.class.path").split(File.pathSeparator))
                .map(File::new)
                .filter(file -> file.getName().matches("forge-game-.*\\.jar"))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no forge-game jar on the classpath"));
    }
}
