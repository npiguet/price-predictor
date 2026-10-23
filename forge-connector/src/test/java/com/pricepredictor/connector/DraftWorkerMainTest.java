package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.StaticData;
import forge.card.CardEdition;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DraftWorkerMain}.
 *
 * <p>Every test drives Forge's draft AI, which is why the class is tagged
 * "integration"; the pure helpers are covered by {@link AgentMixTest} and
 * {@link DraftAgentTest}.
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class DraftWorkerMainTest {

    /** Template draws used to decide whether a set's boosters are a fixed size. */
    private static final int UNIFORMITY_PROBE_DRAWS = 400;

    private static String uniformBoosterSet;

    /**
     * An eligible set whose Draft boosters always hold the same number of cards.
     *
     * <p>{@code generateDraft} skips a draft whose 24 boosters are not all one size,
     * because the reconstruction geometry needs a fixed pack size. Some sets do not
     * oblige: Secrets of Strixhaven yields a 13-card booster instead of 14 about 2 % of
     * the time, which is a 38 % chance that a 24-booster draft is skipped. Those skips
     * are correct — the worker loop just draws again — but a test that asserts a
     * transcript from one attempt on such a set fails at that rate.
     *
     * <p>So the set is chosen rather than assumed. Taking {@code eligible.get(0)} binds
     * the test to whichever set the Forge checkout added most recently, which is exactly
     * where an irregular booster template is most likely to appear.
     */
    private static synchronized String uniformBoosterSet() {
        if (uniformBoosterSet != null) {
            return uniformBoosterSet;
        }
        List<String> eligible = BoosterTemplates.eligibleSealedSets();
        assertFalse(eligible.isEmpty(), "expected at least one eligible sealed-legal set");
        for (String code : eligible) {
            CardEdition edition = StaticData.instance().getEditions().get(code);
            SealedTemplate template =
                    edition == null ? null : edition.getBoosterTemplate(BoosterTemplates.DRAFT_BOOSTER);
            if (template == null) {
                continue;
            }
            if (boosterSizeIsFixed(template)) {
                uniformBoosterSet = code;
                return uniformBoosterSet;
            }
        }
        fail("no eligible set produced a fixed booster size over "
                + UNIFORMITY_PROBE_DRAWS + " draws");
        return null;
    }

    private static boolean boosterSizeIsFixed(SealedTemplate template) {
        int size = -1;
        for (int draw = 0; draw < UNIFORMITY_PROBE_DRAWS; draw++) {
            int drawn = new UnOpenedProduct(template).get().size();
            if (size == -1) {
                size = drawn;
            } else if (drawn != size) {
                return false;
            }
        }
        return size > 0;
    }

    /** A pick channel that routes no agent: every seat picks in-process. */
    private static DraftPickProtocol.PickChannel forgeOnlyChannel() {
        return new DraftPickProtocol.PickChannel(
                new PrintStream(new ByteArrayOutputStream()),
                new BufferedReader(new StringReader("")), Set.of());
    }

    /** A fresh pod context drawing {@code mixSpec} from a seeded stream, no seat forced. */
    private static DraftWorkerMain.DraftSettings settings(
            String mixSpec, DraftPickProtocol.PickChannel channel) {
        return new DraftWorkerMain.DraftSettings(
                BoosterDraft.createDraft(LimitedPoolType.Full), AgentMix.parse(mixSpec),
                new Random(0), channel, null);
    }

    /** A Forge-only draft of {@code setCode}, every seat {@code forge-full}. */
    private static DraftWorkerMain.DraftResult forgeDraft(String setCode) {
        return DraftWorkerMain.generateDraft(
                settings("forge-full:1", forgeOnlyChannel()), setCode, "test-draft");
    }

    // ── Forge-driven full draft ───────────────────────────────────────────────

    @Test
    void generatesFullyDrainedEightSeatThreePackTranscript() {
        String setCode = uniformBoosterSet();

        String line = completedLine(forgeDraft(setCode));
        assertTrue(line.startsWith(DraftWorkerMain.SENTINEL), "line must start with sentinel");
        assertFalse(line.substring(DraftWorkerMain.SENTINEL.length()).contains("\n"),
                "transcript must be a single newline-free line");

        String json = line.substring(DraftWorkerMain.SENTINEL.length());
        // 8 seats * 3 packs = 24 boosters, each with a set_code; 8 seat agents.
        assertEquals(24, countOccurrences(json, "\"set_code\""), "expected 24 boosters");
        assertEquals(8, countOccurrences(json, "\"agent\""), "expected 8 seat agents");
        assertTrue(json.contains("\"draft_id\""));
        // Each booster has a non-empty drained picks array: an empty one would
        // read "picks":[] and not match.
        assertEquals(24, countOccurrences(json, "\"picks\":[\""), "every booster lists picks");
    }

    /**
     * A set named by an alias is recorded under the code its edition goes by:
     * every downstream join compares set codes as strings, so {@code --set
     * CON} must not record a set no pool or match file names. Conflux goes by
     * {@code CFX}.
     */
    @Test
    void anAliasedSetIsRecordedUnderItsCanonicalCode() {
        String json = completedLine(forgeDraft("CON"));

        assertEquals(24, countOccurrences(json, "\"set_code\":\"CFX\""), json);
        assertEquals(0, countOccurrences(json, "\"CON\""), json);
    }

    /** A set named by its own code is recorded under that code. */
    @Test
    void aPlainSetCodeIsRecordedUnchanged() {
        String setCode = uniformBoosterSet();
        String json = completedLine(forgeDraft(setCode));

        assertEquals(24, countOccurrences(json, "\"set_code\":" + Json.string(setCode)), json);
    }

    private static String completedLine(DraftWorkerMain.DraftResult result) {
        DraftWorkerMain.DraftResult.Completed completed = assertInstanceOf(
                DraftWorkerMain.DraftResult.Completed.class, result,
                "the draft should produce a transcript");
        return completed.line();
    }

    @Test
    void aSetWithNoBoosterTemplateIsSkippedNamingTheSet() {
        // Returned rather than thrown, so the worker loop must count it: a bad
        // -Ddraft.set once skipped every draft in silence, forever.
        DraftWorkerMain.DraftResult result = forgeDraft("NOT-A-SET");

        DraftWorkerMain.DraftResult.Skipped skipped =
                assertInstanceOf(DraftWorkerMain.DraftResult.Skipped.class, result);
        assertTrue(skipped.reason().contains("NOT-A-SET"), skipped.reason());
        assertEquals(WorkerLoop.Outcome.SKIPPED,
                DraftWorkerMain.emit(result, "NOT-A-SET", System.out));
    }

    private static int countOccurrences(String haystack, String needle) {
        int count = 0;
        int idx = 0;
        while ((idx = haystack.indexOf(needle, idx)) != -1) {
            count++;
            idx += needle.length();
        }
        return count;
    }

    // ── Live model-seat pick side-channel (Forge-driven) ──────────────────────

    /**
     * Drives a full pod where every seat is the external label "draft-agent": a
     * responder thread answers each {@code <<DRAFT-PICK-REQUEST>>} by echoing the
     * first card in the held pack, and the draft completes with a normal
     * transcript whose seats all carry the model label.
     */
    @Test
    void liveModelSeatDraftProducesTranscript() throws Exception {
        String setCode = uniformBoosterSet();

        // requests: worker `out` → responder; responses: responder → worker `stdin`.
        PipedInputStream reqIn = new PipedInputStream(1 << 16);
        PrintStream out = new PrintStream(new PipedOutputStream(reqIn), true, StandardCharsets.UTF_8);
        BufferedReader reqReader = new BufferedReader(new InputStreamReader(reqIn, StandardCharsets.UTF_8));
        PipedInputStream respIn = new PipedInputStream(1 << 16);
        PrintStream respOut = new PrintStream(new PipedOutputStream(respIn), true, StandardCharsets.UTF_8);
        BufferedReader stdin = new BufferedReader(new InputStreamReader(respIn, StandardCharsets.UTF_8));

        Thread responder = new Thread(() -> {
            try {
                String line;
                while ((line = reqReader.readLine()) != null) {
                    if (!line.startsWith(DraftPickProtocol.PICK_REQUEST_SENTINEL)) {
                        continue;
                    }
                    String json = line.substring(DraftPickProtocol.PICK_REQUEST_SENTINEL.length());
                    respOut.println(DraftPickProtocol.PICK_RESPONSE_SENTINEL
                            + "{\"draft_id\":\"" + jsonStringField(json, "draft_id")
                            + "\",\"seat\":" + jsonIntField(json, "seat")
                            + ",\"pack_number\":" + jsonIntField(json, "pack_number")
                            + ",\"pick_number\":" + jsonIntField(json, "pick_number")
                            + ",\"pick\":" + Json.string(firstPackCard(json)) + "}");
                }
            } catch (IOException e) {
                // The request pipe closed when the draft finished; expected.
            }
        });
        responder.setDaemon(true);
        responder.start();

        DraftWorkerMain.DraftResult result = DraftWorkerMain.generateDraft(
                settings("draft-agent:1",
                        new DraftPickProtocol.PickChannel(out, stdin, Set.of("draft-agent"))),
                setCode, "test-draft");
        out.close();  // signal the responder EOF

        String line = completedLine(result);
        assertTrue(line.startsWith(DraftWorkerMain.SENTINEL));
        String json = line.substring(DraftWorkerMain.SENTINEL.length());
        assertTrue(json.contains("\"draft_id\":\"test-draft\""));
        assertEquals(8, countOccurrences(json, "\"agent\":\"draft-agent\""),
                "every seat should carry the model label");
        assertEquals(24, countOccurrences(json, "\"picks\":[\""), "every booster drained");
    }

    private static String jsonStringField(String json, String key) {
        String marker = "\"" + key + "\":\"";
        int i = json.indexOf(marker) + marker.length();
        int j = json.indexOf('"', i);  // uuid / scalar fields carry no escaped quotes
        return json.substring(i, j);
    }

    private static long jsonIntField(String json, String key) {
        String marker = "\"" + key + "\":";
        int i = json.indexOf(marker) + marker.length();
        int j = i;
        while (Character.isDigit(json.charAt(j))) {
            j++;
        }
        return Long.parseLong(json.substring(i, j));
    }

    private static String firstPackCard(String json) {
        int i = json.indexOf("\"pack\":[\"") + "\"pack\":[\"".length();
        StringBuilder sb = new StringBuilder();
        while (json.charAt(i) != '"') {
            if (json.charAt(i) == '\\') {
                sb.append(json.charAt(i + 1));  // unescape \" and \\ (card names)
                i += 2;
            } else {
                sb.append(json.charAt(i));
                i++;
            }
        }
        return sb.toString();
    }
}
