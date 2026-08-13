package com.pricepredictor.connector;

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
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DraftWorkerMain}.
 *
 * <p>The pure helpers (agent-mix parsing, JSON escaping, random-override
 * fraction) need no Forge; the full-draft transcript test drives Forge's draft
 * AI and is the reason the class is tagged "integration".
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
     * <p>{@code generateDraft} returns null when its 24 boosters are not all one size,
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
        List<String> eligible = MatchGenerator.computeEligibleSets();
        assertFalse(eligible.isEmpty(), "expected at least one eligible sealed-legal set");
        for (String code : eligible) {
            CardEdition edition = StaticData.instance().getEditions().get(code);
            SealedTemplate template =
                    edition == null ? null : edition.getBoosterTemplate("Draft");
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

    // ── Pure helpers ──────────────────────────────────────────────────────────

    @Test
    void randomOverrideFractionByAgent() {
        assertEquals(0.0, DraftWorkerMain.randomOverrideFraction("forge-full"));
        assertEquals(0.30, DraftWorkerMain.randomOverrideFraction("forge-r30"));
        assertEquals(1.0, DraftWorkerMain.randomOverrideFraction("forge-r100"));
        assertEquals(0.0, DraftWorkerMain.randomOverrideFraction("unknown"));
    }

    @Test
    void jsonEscapeHandlesQuotesAndControlChars() {
        assertEquals("Lim-Dûl", DraftWorkerMain.jsonEscape("Lim-Dûl")); // accented passes through
        assertEquals("a\\\"b", DraftWorkerMain.jsonEscape("a\"b"));
        assertEquals("a\\\\b", DraftWorkerMain.jsonEscape("a\\b"));
        assertEquals("a\\nb", DraftWorkerMain.jsonEscape("a\nb"));
    }

    @Test
    void agentMixSamplesByWeight() {
        DraftWorkerMain.AgentMix mix = DraftWorkerMain.AgentMix.parse("forge-full:9,forge-r100:1");
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
    void agentMixRejectsMalformed() {
        assertThrows(IllegalArgumentException.class, () -> DraftWorkerMain.AgentMix.parse("forge-full"));
        assertThrows(IllegalArgumentException.class, () -> DraftWorkerMain.AgentMix.parse("forge-full:0"));
        assertThrows(NumberFormatException.class, () -> DraftWorkerMain.AgentMix.parse("forge-full:x"));
    }

    // ── Forge-driven full draft ───────────────────────────────────────────────

    @Test
    void generatesFullyDrainedEightSeatThreePackTranscript() {
        String setCode = uniformBoosterSet();

        BoosterDraft context = BoosterDraft.createDraft(LimitedPoolType.Full);
        DraftWorkerMain.AgentMix mix = DraftWorkerMain.AgentMix.parse("forge-full:1");

        String line = DraftWorkerMain.generateDraft(context, setCode, mix, new Random(0));
        assertNotNull(line, "draft should produce a transcript for " + setCode);
        assertTrue(line.startsWith(DraftWorkerMain.SENTINEL), "line must start with sentinel");
        assertFalse(line.substring(DraftWorkerMain.SENTINEL.length()).contains("\n"),
                "transcript must be a single newline-free line");

        String json = line.substring(DraftWorkerMain.SENTINEL.length());
        // 8 seats * 3 packs = 24 boosters, each with a set_code; 8 seat agents.
        assertEquals(24, countOccurrences(json, "\"set_code\""), "expected 24 boosters");
        assertEquals(8, countOccurrences(json, "\"agent\""), "expected 8 seat agents");
        assertTrue(json.contains("\"draft_id\""));
        // Each booster has a non-empty drained picks array.
        assertEquals(24, countOccurrences(json, "\"picks\":["), "every booster lists picks");
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
                    if (!line.startsWith(DraftWorkerMain.PICK_REQUEST_SENTINEL)) {
                        continue;
                    }
                    String json = line.substring(DraftWorkerMain.PICK_REQUEST_SENTINEL.length());
                    respOut.println(DraftWorkerMain.PICK_RESPONSE_SENTINEL
                            + "{\"draft_id\":\"" + jsonStringField(json, "draft_id")
                            + "\",\"seat\":" + jsonIntField(json, "seat")
                            + ",\"pack_number\":" + jsonIntField(json, "pack_number")
                            + ",\"pick_number\":" + jsonIntField(json, "pick_number")
                            + ",\"pick\":\"" + DraftWorkerMain.jsonEscape(firstPackCard(json)) + "\"}");
                }
            } catch (IOException e) {
                // The request pipe closed when the draft finished; expected.
            }
        });
        responder.setDaemon(true);
        responder.start();

        BoosterDraft context = BoosterDraft.createDraft(LimitedPoolType.Full);
        DraftWorkerMain.AgentMix mix = DraftWorkerMain.AgentMix.parse("draft-agent:1");
        String line = DraftWorkerMain.generateDraft(
                context, setCode, mix, new Random(0), out, stdin,
                Set.of("draft-agent"), "test-draft");
        out.close();  // signal the responder EOF

        assertNotNull(line, "model-piloted draft should produce a transcript");
        assertTrue(line.startsWith(DraftWorkerMain.SENTINEL));
        String json = line.substring(DraftWorkerMain.SENTINEL.length());
        assertTrue(json.contains("\"draft_id\":\"test-draft\""));
        assertEquals(8, countOccurrences(json, "\"agent\":\"draft-agent\""),
                "every seat should carry the model label");
        assertEquals(24, countOccurrences(json, "\"picks\":["), "every booster drained");
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
