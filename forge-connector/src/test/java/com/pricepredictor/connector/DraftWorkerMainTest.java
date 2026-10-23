package com.pricepredictor.connector;

import com.pricepredictor.connector.json.Json;
import forge.StaticData;
import forge.card.CardEdition;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import org.eclipse.jetty.util.ajax.JSON;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
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
        assertFalse(line.substring(DraftWorkerMain.SENTINEL.length()).contains("\n"),
                "transcript must be a single newline-free line");

        Map<String, Object> transcript = transcript(line);
        assertNotNull(transcript.get("draft_id"));
        assertEquals(8, entries(transcript, "seats").size(), "expected 8 seats");
        // 8 seats * 3 packs = 24 boosters, each drained into its picks.
        List<Map<String, Object>> boosters = entries(transcript, "boosters");
        assertEquals(24, boosters.size(), "expected 24 boosters");
        for (Map<String, Object> booster : boosters) {
            assertNotNull(booster.get("set_code"), booster.toString());
            assertTrue(((Object[]) booster.get("picks")).length > 0, "every booster lists picks");
        }
    }

    /**
     * A set named by an alias is recorded under the code its edition goes by:
     * every downstream join compares set codes as strings, so {@code --set
     * CON} must not record a set no pool or match file names. Conflux goes by
     * {@code CFX}.
     */
    @Test
    void anAliasedSetIsRecordedUnderItsCanonicalCode() {
        assertEquals(Collections.nCopies(24, "CFX"), setCodes(completedLine(forgeDraft("CON"))));
    }

    /** A set named by its own code is recorded under that code. */
    @Test
    void aPlainSetCodeIsRecordedUnchanged() {
        String setCode = uniformBoosterSet();
        assertEquals(Collections.nCopies(24, setCode), setCodes(completedLine(forgeDraft(setCode))));
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

    /** The transcript a completed draft's line carries, parsed. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> transcript(String line) {
        assertTrue(line.startsWith(DraftWorkerMain.SENTINEL), "line must start with sentinel");
        return (Map<String, Object>) JSON.parse(line.substring(DraftWorkerMain.SENTINEL.length()));
    }

    /** The objects of one of the transcript's arrays, in order. */
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> entries(Map<String, Object> transcript, String key) {
        return Arrays.stream((Object[]) transcript.get(key))
                .map(entry -> (Map<String, Object>) entry)
                .toList();
    }

    /** The set code each booster of a completed draft is recorded under, in order. */
    private static List<Object> setCodes(String line) {
        return entries(transcript(line), "boosters").stream()
                .map(booster -> booster.get("set_code"))
                .toList();
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
                    Map<?, ?> request = (Map<?, ?>) JSON.parse(
                            line.substring(DraftPickProtocol.PICK_REQUEST_SENTINEL.length()));
                    respOut.println(DraftPickProtocol.PICK_RESPONSE_SENTINEL + Json.object()
                            .value("draft_id", request.get("draft_id"))
                            .value("seat", request.get("seat"))
                            .value("pack_number", request.get("pack_number"))
                            .value("pick_number", request.get("pick_number"))
                            .value("pick", ((Object[]) request.get("pack"))[0])
                            .render());
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

        Map<String, Object> transcript = transcript(completedLine(result));
        assertEquals("test-draft", transcript.get("draft_id"));
        List<Map<String, Object>> seats = entries(transcript, "seats");
        assertEquals(8, seats.size());
        for (Map<String, Object> seat : seats) {
            assertEquals("draft-agent", seat.get("agent"), "every seat should carry the model label");
        }
        List<Map<String, Object>> boosters = entries(transcript, "boosters");
        assertEquals(24, boosters.size());
        for (Map<String, Object> booster : boosters) {
            assertTrue(((Object[]) booster.get("picks")).length > 0, "every booster drained");
        }
    }
}
