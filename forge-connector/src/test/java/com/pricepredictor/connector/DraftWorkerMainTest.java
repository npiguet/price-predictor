package com.pricepredictor.connector;

import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.LimitedPoolType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Random;

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
        List<String> eligible = MatchGenerator.computeEligibleSets();
        assertFalse(eligible.isEmpty(), "expected at least one eligible sealed-legal set");
        String setCode = eligible.get(0);

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
}
