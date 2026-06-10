package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure (Forge-free) tests for the live-pick side-channel protocol in
 * {@link DraftWorkerMain}: request formatting, response parsing, routing-field
 * validation, held-pack membership, and the abort/abandon outcomes (T009).
 */
class DraftPickProtocolTest {

    private static String response(
            String draftId, int seat, int packNumber, int pickNumber, String body) {
        return DraftWorkerMain.PICK_RESPONSE_SENTINEL
                + "{\"draft_id\":\"" + draftId + "\",\"seat\":" + seat
                + ",\"pack_number\":" + packNumber + ",\"pick_number\":" + pickNumber
                + "," + body + "}";
    }

    // ── Request formatting (data-model §2.1) ─────────────────────────────────

    @Test
    void formatPickRequestEmitsAllFields() {
        String line = DraftWorkerMain.formatPickRequest(
                "d1", 3, "draft-agent", 8, 1, 5, "BLB",
                List.of("Card A", "Card B"));
        assertTrue(line.startsWith(DraftWorkerMain.PICK_REQUEST_SENTINEL));
        String json = line.substring(DraftWorkerMain.PICK_REQUEST_SENTINEL.length());
        assertTrue(json.contains("\"draft_id\":\"d1\""));
        assertTrue(json.contains("\"seat\":3"));
        assertTrue(json.contains("\"agent\":\"draft-agent\""));
        assertTrue(json.contains("\"pod_size\":8"));
        assertTrue(json.contains("\"pack_number\":1"));
        assertTrue(json.contains("\"pick_number\":5"));
        assertTrue(json.contains("\"set_code\":\"BLB\""));
        assertTrue(json.contains("\"pack\":[\"Card A\",\"Card B\"]"));
    }

    @Test
    void formatPickRequestEscapesCardNames() {
        String line = DraftWorkerMain.formatPickRequest(
                "d1", 0, "a", 8, 1, 1, "TST", List.of("Lim-Dûl's \"Vault\""));
        assertTrue(line.contains("Lim-Dûl's \\\"Vault\\\""));  // accented passes, quote escaped
    }

    // ── Response parsing (flat JSON) ─────────────────────────────────────────

    @Test
    void parseFlatJsonTypesValues() {
        Map<String, Object> m = DraftWorkerMain.parseFlatJson(
                "{\"draft_id\":\"d1\",\"seat\":3,\"pick\":\"Card B\",\"abort\":false}");
        assertEquals("d1", m.get("draft_id"));
        assertEquals(3L, m.get("seat"));
        assertEquals("Card B", m.get("pick"));
        assertEquals(Boolean.FALSE, m.get("abort"));
    }

    @Test
    void parseResponseRejectsMissingSentinel() {
        assertThrows(DraftWorkerMain.DraftAbandonedException.class,
                () -> DraftWorkerMain.parseResponse("{\"pick\":\"X\"}"));  // no sentinel
    }

    // ── resolvePick: routing + membership + abort ────────────────────────────

    @Test
    void resolvePickReturnsHeldPackCard() {
        Map<String, Object> resp = DraftWorkerMain.parseResponse(
                response("d1", 0, 1, 5, "\"pick\":\"Card B\""));
        String name = DraftWorkerMain.resolvePick(
                resp, "d1", 0, 1, 5, List.of("Card A", "Card B", "Card C"));
        assertEquals("Card B", name);
    }

    @Test
    void resolvePickAbortThrowsAborted() {
        Map<String, Object> resp = DraftWorkerMain.parseResponse(
                response("d1", 0, 1, 5, "\"abort\":true"));
        assertThrows(DraftWorkerMain.DraftAbortedException.class,
                () -> DraftWorkerMain.resolvePick(
                        resp, "d1", 0, 1, 5, List.of("Card A")));
    }

    @Test
    void resolvePickRoutingMismatchThrowsAbandoned() {
        Map<String, Object> resp = DraftWorkerMain.parseResponse(
                response("d1", 7, 1, 5, "\"pick\":\"Card A\""));  // seat 7 != 0
        assertThrows(DraftWorkerMain.DraftAbandonedException.class,
                () -> DraftWorkerMain.resolvePick(
                        resp, "d1", 0, 1, 5, List.of("Card A")));
    }

    @Test
    void resolvePickNotInPackThrowsAbandoned() {
        Map<String, Object> resp = DraftWorkerMain.parseResponse(
                response("d1", 0, 1, 5, "\"pick\":\"Not In Pack\""));
        assertThrows(DraftWorkerMain.DraftAbandonedException.class,
                () -> DraftWorkerMain.resolvePick(
                        resp, "d1", 0, 1, 5, List.of("Card A", "Card B")));
    }

    @Test
    void resolvePickMissingPickThrowsAbandoned() {
        Map<String, Object> resp = DraftWorkerMain.parseResponse(
                response("d1", 0, 1, 5, "\"note\":\"none\""));  // neither pick nor abort
        assertThrows(DraftWorkerMain.DraftAbandonedException.class,
                () -> DraftWorkerMain.resolvePick(
                        resp, "d1", 0, 1, 5, List.of("Card A")));
    }

    // ── external-agent set parsing ───────────────────────────────────────────

    @Test
    void parseExternalAgentsSplitsAndTrims() {
        assertEquals(java.util.Set.of("a", "b"),
                DraftWorkerMain.parseExternalAgents(" a , b "));
        assertTrue(DraftWorkerMain.parseExternalAgents(null).isEmpty());
        assertTrue(DraftWorkerMain.parseExternalAgents("  ").isEmpty());
    }
}
