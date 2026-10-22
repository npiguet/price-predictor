package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure (Forge-free) tests for the live-pick side-channel protocol in
 * {@link DraftPickProtocol}: request formatting, response parsing, routing-field
 * validation, held-pack membership, and the abort/abandon outcomes (T009).
 */
class DraftPickProtocolTest {

    private static String response(
            String draftId, int seat, int packNumber, int pickNumber, String body) {
        return DraftPickProtocol.PICK_RESPONSE_SENTINEL
                + "{\"draft_id\":\"" + draftId + "\",\"seat\":" + seat
                + ",\"pack_number\":" + packNumber + ",\"pick_number\":" + pickNumber
                + "," + body + "}";
    }

    private static DraftPickProtocol.PickRequest request(
            String draftId, int seat, int packNumber, int pickNumber, List<String> pack) {
        return new DraftPickProtocol.PickRequest(
                draftId, seat, "draft-agent", 8, packNumber, pickNumber, "BLB", pack);
    }

    // ── Request formatting (data-model §2.1) ─────────────────────────────────

    @Test
    void theRequestLineIsByteForByteTheProtocolLine() {
        // The supervisor's reader is the contract; pin the whole line, key
        // order and spacing included.
        assertEquals("<<DRAFT-PICK-REQUEST>>{\"draft_id\":\"d1\",\"seat\":3,"
                        + "\"agent\":\"draft-agent\",\"pod_size\":8,\"pack_number\":1,"
                        + "\"pick_number\":5,\"set_code\":\"BLB\","
                        + "\"pack\":[\"Card A\",\"Card B\"]}",
                request("d1", 3, 1, 5, List.of("Card A", "Card B")).toWireLine());
    }

    @Test
    void anAbandonedLineNamesTheDraftAndTheReason() {
        assertEquals("<<DRAFT-ABANDONED>>{\"draft_id\":\"d1\",\"reason\":\"pick 'X' not in held pack\"}",
                DraftPickProtocol.formatAbandoned("d1", "pick 'X' not in held pack"));
    }

    @Test
    void formatPickRequestEmitsAllFields() {
        String line = request("d1", 3, 1, 5, List.of("Card A", "Card B")).toWireLine();
        assertTrue(line.startsWith(DraftPickProtocol.PICK_REQUEST_SENTINEL));
        String json = line.substring(DraftPickProtocol.PICK_REQUEST_SENTINEL.length());
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
        String line = new DraftPickProtocol.PickRequest(
                "d1", 0, "a", 8, 1, 1, "TST", List.of("Lim-Dûl's \"Vault\"")).toWireLine();
        assertTrue(line.contains("Lim-Dûl's \\\"Vault\\\""));  // accented passes, quote escaped
    }

    // ── Response parsing (flat JSON) ─────────────────────────────────────────

    @Test
    void parseFlatJsonTypesValues() {
        Map<String, Object> m = FlatJson.parse(
                "{\"draft_id\":\"d1\",\"seat\":3,\"pick\":\"Card B\",\"abort\":false}");
        assertEquals("d1", m.get("draft_id"));
        assertEquals(3L, m.get("seat"));
        assertEquals("Card B", m.get("pick"));
        assertEquals(Boolean.FALSE, m.get("abort"));
    }

    @Test
    void aMisspelledLiteralIsRejectedRatherThanSkipped() {
        // "tru" and "nul" are shorter than the literals they imitate, and
        // "trux" is the right length: all must fail, none may be jumped over.
        for (String bad : List.of("tru", "trux", "falsy", "nul", "nulx")) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> FlatJson.parse("{\"abort\":" + bad + "}"), bad);
            assertTrue(e.getMessage().startsWith("expected '"), e.getMessage());
        }
    }

    @Test
    void parseExternalAgentsSplitsAndTrims() {
        assertEquals(java.util.Set.of("a", "b"),
                DraftWorkerMain.parseExternalAgents(" a , b "));
        assertTrue(DraftWorkerMain.parseExternalAgents(null).isEmpty());
        assertTrue(DraftWorkerMain.parseExternalAgents("  ").isEmpty());
    }

    @Test
    void parseResponseRejectsMissingSentinel() {
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> DraftPickProtocol.parseResponse("{\"pick\":\"X\"}"));  // no sentinel
    }

    // ── resolvePick: routing + membership + abort ────────────────────────────

    @Test
    void resolvePickReturnsHeldPackCard() {
        Map<String, Object> resp = DraftPickProtocol.parseResponse(
                response("d1", 0, 1, 5, "\"pick\":\"Card B\""));
        String name = DraftPickProtocol.resolvePick(
                resp, request("d1", 0, 1, 5, List.of("Card A", "Card B", "Card C")));
        assertEquals("Card B", name);
    }

    @Test
    void resolvePickAbortThrowsAborted() {
        Map<String, Object> resp = DraftPickProtocol.parseResponse(
                response("d1", 0, 1, 5, "\"abort\":true"));
        assertThrows(DraftPickProtocol.DraftAbortedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A"))));
    }

    @Test
    void resolvePickRoutingMismatchThrowsAbandoned() {
        Map<String, Object> resp = DraftPickProtocol.parseResponse(
                response("d1", 7, 1, 5, "\"pick\":\"Card A\""));  // seat 7 != 0
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A"))));
    }

    @Test
    void resolvePickNotInPackThrowsAbandoned() {
        Map<String, Object> resp = DraftPickProtocol.parseResponse(
                response("d1", 0, 1, 5, "\"pick\":\"Not In Pack\""));
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A", "Card B"))));
    }

    @Test
    void resolvePickMissingPickThrowsAbandoned() {
        Map<String, Object> resp = DraftPickProtocol.parseResponse(
                response("d1", 0, 1, 5, "\"note\":\"none\""));  // neither pick nor abort
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A"))));
    }
}
