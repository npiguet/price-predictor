package com.pricepredictor.connector;

import com.pricepredictor.connector.DraftPickProtocol.PickPosition;
import com.pricepredictor.connector.DraftPickProtocol.PickResponse;
import com.pricepredictor.connector.json.FlatJson;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure (Forge-free) tests for the live-pick side-channel protocol in
 * {@link DraftPickProtocol}: request formatting, response parsing into
 * {@link PickResponse}, position validation, held-pack membership, and the
 * abort/abandon outcomes (T009).
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
                new PickPosition(draftId, seat, packNumber, pickNumber),
                "draft-agent", 8, "BLB", pack);
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
    void formatPickRequestEscapesCardNames() {
        String line = new DraftPickProtocol.PickRequest(new PickPosition("d1", 0, 1, 1),
                "a", 8, "TST", List.of("Lim-Dûl's \"Vault\"")).toWireLine();
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
    void parseResponseRejectsMissingSentinel() {
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> PickResponse.parse("{\"pick\":\"X\"}"));  // no sentinel
    }

    @Test
    void aPickResponseCarriesItsPositionAndCard() {
        assertEquals(new PickResponse.Pick(new PickPosition("d1", 3, 2, 5), "Card B"),
                PickResponse.parse(response("d1", 3, 2, 5, "\"pick\":\"Card B\"")));
    }

    @Test
    void anAbortNeedsNothingElse() {
        assertEquals(new PickResponse.Abort(), PickResponse.parse(
                DraftPickProtocol.PICK_RESPONSE_SENTINEL + "{\"abort\":true}"));
    }

    @Test
    void aResponseWithAnIncompletePositionIsRefused() {
        for (String body : List.of(
                "{\"seat\":0,\"pack_number\":1,\"pick_number\":5,\"pick\":\"A\"}",
                "{\"draft_id\":\"d1\",\"seat\":\"0\",\"pack_number\":1,\"pick_number\":5,\"pick\":\"A\"}",
                "{\"draft_id\":\"d1\",\"seat\":0,\"pack_number\":1,\"pick\":\"A\"}",
                "{\"draft_id\":\"d1\",\"seat\":4294967296,\"pack_number\":1,\"pick_number\":5,"
                        + "\"pick\":\"A\"}")) {
            assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                    () -> PickResponse.parse(DraftPickProtocol.PICK_RESPONSE_SENTINEL + body), body);
        }
    }

    @Test
    void aResponseWithNoPickIsRefused() {
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> PickResponse.parse(response("d1", 0, 1, 5, "\"note\":\"none\"")));
    }

    // ── resolvePick: routing + membership + abort ────────────────────────────

    @Test
    void resolvePickReturnsHeldPackCard() {
        String name = DraftPickProtocol.resolvePick(
                PickResponse.parse(response("d1", 0, 1, 5, "\"pick\":\"Card B\"")),
                request("d1", 0, 1, 5, List.of("Card A", "Card B", "Card C")));
        assertEquals("Card B", name);
    }

    @Test
    void resolvePickAbortThrowsAborted() {
        PickResponse resp = PickResponse.parse(response("d1", 0, 1, 5, "\"abort\":true"));
        assertThrows(DraftPickProtocol.DraftAbortedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A"))));
    }

    @Test
    void resolvePickRoutingMismatchThrowsAbandoned() {
        for (String mismatched : List.of(
                response("d2", 0, 1, 5, "\"pick\":\"Card A\""),
                response("d1", 7, 1, 5, "\"pick\":\"Card A\""),
                response("d1", 0, 2, 5, "\"pick\":\"Card A\""),
                response("d1", 0, 1, 6, "\"pick\":\"Card A\""))) {
            PickResponse resp = PickResponse.parse(mismatched);
            assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                    () -> DraftPickProtocol.resolvePick(
                            resp, request("d1", 0, 1, 5, List.of("Card A"))), mismatched);
        }
    }

    @Test
    void resolvePickNotInPackThrowsAbandoned() {
        PickResponse resp = PickResponse.parse(response("d1", 0, 1, 5, "\"pick\":\"Not In Pack\""));
        assertThrows(DraftPickProtocol.DraftAbandonedException.class,
                () -> DraftPickProtocol.resolvePick(
                        resp, request("d1", 0, 1, 5, List.of("Card A", "Card B"))));
    }
}
