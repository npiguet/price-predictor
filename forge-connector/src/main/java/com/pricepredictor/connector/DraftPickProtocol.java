package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.gamemodes.limited.DraftPack;
import forge.item.PaperCard;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The live model-pilot side-channel of the draft worker
 * (contracts/pick-protocol.md): request and response lines, their validation,
 * and the faults that end a draft.
 */
final class DraftPickProtocol {

    static final String PICK_REQUEST_SENTINEL = "<<DRAFT-PICK-REQUEST>>";
    static final String PICK_RESPONSE_SENTINEL = "<<DRAFT-PICK-RESPONSE>>";
    static final String ABANDONED_SENTINEL = "<<DRAFT-ABANDONED>>";

    private DraftPickProtocol() {
    }

    /**
     * The pick side-channel to the supervisor: where requests are written,
     * where responses are read, and which agent labels are routed through it.
     */
    record PickChannel(PrintStream out, BufferedReader in, Set<String> externalAgents) {

        PickChannel {
            externalAgents = Set.copyOf(externalAgents);
        }

        /** Whether a seat drafting as {@code agent} asks the supervisor for its picks. */
        boolean routes(String agent) {
            return externalAgents.contains(agent);
        }
    }

    /** A worker-detected fault: abandon the draft, telling the supervisor. */
    static final class DraftAbandonedException extends RuntimeException {
        DraftAbandonedException(String message) {
            super(message);
        }
    }

    /** Supervisor-initiated abandonment ({@code "abort":true}); drop silently. */
    static final class DraftAbortedException extends RuntimeException {
    }

    /** stdin closed/EOF awaiting a response; the worker must exit. */
    static final class WorkerInputClosed extends RuntimeException {
        WorkerInputClosed(String message) {
            super(message);
        }
    }

    /**
     * One outstanding pick request: the seat asking, where it is in the draft,
     * and the pack it holds. A response must name the same draft, seat, pack
     * and pick, and choose a card from {@code pack}.
     */
    record PickRequest(
            String draftId, int seat, String agent, int podSize, int packNumber,
            int pickNumber, String setCode, List<String> pack) {

        PickRequest {
            pack = List.copyOf(pack);
        }

        /** The {@code <<DRAFT-PICK-REQUEST>>} line (contracts/pick-protocol.md). */
        String toWireLine() {
            return PICK_REQUEST_SENTINEL
                    + "{\"draft_id\":" + Json.string(draftId)
                    + ",\"seat\":" + seat
                    + ",\"agent\":" + Json.string(agent)
                    + ",\"pod_size\":" + podSize
                    + ",\"pack_number\":" + packNumber
                    + ",\"pick_number\":" + pickNumber
                    + ",\"set_code\":" + Json.string(setCode == null ? "" : setCode)
                    + ",\"pack\":" + Json.stringArray(pack)
                    + "}";
        }
    }

    /**
     * Ask the supervisor for a model-piloted seat's pick. Emits the request,
     * blocks on the channel for the matching response, validates it, and
     * returns the chosen card from the held pack.
     */
    static PaperCard requestExternalPick(
            PickChannel channel, PickRequest request, DraftPack pack) {
        channel.out().println(request.toWireLine());
        channel.out().flush();

        String line;
        try {
            line = readResponseLine(channel.in());
        } catch (IOException e) {
            throw new WorkerInputClosed("stdin read failed: " + e.getMessage());
        }
        if (line == null) {
            throw new WorkerInputClosed("stdin EOF awaiting pick response");
        }
        String name = resolvePick(parseResponse(line), request);
        for (int i = 0; i < pack.size(); i++) {
            if (pack.get(i).getName().equals(name)) {
                return pack.get(i);
            }
        }
        // Unreachable: resolvePick already validated held-pack membership.
        throw new DraftAbandonedException("resolved pick '" + name + "' not in held pack");
    }

    private static String readResponseLine(BufferedReader stdin) throws IOException {
        String line;
        while ((line = stdin.readLine()) != null) {
            if (!line.isBlank()) {
                return line;
            }
        }
        return null;  // EOF
    }

    static Map<String, Object> parseResponse(String line) {
        if (!line.startsWith(PICK_RESPONSE_SENTINEL)) {
            throw new DraftAbandonedException("response missing sentinel: " + line);
        }
        String suffix = line.substring(PICK_RESPONSE_SENTINEL.length()).trim();
        try {
            return FlatJson.parse(suffix);
        } catch (RuntimeException e) {
            throw new DraftAbandonedException("unparseable pick response: " + e.getMessage());
        }
    }

    /**
     * Validate a parsed response against the outstanding request and return the
     * chosen card name. Throws {@link DraftAbortedException} on {@code abort},
     * {@link DraftAbandonedException} on a routing mismatch or an out-of-pack pick.
     */
    static String resolvePick(Map<String, Object> response, PickRequest request) {
        if (Boolean.TRUE.equals(response.get("abort"))) {
            throw new DraftAbortedException();
        }
        if (!request.draftId().equals(response.get("draft_id"))
                || !eqInt(response.get("seat"), request.seat())
                || !eqInt(response.get("pack_number"), request.packNumber())
                || !eqInt(response.get("pick_number"), request.pickNumber())) {
            throw new DraftAbandonedException(
                    "response routing mismatch at seat " + request.seat() + " pack "
                            + request.packNumber() + " pick " + request.pickNumber());
        }
        Object pick = response.get("pick");
        if (!(pick instanceof String)) {
            throw new DraftAbandonedException("response carried no pick");
        }
        if (!request.pack().contains(pick)) {
            throw new DraftAbandonedException("pick '" + pick + "' not in held pack");
        }
        return (String) pick;
    }

    private static boolean eqInt(Object value, int expected) {
        return value instanceof Long && ((Long) value) == (long) expected;
    }

    static String formatAbandoned(String draftId, String reason) {
        return ABANDONED_SENTINEL + "{\"draft_id\":" + Json.string(draftId)
                + ",\"reason\":" + Json.string(reason == null ? "" : reason) + "}";
    }
}
