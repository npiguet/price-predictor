package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.item.PaperCard;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
     * Where in which draft a pick is made: the routing fields a request
     * carries and its response must echo. Strict synchrony means at most one
     * request is outstanding, so a response naming any other position is a
     * desynchronised supervisor, not a late answer.
     *
     * @param pack 1-based pack number
     * @param pick 1-based pick number within the pack
     */
    record PickPosition(String draftId, int seat, int pack, int pick) {

        /**
         * The position a parsed response names, or empty when any of its four
         * routing fields is missing or of the wrong type.
         */
        static Optional<PickPosition> of(Map<String, Object> fields) {
            if (!(fields.get("draft_id") instanceof String draftId)) {
                return Optional.empty();
            }
            Optional<Integer> seat = intField(fields, "seat");
            Optional<Integer> pack = intField(fields, "pack_number");
            Optional<Integer> pick = intField(fields, "pick_number");
            if (seat.isEmpty() || pack.isEmpty() || pick.isEmpty()) {
                return Optional.empty();
            }
            return Optional.of(new PickPosition(draftId, seat.get(), pack.get(), pick.get()));
        }

        private static Optional<Integer> intField(Map<String, Object> fields, String key) {
            if (fields.get(key) instanceof Long value
                    && value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
                return Optional.of(value.intValue());
            }
            return Optional.empty();
        }

        @Override
        public String toString() {
            return "seat " + seat + " pack " + pack + " pick " + pick;
        }
    }

    /**
     * One outstanding pick request: the seat asking, where it is in the draft,
     * and the pack it holds. A response must name the same position and
     * choose a card from {@code pack}.
     */
    record PickRequest(
            PickPosition position, String agent, int podSize, String setCode, List<String> pack) {

        PickRequest {
            pack = List.copyOf(pack);
        }

        /** The {@code <<DRAFT-PICK-REQUEST>>} line (contracts/pick-protocol.md). */
        String toWireLine() {
            return PICK_REQUEST_SENTINEL + Json.object()
                    .value("draft_id", position.draftId())
                    .value("seat", position.seat())
                    .value("agent", agent)
                    .value("pod_size", podSize)
                    .value("pack_number", position.pack())
                    .value("pick_number", position.pick())
                    .value("set_code", setCode == null ? "" : setCode)
                    .raw("pack", Json.stringArray(pack))
                    .render();
        }
    }

    /**
     * A {@code <<DRAFT-PICK-RESPONSE>>} line, parsed: the supervisor either
     * aborts the draft or picks a card at a position.
     */
    sealed interface PickResponse {

        /** {@code "abort":true}: the supervisor gives the draft up, whatever else the line says. */
        record Abort() implements PickResponse {}

        /** The supervisor's choice of {@code card} at {@code position}. */
        record Pick(PickPosition position, String card) implements PickResponse {}

        /**
         * Parse a response line.
         *
         * @throws DraftAbandonedException on a line that is not a well-formed
         *         response: no sentinel, unparseable JSON, no complete
         *         position, or no pick
         */
        static PickResponse parse(String line) {
            if (!line.startsWith(PICK_RESPONSE_SENTINEL)) {
                throw new DraftAbandonedException("response missing sentinel: " + line);
            }
            String suffix = line.substring(PICK_RESPONSE_SENTINEL.length()).trim();
            Map<String, Object> fields;
            try {
                fields = FlatJson.parse(suffix);
            } catch (RuntimeException e) {
                throw new DraftAbandonedException("unparseable pick response: " + e.getMessage());
            }
            if (Boolean.TRUE.equals(fields.get("abort"))) {
                return new Abort();
            }
            PickPosition position = PickPosition.of(fields).orElseThrow(
                    () -> new DraftAbandonedException("response carried no complete position: " + suffix));
            if (!(fields.get("pick") instanceof String card)) {
                throw new DraftAbandonedException("response carried no pick");
            }
            return new Pick(position, card);
        }
    }

    /**
     * Ask the supervisor for a model-piloted seat's pick. Emits the request,
     * blocks on the channel for the matching response, validates it, and
     * returns the chosen card from the held pack.
     */
    static PaperCard requestExternalPick(
            PickChannel channel, PickRequest request, HeldPack held) {
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
        String name = resolvePick(PickResponse.parse(line), request);
        // Unreachable miss: resolvePick already validated held-pack membership.
        return held.card(name).orElseThrow(() -> new DraftAbandonedException(
                "resolved pick '" + name + "' not in held pack"));
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

    /**
     * Validate a parsed response against the outstanding request and return the
     * chosen card name. Throws {@link DraftAbortedException} on an abort,
     * {@link DraftAbandonedException} on a routing mismatch or an out-of-pack pick.
     */
    static String resolvePick(PickResponse response, PickRequest request) {
        if (!(response instanceof PickResponse.Pick pick)) {
            throw new DraftAbortedException();
        }
        if (!request.position().equals(pick.position())) {
            throw new DraftAbandonedException(
                    "response routing mismatch at " + request.position());
        }
        if (!request.pack().contains(pick.card())) {
            throw new DraftAbandonedException("pick '" + pick.card() + "' not in held pack");
        }
        return pick.card();
    }

    static String formatAbandoned(String draftId, String reason) {
        return ABANDONED_SENTINEL + Json.object()
                .value("draft_id", draftId)
                .value("reason", reason == null ? "" : reason)
                .render();
    }
}
