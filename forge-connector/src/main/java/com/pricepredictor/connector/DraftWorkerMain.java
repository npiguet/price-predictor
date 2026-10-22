package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.DraftPack;
import forge.gamemodes.limited.LimitedPlayerAI;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

/**
 * Draft worker entry point.
 *
 * <p>Drives Forge's draft AI for all pod seats and streams one completed-draft
 * transcript per draft on stdout. The Python supervisor
 * ({@code draft.application.generate_draft_data}) reads these lines, builds and
 * scores a deck per seat, and appends one self-contained JSON record to
 * {@code drafts.jsonl}.
 *
 * <p>Transport (worker-protocol.md): one flushed line per completed draft,
 * <pre>{@code <<DRAFT-EVENT-JSON>>{"draft_id":..,"boosters":[..],"seats":[{"agent":..}]}}</pre>
 * Diagnostics and Forge's incidental logging go to stderr.
 *
 * <p>Live model-pilot side-channel (contracts/pick-protocol.md): when a seat's
 * sampled agent is in {@code -Ddraft.external.agents}, the worker emits a
 * <pre>{@code <<DRAFT-PICK-REQUEST>>{..}}</pre> line and blocks on stdin for the
 * matching <pre>{@code <<DRAFT-PICK-RESPONSE>>{..}}</pre> (the supervisor's
 * policy pick, or {@code "abort":true}). Strict synchrony holds: at most one
 * request is outstanding. A mismatched/garbled response abandons the in-flight
 * draft (emitting {@code <<DRAFT-ABANDONED>>}, no transcript); an {@code abort}
 * drops it silently; stdin EOF exits the worker (supervisor restart path).
 *
 * <p>System properties:
 * <ul>
 *   <li>{@code -Ddraft.agent.mix=<label:weight,...>} — required; each seat draws
 *       its agent independently per draft (FR-006).</li>
 *   <li>{@code -Ddraft.set=<CODE>} — optional; restrict every draft to one set.
 *       When absent, a random sealed-legal set is chosen per draft (FR-009).</li>
 *   <li>{@code -Ddraft.external.agents=<label,...>} — optional; the mix labels
 *       routed through the pick side-channel. Absent/empty ⇒ byte-for-byte
 *       gen-1 behavior (no pick requests are ever emitted, SC-004).</li>
 *   <li>{@code -Ddraft.required.agent=<label>} — optional; after the per-seat
 *       mix draw, if no seat carries this label one uniformly-chosen seat is
 *       overwritten with it. Guarantees every played pod carries at least one
 *       such seat (spec 021 FR-003) — a learner-free pod carries no on-policy
 *       picks and must never be drafted. Absent/blank ⇒ sampling is unchanged
 *       and the random stream is not touched.</li>
 * </ul>
 *
 * <p>The worker loops forever; the supervisor terminates it once it has enough
 * drafts (or on crash, restarts a fresh one).
 */
public class DraftWorkerMain {

    static final String SENTINEL = "<<DRAFT-EVENT-JSON>>";
    static final String PICK_REQUEST_SENTINEL = "<<DRAFT-PICK-REQUEST>>";
    static final String PICK_RESPONSE_SENTINEL = "<<DRAFT-PICK-RESPONSE>>";
    static final String ABANDONED_SENTINEL = "<<DRAFT-ABANDONED>>";
    static final int POD_SIZE = 8;
    static final int PACKS = 3;

    public static void main(String[] args) {
        // Bind the sentinel channel to the real stdout FD as UTF-8, then route
        // everything else printed to System.out (Forge's incidental logging) to
        // stderr. This guarantees the pipe the supervisor reads carries ONLY our
        // UTF-8 sentinel lines — no platform-encoded Forge chatter, no
        // mixed-encoding stream — so Python can decode it as UTF-8 (accented
        // card names included). Diagnostics keep using System.err.
        PrintStream out = new PrintStream(
                new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8);
        System.setOut(System.err);

        String agentMixSpec = WorkerProperties.require("draft.agent.mix");
        String setCode = System.getProperty("draft.set");  // null => random per draft
        Set<String> externalAgents = parseExternalAgents(
                System.getProperty("draft.external.agents"));
        String requiredAgent = System.getProperty("draft.required.agent");
        PickChannel channel = new PickChannel(out, new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8)), externalAgents);

        AgentMix mix = AgentMix.parse(agentMixSpec);

        System.err.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.err.println("Forge initialized. Starting draft generation.");

        Random random = MyRandom.getRandom();

        List<String> eligibleSets = null;
        if (setCode == null) {
            eligibleSets = MatchGenerator.computeEligibleSets();
            if (eligibleSets.isEmpty()) {
                System.err.println("Error: no eligible sealed-legal sets found");
                System.exit(1);
            }
        }

        // Shared context object for the AI seats (used only by conspiracy-card
        // hooks, which standard sets do not trigger).
        BoosterDraft context = BoosterDraft.createDraft(LimitedPoolType.Full);

        final List<String> sets = eligibleSets;
        WorkerLoop.runUntilBroken(
                () -> {
                    String draftSet = (setCode != null)
                            ? setCode
                            : sets.get(random.nextInt(sets.size()));
                    DraftResult result;
                    try {
                        result = generateDraft(
                                context, draftSet, mix, random, channel,
                                UUID.randomUUID().toString(), requiredAgent);
                    } catch (WorkerInputClosed e) {
                        // The supervisor closed our stdin (it stopped or crashed); exit
                        // so it can restart a fresh worker if it still wants more drafts.
                        System.err.println("Pick channel closed: " + e.getMessage());
                        System.exit(0);
                        return WorkerLoop.Outcome.ABANDONED;  // unreachable
                    }
                    return emit(result, draftSet, out);
                },
                // stderr: stdout carries nothing but the protocol's sentinel lines.
                new WorkerLoop.Progress(
                        System.err, 10, "drafts generated", "Error generating draft"));
    }

    /** Write a completed draft's transcript; say why a skipped one produced none. */
    static WorkerLoop.Outcome emit(DraftResult result, String setCode, PrintStream out) {
        if (result instanceof DraftResult.Completed completed) {
            out.println(completed.line());
            out.flush();
            return WorkerLoop.Outcome.COMPLETED;
        }
        if (result instanceof DraftResult.Skipped skipped) {
            System.err.println("Skipped a draft of " + setCode + ": " + skipped.reason());
            return WorkerLoop.Outcome.SKIPPED;
        }
        return WorkerLoop.Outcome.ABANDONED;
    }

    /**
     * What one {@link #generateDraft} call came to.
     *
     * <p>Three outcomes rather than a nullable line, because two of them must
     * be told apart: a skip is the worker's own doing and will recur — a set
     * with no booster template skips every draft — so the worker loop counts
     * it as a failure, while an abandoned draft is the supervisor's (it aborted,
     * or its response was malformed and it has been told) and counts as neither.
     */
    sealed interface DraftResult {
        /** A full transcript, sentinel included, ready for stdout. */
        record Completed(String line) implements DraftResult {}

        /** No draft was attempted, for {@code reason}. */
        record Skipped(String reason) implements DraftResult {}

        /** The draft was dropped part-way: the supervisor aborted it, or has been told why. */
        record Abandoned() implements DraftResult {}
    }

    /**
     * The pick side-channel to the supervisor: where requests are written,
     * where responses are read, and which agent labels are routed through it.
     */
    record PickChannel(PrintStream out, BufferedReader in, Set<String> externalAgents) {

        /** A Forge-only worker: every seat picks in-process. */
        static final PickChannel NONE = new PickChannel(null, null, Set.of());

        PickChannel {
            externalAgents = Set.copyOf(externalAgents);
        }

        /** Whether a seat drafting as {@code agent} asks the supervisor for its picks. */
        boolean routes(String agent) {
            return externalAgents.contains(agent);
        }
    }

    /**
     * Overwrite one uniformly-chosen seat with {@code requiredAgent} when the
     * per-seat draw produced none (spec 021 FR-003).
     *
     * <p>A no-op — and, deliberately, no draw from {@code random} at all — when
     * the label is absent, blank, or already present, so a run without the
     * property samples byte-for-byte as it always did. Mutates {@code agents} in
     * place before any pick is made, so the emitted transcript's
     * {@code seats[].agent} array always reflects the realised pod.
     */
    static void forceRequiredAgent(String[] agents, String requiredAgent, Random random) {
        if (requiredAgent == null || requiredAgent.isBlank()) {
            return;
        }
        for (String agent : agents) {
            if (requiredAgent.equals(agent)) {
                return;
            }
        }
        agents[random.nextInt(agents.length)] = requiredAgent;
    }

    /**
     * Run one full pod draft. On a worker-detected pick fault this emits a
     * {@code <<DRAFT-ABANDONED>>} line via the channel first.
     *
     * @param requiredAgent a label at least one seat must carry, or null
     */
    static DraftResult generateDraft(
            BoosterDraft context, String setCode, AgentMix mix, Random random,
            PickChannel channel, String draftId, String requiredAgent) {
        Optional<SealedTemplate> template = BoosterTemplates.forSet(setCode);
        if (template.isEmpty()) {
            return new DraftResult.Skipped("no booster template for set " + setCode);
        }
        DraftPack[] boosters = openBoosters(template.get());
        int packSize = boosters[0].size();
        for (DraftPack booster : boosters) {
            if (booster.size() != packSize) {
                // A non-uniform pack size breaks the reconstruction geometry.
                return new DraftResult.Skipped("boosters of " + booster.size() + " and "
                        + packSize + " cards; the transcript geometry needs one size");
            }
        }
        if (packSize == 0) {
            return new DraftResult.Skipped("empty boosters");
        }

        PodDraft pod = new PodDraft(context, setCode, draftId, mix, random, channel, packSize);
        forceRequiredAgent(pod.agents, requiredAgent, random);
        try {
            for (int pack = 1; pack <= PACKS; pack++) {
                pod.runPack(pack, boosters);
            }
        } catch (DraftAbortedException e) {
            // Supervisor-initiated abandonment (it already knows); drop silently.
            return new DraftResult.Abandoned();
        } catch (DraftAbandonedException e) {
            // Worker-detected fault: tell the supervisor (it logs + counts it).
            if (channel.out() != null) {
                channel.out().println(formatAbandoned(draftId, e.getMessage()));
                channel.out().flush();
            }
            return new DraftResult.Abandoned();
        }
        return new DraftResult.Completed(SENTINEL + pod.toJson());
    }

    /** Every booster of the pod: index k = (pack - 1) * POD_SIZE + openingSeat. */
    private static DraftPack[] openBoosters(SealedTemplate template) {
        DraftPack[] boosters = new DraftPack[PACKS * POD_SIZE];
        for (int k = 0; k < boosters.length; k++) {
            boosters[k] = new DraftPack(new ArrayList<>(new UnOpenedProduct(template).get()), k);
        }
        return boosters;
    }

    /** One draft in progress: the seats, their agents, and every pick so far. */
    private static final class PodDraft {
        private final String setCode;
        private final String draftId;
        private final Random random;
        private final PickChannel channel;
        private final int packSize;
        private final List<LimitedPlayerAI> players = new ArrayList<>(POD_SIZE);
        private final String[] agents = new String[POD_SIZE];
        /** The cards drafted from each booster, by booster index, in pick order. */
        private final List<List<String>> picksByBooster = new ArrayList<>(PACKS * POD_SIZE);

        PodDraft(BoosterDraft context, String setCode, String draftId, AgentMix mix,
                 Random random, PickChannel channel, int packSize) {
            this.setCode = setCode;
            this.draftId = draftId;
            this.random = random;
            this.channel = channel;
            this.packSize = packSize;
            // One AI seat per pod position; agents sampled per draft (FR-006).
            for (int s = 0; s < POD_SIZE; s++) {
                players.add(new LimitedPlayerAI(s, context));
                agents[s] = mix.sample(random);
            }
            for (int k = 0; k < PACKS * POD_SIZE; k++) {
                picksByBooster.add(new ArrayList<>(packSize));
            }
        }

        /** Drain one pack round: every seat picks, then the packs pass. */
        void runPack(int pack, DraftPack[] boosters) {
            int dir = (pack % 2 == 1) ? 1 : -1;  // pass left on odd packs, right on even
            DraftPack[] held = new DraftPack[POD_SIZE];
            for (int o = 0; o < POD_SIZE; o++) {
                held[o] = boosters[(pack - 1) * POD_SIZE + o];
            }
            for (int pick = 1; pick <= packSize; pick++) {
                DraftPack[] nextHeld = new DraftPack[POD_SIZE];
                for (int s = 0; s < POD_SIZE; s++) {
                    nextHeld[s] = pickFor(s, pack, pick, held[s]);
                }
                for (int s = 0; s < POD_SIZE; s++) {
                    held[(s + dir + POD_SIZE) % POD_SIZE] = nextHeld[s];
                }
            }
        }

        /** Seat {@code s} picks from {@code current}; returns the pack it passes on. */
        private DraftPack pickFor(int s, int pack, int pick, DraftPack current) {
            LimitedPlayerAI player = players.get(s);
            player.receiveOpenedPack(current);  // make it the player's choice pack
            PaperCard chosen = channel.routes(agents[s])
                    ? requestExternalPick(channel, new PickRequest(
                            draftId, s, agents[s], POD_SIZE, pack, pick, setCode,
                            namesIn(current)), current)
                    : decidePick(player, current, agents[s], random);
            picksByBooster.get(current.getId()).add(chosen.getName());
            player.draftCard(chosen);
            return player.passPack();  // pull the pack back out
        }

        /** The transcript the supervisor reads, without the sentinel. */
        String toJson() {
            StringBuilder sb = new StringBuilder(4096);
            sb.append("{\"draft_id\":\"").append(draftId).append("\",\"boosters\":[");
            for (int k = 0; k < picksByBooster.size(); k++) {
                if (k > 0) {
                    sb.append(',');
                }
                sb.append("{\"set_code\":").append(Json.string(setCode))
                        .append(",\"picks\":").append(Json.stringArray(picksByBooster.get(k)))
                        .append('}');
            }
            sb.append("],\"seats\":[");
            for (int s = 0; s < agents.length; s++) {
                if (s > 0) {
                    sb.append(',');
                }
                sb.append("{\"agent\":").append(Json.string(agents[s])).append('}');
            }
            sb.append("]}");
            return sb.toString();
        }
    }

    private static List<String> namesIn(DraftPack pack) {
        List<String> names = new ArrayList<>(pack.size());
        for (int i = 0; i < pack.size(); i++) {
            names.add(pack.get(i).getName());
        }
        return names;
    }

    /**
     * Decide a seat's pick: the Forge AI's choice ({@code forge-full}), or a
     * uniform-random legal card for the random-override agents
     * ({@code forge-r30} replaces 30% of picks, {@code forge-r100} all of them).
     */
    static PaperCard decidePick(
            LimitedPlayerAI player, DraftPack pack, String agent, Random random) {
        double randomFraction = randomOverrideFraction(agent);
        if (randomFraction > 0 && random.nextDouble() < randomFraction) {
            return pack.get(random.nextInt(pack.size()));
        }
        PaperCard aiPick = player.chooseCard();
        if (aiPick == null) {
            // Defensive: AI declined (shouldn't happen with a non-empty pack).
            return pack.get(random.nextInt(pack.size()));
        }
        return aiPick;
    }

    static double randomOverrideFraction(String agent) {
        return switch (agent) {
            case "forge-r30" -> 0.30;
            case "forge-r100" -> 1.0;
            default -> 0.0;  // forge-full and any unknown label: pure AI
        };
    }

    // ── Live pick side-channel ────────────────────────────────────────────────

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
            return parseFlatJson(suffix);
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

    static Set<String> parseExternalAgents(String spec) {
        if (spec == null || spec.isBlank()) {
            return Collections.emptySet();
        }
        Set<String> labels = new LinkedHashSet<>();
        for (String token : spec.split(",")) {
            String trimmed = token.trim();
            if (!trimmed.isEmpty()) {
                labels.add(trimmed);
            }
        }
        return labels;
    }

    static String formatAbandoned(String draftId, String reason) {
        return ABANDONED_SENTINEL + "{\"draft_id\":" + Json.string(draftId)
                + ",\"reason\":" + Json.string(reason == null ? "" : reason) + "}";
    }

    /** Minimal parser for a flat JSON object (string/integer/boolean/null values). */
    static Map<String, Object> parseFlatJson(String json) {
        Map<String, Object> map = new HashMap<>();
        int n = json.length();
        int i = skipWs(json, 0);
        if (i >= n || json.charAt(i) != '{') {
            throw new IllegalArgumentException("expected '{'");
        }
        i = skipWs(json, i + 1);
        if (i < n && json.charAt(i) == '}') {
            return map;
        }
        while (i < n) {
            i = skipWs(json, i);
            if (json.charAt(i) != '"') {
                throw new IllegalArgumentException("expected key string");
            }
            StringBuilder key = new StringBuilder();
            i = parseJsonString(json, i, key);
            i = skipWs(json, i);
            if (i >= n || json.charAt(i) != ':') {
                throw new IllegalArgumentException("expected ':'");
            }
            i = skipWs(json, i + 1);
            char c = json.charAt(i);
            if (c == '"') {
                StringBuilder val = new StringBuilder();
                i = parseJsonString(json, i, val);
                map.put(key.toString(), val.toString());
            } else if (c == 't' || c == 'f') {
                boolean b = c == 't';
                i += b ? 4 : 5;
                map.put(key.toString(), b);
            } else if (c == 'n') {
                i += 4;
                map.put(key.toString(), null);
            } else {
                int start = i;
                while (i < n && (Character.isDigit(json.charAt(i))
                        || json.charAt(i) == '-' || json.charAt(i) == '+')) {
                    i++;
                }
                map.put(key.toString(), Long.parseLong(json.substring(start, i)));
            }
            i = skipWs(json, i);
            if (i < n && json.charAt(i) == ',') {
                i++;
                continue;
            }
            if (i < n && json.charAt(i) == '}') {
                break;
            }
            throw new IllegalArgumentException("expected ',' or '}'");
        }
        return map;
    }

    private static int skipWs(String s, int i) {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
        return i;
    }

    /** Parse a JSON string starting at the opening quote; returns the index past it. */
    private static int parseJsonString(String s, int i, StringBuilder out) {
        i++;  // opening quote
        int n = s.length();
        while (i < n) {
            char c = s.charAt(i++);
            if (c == '"') {
                return i;
            }
            if (c == '\\') {
                char e = s.charAt(i++);
                switch (e) {
                    case '"' -> out.append('"');
                    case '\\' -> out.append('\\');
                    case '/' -> out.append('/');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'u' -> {
                        out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> out.append(e);
                }
            } else {
                out.append(c);
            }
        }
        throw new IllegalArgumentException("unterminated string");
    }

    /** Weighted categorical over agent labels, parsed from {@code label:weight,...}. */
    static final class AgentMix {
        private final String[] labels;
        private final int[] cumulative;
        private final int total;

        private AgentMix(String[] labels, int[] cumulative, int total) {
            this.labels = labels;
            this.cumulative = cumulative;
            this.total = total;
        }

        static AgentMix parse(String spec) {
            String[] tokens = spec.split(",");
            List<String> labels = new ArrayList<>();
            List<Integer> weights = new ArrayList<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.isEmpty()) {
                    continue;
                }
                int colon = token.indexOf(':');
                if (colon <= 0 || colon != token.lastIndexOf(':')) {
                    throw new IllegalArgumentException(
                            "agent-mix entry must be 'label:weight', got: " + token);
                }
                String label = token.substring(0, colon).trim();
                int weight = Integer.parseInt(token.substring(colon + 1).trim());
                if (label.isEmpty() || weight < 1) {
                    throw new IllegalArgumentException(
                            "agent-mix entry must have a non-empty label and weight >= 1: " + token);
                }
                labels.add(label);
                weights.add(weight);
            }
            if (labels.isEmpty()) {
                throw new IllegalArgumentException("agent-mix is empty: " + spec);
            }
            int[] cumulative = new int[labels.size()];
            int running = 0;
            for (int i = 0; i < weights.size(); i++) {
                running += weights.get(i);
                cumulative[i] = running;
            }
            return new AgentMix(labels.toArray(new String[0]), cumulative, running);
        }

        String sample(Random random) {
            int roll = random.nextInt(total);
            for (int i = 0; i < cumulative.length; i++) {
                if (roll < cumulative[i]) {
                    return labels[i];
                }
            }
            return labels[labels.length - 1];  // unreachable
        }
    }
}
