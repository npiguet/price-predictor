package com.pricepredictor.connector;

import com.pricepredictor.connector.DraftPickProtocol.DraftAbandonedException;
import com.pricepredictor.connector.DraftPickProtocol.DraftAbortedException;
import com.pricepredictor.connector.DraftPickProtocol.PickChannel;
import com.pricepredictor.connector.DraftPickProtocol.PickPosition;
import com.pricepredictor.connector.DraftPickProtocol.PickRequest;
import com.pricepredictor.connector.DraftPickProtocol.WorkerInputClosed;
import com.pricepredictor.connector.effects.Json;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.DraftPack;
import forge.gamemodes.limited.LimitedPlayerAI;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

import static com.pricepredictor.connector.DraftPickProtocol.formatAbandoned;
import static com.pricepredictor.connector.DraftPickProtocol.requestExternalPick;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
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
 *   <li>{@code -Ddraft.set=<CODE>} — optional; restrict every draft to one set,
 *       recorded under its edition's code when named by an alias.
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
    static final String AGENT_MIX = "draft.agent.mix";
    static final String SET = "draft.set";
    static final String EXTERNAL_AGENTS = "draft.external.agents";
    static final String REQUIRED_AGENT = "draft.required.agent";

    public static void main(String[] args) {
        // The sentinel channel is UTF-8 so Python can decode accented card
        // names; diagnostics keep using System.err.
        PrintStream out = ToolMain.claimStdout();

        String agentMixSpec = WorkerProperties.require(AGENT_MIX);
        String setCode = WorkerProperties.optional(SET).orElse(null);  // null => random per draft
        Set<String> externalAgents = Set.copyOf(
                WorkerProperties.listProperty(EXTERNAL_AGENTS));
        String requiredAgent = WorkerProperties.optional(REQUIRED_AGENT).orElse(null);
        PickChannel channel = new PickChannel(out, new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8)), externalAgents);

        AgentMix mix = AgentMix.parse(agentMixSpec);

        System.err.println("Initializing Forge environment...");
        ForgeEnvironmentInitializer.initialize();
        System.err.println("Forge initialized. Starting draft generation.");

        Random random = MyRandom.getRandom();

        SetChooser sets;
        try {
            sets = SetChooser.of(setCode, random);
        } catch (IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            throw ToolMain.exit(1);
        }

        DraftSettings settings = new DraftSettings(
                BoosterDraft.createDraft(LimitedPoolType.Full), mix, random, channel,
                requiredAgent);

        WorkerLoop.runUntilBroken(
                () -> {
                    String draftSet = sets.next();
                    DraftResult result;
                    try {
                        result = generateDraft(settings, draftSet, UUID.randomUUID().toString());
                    } catch (WorkerInputClosed e) {
                        // The supervisor closed our stdin (it stopped or crashed); exit
                        // so it can restart a fresh worker if it still wants more drafts.
                        System.err.println("Pick channel closed: " + e.getMessage());
                        throw ToolMain.exit(0);
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
     * What every draft of one worker shares: the Forge context its AI seats
     * hang off, the agent mix each seat is drawn from, the random stream the
     * draws and random picks take, and the channel model-piloted seats ask
     * through.
     *
     * @param context       shared context object for the AI seats (used only
     *                      by conspiracy-card hooks, which standard sets do
     *                      not trigger)
     * @param requiredAgent a label at least one seat of every pod must carry,
     *                      or null (or blank) for none; see
     *                      {@link #withRequiredAgent}
     */
    record DraftSettings(
            BoosterDraft context, AgentMix mix, Random random, PickChannel channel,
            String requiredAgent) {
    }

    /** One pod position: its index, its Forge AI seat, and the agent drafting it. */
    record Seat(int index, LimitedPlayerAI player, String agent) {
    }

    /**
     * {@code agents} with one uniformly-chosen seat overwritten by
     * {@code requiredAgent} when the per-seat draw produced none (spec 021
     * FR-003).
     *
     * <p>{@code agents} itself, unchanged — and, deliberately, with no draw
     * from {@code random} at all — when the label is absent, blank, or
     * already present, so a run without the property samples byte-for-byte as
     * it always did. Applied before any pick is made, so the emitted
     * transcript's {@code seats[].agent} array always reflects the realised
     * pod.
     */
    static List<String> withRequiredAgent(
            List<String> agents, String requiredAgent, Random random) {
        if (requiredAgent == null || requiredAgent.isBlank() || agents.contains(requiredAgent)) {
            return agents;
        }
        List<String> forced = new ArrayList<>(agents);
        forced.set(random.nextInt(forced.size()), requiredAgent);
        return forced;
    }

    /**
     * Run one full pod draft. On a worker-detected pick fault this emits a
     * {@code <<DRAFT-ABANDONED>>} line via the channel first.
     *
     * @param requested the set as the caller names it; an alias such as
     *                  {@code CON} is drafted and recorded, in the
     *                  transcript and in every pick request, under the
     *                  code its edition goes by ({@code CFX}), because
     *                  every downstream join compares set codes as strings
     */
    static DraftResult generateDraft(DraftSettings settings, String requested, String draftId) {
        String setCode = BoosterTemplates.canonicalCode(requested);
        Optional<SealedTemplate> template = BoosterTemplates.forSet(setCode);
        if (template.isEmpty()) {
            return new DraftResult.Skipped("no booster template for set " + setCode);
        }
        DraftPack[] boosters = openBoosters(template.get());
        Optional<String> problem = boosterGeometryProblem(boosters);
        if (problem.isPresent()) {
            return new DraftResult.Skipped(problem.get());
        }

        PodDraft pod = new PodDraft(settings, setCode, draftId, boosters[0].size());
        try {
            for (int pack = 1; pack <= PodGeometry.PACKS; pack++) {
                pod.runPack(pack, boosters);
            }
        } catch (DraftAbortedException e) {
            // Supervisor-initiated abandonment (it already knows); drop silently.
            return new DraftResult.Abandoned();
        } catch (DraftAbandonedException e) {
            // Worker-detected fault: tell the supervisor (it logs + counts it).
            settings.channel().out().println(formatAbandoned(draftId, e.getMessage()));
            settings.channel().out().flush();
            return new DraftResult.Abandoned();
        }
        return new DraftResult.Completed(SENTINEL + pod.toJson());
    }

    /**
     * Why the opened boosters cannot be drafted into a transcript, or empty
     * when they can. The reader recovers every pick's seat from a single
     * pack size ({@link PodGeometry}), so the boosters must all hold the
     * same number of cards, and at least one.
     */
    static Optional<String> boosterGeometryProblem(DraftPack[] boosters) {
        int packSize = boosters[0].size();
        for (DraftPack booster : boosters) {
            if (booster.size() != packSize) {
                return Optional.of("boosters of " + booster.size() + " and "
                        + packSize + " cards; the transcript geometry needs one size");
            }
        }
        if (packSize == 0) {
            return Optional.of("empty boosters");
        }
        return Optional.empty();
    }

    /** Every booster of the pod, at {@link PodGeometry#boosterIndex}. */
    private static DraftPack[] openBoosters(SealedTemplate template) {
        DraftPack[] boosters = new DraftPack[PodGeometry.BOOSTERS];
        for (int k = 0; k < boosters.length; k++) {
            boosters[k] = new DraftPack(new ArrayList<>(new UnOpenedProduct(template).get()), k);
        }
        return boosters;
    }

    /** One draft in progress: the seats, their agents, and every pick so far. */
    private static final class PodDraft {
        private final DraftSettings settings;
        private final String setCode;
        private final String draftId;
        private final int packSize;
        private final List<Seat> seats;
        /** The cards drafted from each booster, by booster index, in pick order. */
        private final List<List<String>> picksByBooster = new ArrayList<>(PodGeometry.BOOSTERS);

        PodDraft(DraftSettings settings, String setCode, String draftId, int packSize) {
            this.settings = settings;
            this.setCode = setCode;
            this.draftId = draftId;
            this.packSize = packSize;
            // One AI seat per pod position; agents sampled per draft (FR-006).
            List<LimitedPlayerAI> players = new ArrayList<>(PodGeometry.POD_SIZE);
            List<String> agents = new ArrayList<>(PodGeometry.POD_SIZE);
            for (int s = 0; s < PodGeometry.POD_SIZE; s++) {
                players.add(new LimitedPlayerAI(s, settings.context()));
                agents.add(settings.mix().sample(settings.random()));
            }
            agents = withRequiredAgent(agents, settings.requiredAgent(), settings.random());
            List<Seat> pod = new ArrayList<>(PodGeometry.POD_SIZE);
            for (int s = 0; s < PodGeometry.POD_SIZE; s++) {
                pod.add(new Seat(s, players.get(s), agents.get(s)));
            }
            this.seats = List.copyOf(pod);
            for (int k = 0; k < PodGeometry.BOOSTERS; k++) {
                picksByBooster.add(new ArrayList<>(packSize));
            }
        }

        /** Drain one pack round: every seat picks, then the packs pass. */
        void runPack(int pack, DraftPack[] boosters) {
            DraftPack[] held = new DraftPack[PodGeometry.POD_SIZE];
            for (int o = 0; o < PodGeometry.POD_SIZE; o++) {
                held[o] = boosters[PodGeometry.boosterIndex(pack, o)];
            }
            for (int pick = 1; pick <= packSize; pick++) {
                DraftPack[] nextHeld = new DraftPack[PodGeometry.POD_SIZE];
                for (Seat seat : seats) {
                    nextHeld[seat.index()] = pickFor(seat, pack, pick, held[seat.index()]);
                }
                for (int s = 0; s < PodGeometry.POD_SIZE; s++) {
                    held[PodGeometry.nextSeat(s, pack)] = nextHeld[s];
                }
            }
        }

        /** {@code seat} picks from {@code current}; returns the pack it passes on. */
        private DraftPack pickFor(Seat seat, int pack, int pick, DraftPack current) {
            LimitedPlayerAI player = seat.player();
            player.receiveOpenedPack(current);  // make it the player's choice pack
            HeldPack held = new HeldPack(current);
            PaperCard chosen = settings.channel().routes(seat.agent())
                    ? requestExternalPick(settings.channel(), new PickRequest(
                            new PickPosition(draftId, seat.index(), pack, pick),
                            seat.agent(), PodGeometry.POD_SIZE, setCode, held.names()), held)
                    : decidePick(player, current, seat.agent(), settings.random());
            picksByBooster.get(current.getId()).add(chosen.getName());
            player.draftCard(chosen);
            return player.passPack();  // pull the pack back out
        }

        /** The transcript the supervisor reads, without the sentinel. */
        String toJson() {
            return Json.object()
                    .value("draft_id", draftId)
                    .raw("boosters", Json.array(picksByBooster, picks -> Json.object()
                            .value("set_code", setCode)
                            .raw("picks", Json.stringArray(picks))
                            .render()))
                    .raw("seats", Json.array(seats, seat -> Json.object()
                            .value("agent", seat.agent())
                            .render()))
                    .render();
        }
    }

    /**
     * Decide a seat's pick: the Forge AI's choice ({@code forge-full}), or a
     * uniform-random legal card for the random-override agents
     * ({@code forge-r30} replaces 30% of picks, {@code forge-r100} all of them).
     */
    static PaperCard decidePick(
            LimitedPlayerAI player, DraftPack pack, String agent, Random random) {
        double randomFraction = DraftAgent.randomOverrideFraction(agent);
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
}
