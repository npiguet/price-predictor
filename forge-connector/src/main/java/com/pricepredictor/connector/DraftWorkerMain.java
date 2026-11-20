package com.pricepredictor.connector;

import com.pricepredictor.connector.DraftPickProtocol.DraftAbandonedException;
import com.pricepredictor.connector.DraftPickProtocol.DraftAbortedException;
import com.pricepredictor.connector.DraftPickProtocol.PickChannel;
import com.pricepredictor.connector.DraftPickProtocol.WorkerInputClosed;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.DraftPack;
import forge.gamemodes.limited.LimitedPoolType;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

import static com.pricepredictor.connector.DraftPickProtocol.formatAbandoned;

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
    /** A progress line every this many drafts. */
    static final int PROGRESS_EVERY = 10;

    public static void main(String[] args) {
        ToolMain.guard(() -> work(args));
    }

    /** The worker itself; an exit it requests ends the process through {@link ToolMain#guard}. */
    static void work(String[] args) {
        // The sentinel channel is UTF-8 so Python can decode accented card
        // names; diagnostics keep using System.err.
        PrintStream out = ToolMain.claimStdout();

        DraftWorkerConfig config = DraftWorkerConfig.fromProperties();
        Set<String> externalAgents = config.externalAgents();
        PickChannel channel = new PickChannel(out, new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8)), externalAgents);

        AgentMix mix;
        try {
            mix = AgentMix.parse(config.agentMixSpec());
            DraftAgent.requireKnown(mix.labels(), externalAgents);
        } catch (IllegalArgumentException e) {
            throw ToolMain.badConfig("-D" + AGENT_MIX + " " + e.getMessage());
        }

        // stderr: stdout carries nothing but the protocol's sentinel lines.
        ForgeEnvironmentInitializer.initializeWorker(System.err, "Starting draft generation.");

        Random random = MyRandom.getRandom();

        SetChooser sets;
        try {
            sets = SetChooser.of(config.setCode(), random);
        } catch (IllegalStateException e) {
            throw ToolMain.badConfig(e.getMessage());
        }

        DraftSettings settings = new DraftSettings(
                BoosterDraft.createDraft(LimitedPoolType.Full), mix, random, channel,
                config.requiredAgent());

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
                        throw ToolMain.exit(ExitCode.OK);
                    }
                    return emit(result, draftSet, out);
                },
                // stderr: stdout carries nothing but the protocol's sentinel lines.
                new WorkerLoop.Progress(
                        System.err, PROGRESS_EVERY, "drafts generated", "Error generating draft"));
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
     *                      {@link PodDraft#withRequiredAgent}
     */
    record DraftSettings(
            BoosterDraft context, AgentMix mix, Random random, PickChannel channel,
            String requiredAgent) {
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
}
