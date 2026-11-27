package com.pricepredictor.connector;

import com.pricepredictor.connector.DraftPickProtocol.PickPosition;
import com.pricepredictor.connector.DraftPickProtocol.PickRequest;
import com.pricepredictor.connector.json.Json;
import forge.gamemodes.limited.BoosterDraft;
import forge.gamemodes.limited.DraftPack;
import forge.gamemodes.limited.LimitedPlayerAI;
import forge.item.PaperCard;

import static com.pricepredictor.connector.DraftPickProtocol.requestExternalPick;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** One draft in progress: the seats, their agents, and every pick so far. */
final class PodDraft {

    /** One pod position: its index, its Forge AI seat, and the agent drafting it. */
    record Seat(int index, LimitedPlayerAI player, String agent) {
    }

    private final DraftWorkerMain.DraftSettings settings;
    private final String setCode;
    private final String draftId;
    private final int packSize;
    private final List<Seat> seats;
    /** The cards drafted from each booster, by booster index, in pick order. */
    private final List<List<String>> picksByBooster = new ArrayList<>(PodGeometry.BOOSTERS);

    PodDraft(DraftWorkerMain.DraftSettings settings, String setCode, String draftId, int packSize) {
        this.settings = settings;
        this.setCode = setCode;
        this.draftId = draftId;
        this.packSize = packSize;
        // One AI seat per pod position; agents sampled per draft (021 FR-006).
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
        if (requiredAgent == null || requiredAgent.isBlank()) {
            return agents;
        }
        if (agents.contains(requiredAgent)) {
            return agents;
        }
        List<String> forced = new ArrayList<>(agents);
        forced.set(random.nextInt(forced.size()), requiredAgent);
        return forced;
    }

    /**
     * Decide a seat's pick: the Forge AI's choice ({@code forge-full}), or a
     * uniform-random legal card for the random-override agents
     * ({@code forge-r30} replaces 30% of picks, {@code forge-r100} all of them).
     */
    private static PaperCard decidePick(
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
