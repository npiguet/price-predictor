package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.CardTraitBase;

import java.util.ArrayList;
import java.util.List;

/**
 * One {@code trigger} record per evaluated trigger condition, fired or not.
 *
 * <p>The condition hook is the busiest in the collector, and most of what it
 * reports is a condition that did not fire; {@link TriggerNegativeSampler}
 * decides which of those are worth a record.
 *
 * <p>A condition that held is a firing only once Forge runs the trigger
 * ({@code onTriggerRun}, which hands back the very trigger and run
 * parameters the evaluation was handed): its record, read on the board at
 * the evaluation, waits for that run, and an evaluation never run (a trigger
 * Forge dropped before running it) writes no record. Several runs of one
 * evaluation (Panharmonicon) confirm it once.
 */
final class TriggerChannel {

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    private final TriggerNegativeSampler negatives;
    /** Each acting line's fired records, held until the game ends. */
    private final RecordReservoir positives;
    /**
     * The firings evaluated and not yet run, with the reservoir slot each
     * drew and its record, read on the board at the evaluation. Few: Forge
     * runs a trigger right after the evaluation that fired it.
     */
    private final List<Firing> unconfirmed = new ArrayList<>();
    /**
     * Whether Forge tells a trigger's run: a jar without {@code onTriggerRun}
     * never does, and there an evaluation that fired is held at once.
     */
    private final boolean runsAreTold;

    /** An evaluation that fired, waiting for its run. */
    private record Firing(
            Object trigger, Object runParams, String key, int slot, String mode,
            EffectRecord record) {
    }

    TriggerChannel(
            LiveGame live, CollectorContext context, RecordDedup dedup,
            TriggerNegativeSampler negatives, RecordReservoir positives) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
        this.negatives = negatives;
        this.positives = positives;
        this.runsAreTold = PatchHooks.TRIGGER_LISTENER.listenerDeclares(
                PatchHooks.Callback.ON_TRIGGER_RUN.methodName());
    }

    /**
     * The condition hook's listener.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    PatchHooks.Listener handler() {
        return PatchHooks.TRIGGER_LISTENER.listener((callback, args) -> {
            if (callback == PatchHooks.Callback.ON_TRIGGER_RUN) {
                if (!live.foreign(args.get(0))) {
                    ran(args.get(0), args.get(1));
                }
                return;
            }
            // Checked first, before this evaluation can spend any of the
            // mode's negative-sampling budget (offer/keep below) -- a fork's
            // trigger evaluations must not skew the live game's ratio
            // bookkeeping any more than they must reach its shard. args[0] is
            // declared Trigger on Forge's side, always a CardTraitBase with a
            // real host.
            Object trait = args.get(0);
            if (live.foreign(trait)) {
                return;
            }
            boolean fired = Boolean.TRUE.equals(args.get(2));
            String mode = RunParamNormalizer.modeOf(trait);
            if (mode == null) {
                mode = RunParamNormalizer.UNKNOWN_MODE;
            }
            if (!negatives.offer(mode, fired)) {
                return;
            }
            ProvenanceKey.Resolved line = LiveGame.lineOf(trait);
            String key = lineKey(trait, line, mode);
            if (fired) {
                evaluatedFiring(trait, line, args.get(1), mode, key);
                return;
            }
            // The line's flood cap is asked before the snapshot is built, so
            // a line at its cap costs a key lookup rather than a whole board.
            if (!negatives.lineHasRoomForNegative(key)) {
                return;
            }
            if (!context.emitDistinct(record(trait, line, args.get(1), fired), dedup)) {
                return;
            }
            // Counted only for a record that is actually written, so a
            // suppressed duplicate spends neither the negatives' quota nor
            // the line's cap, and the ratio holds over what reaches the shard.
            negatives.keep(mode, key, fired);
        });
    }

    /**
     * The key a line's reservoir and negative cap are kept under: its
     * provenance key, or, for a trigger that keys to no printed line (an
     * Aura's or Equipment's granted trigger, an effect card's, a delayed or
     * keyword trigger), its reason, host card, mode and parameters, so
     * unrelated unresolved triggers do not share one reservoir (C2 #16).
     */
    static String lineKey(Object trait, ProvenanceKey.Resolved line, String mode) {
        if (line.key() != null) {
            return line.key().describe();
        }
        CardTraitBase base = trait instanceof CardTraitBase ctb ? ctb : null;
        String host = base == null || base.getHostCard() == null
                ? "?" : base.getHostCard().getName();
        return "unresolved " + line.reason() + " " + host + " " + mode + " "
                + (base == null ? "?" : base.getMapParams());
    }

    /**
     * A firing evaluated: drawn for its line's reservoir now, and built on
     * this board when it takes a slot, but held only once Forge runs it.
     */
    private void evaluatedFiring(
            Object trait, ProvenanceKey.Resolved line, Object runParams, String mode,
            String key) {
        int slot = positives.slot(key);
        if (slot < 0) {
            return;
        }
        unconfirmed.add(new Firing(
                trait, runParams, key, slot, mode, record(trait, line, runParams, true)));
        if (!runsAreTold) {
            ran(trait, runParams);
        }
    }

    /**
     * Forge ran {@code trigger} on {@code runParams}: the evaluation handed
     * those very objects fired, and is held in its line's reservoir; a new
     * positive for its mode's balance only when it fills a free place.
     */
    private void ran(Object trigger, Object runParams) {
        for (int i = 0; i < unconfirmed.size(); i++) {
            Firing firing = unconfirmed.get(i);
            if (firing.trigger() == trigger && firing.runParams() == runParams) {
                unconfirmed.remove(i);
                // A slot past the records held is a free place, whatever an
                // earlier build that threw left unfilled (C2 #13).
                boolean fills = firing.slot() >= positives.held(firing.key());
                positives.hold(firing.key(), firing.slot(), firing.record());
                if (fills) {
                    negatives.keep(firing.mode(), firing.key(), true);
                }
                return;
            }
        }
    }

    /** Whether a firing evaluated here waits for its run; for a test. */
    boolean awaitsRun() {
        return !unconfirmed.isEmpty();
    }

    /**
     * Write the fired records the reservoirs kept, once the game can fire no
     * more, each unless it repeats a record already written; an evaluation
     * never run writes nothing.
     */
    void flush() {
        unconfirmed.clear();
        for (EffectRecord record : positives.drain()) {
            context.emitDistinctAtGameEnd(record, dedup);
        }
    }

    /**
     * One evaluated condition's record, built from the trigger itself.
     *
     * <p>The handler builds it after the sampling gate and the line's cap, so
     * an evaluation the sampler refuses never pays for resolving its key, and
     * one at its line's cap never pays for the snapshot — which matters,
     * because this is the hook that fires most often in the whole collector.
     */
    EffectRecord record(Object trait, Object runParams, boolean fired) {
        return record(trait, LiveGame.lineOf(trait), runParams, fired);
    }

    private EffectRecord record(
            Object trait, ProvenanceKey.Resolved line, Object runParams, boolean fired) {
        EffectEvent event = RunParamNormalizer.describeParams(trait, runParams);
        return live.traitRecord(context, RecordKind.TRIGGER, trait, line, event)
                .payload(Json.object()
                        .raw("event", event.toJson())
                        .value("fired", fired)
                        .render());
    }
}
