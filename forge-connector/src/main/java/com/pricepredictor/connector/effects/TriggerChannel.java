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
 * Forge dropped before running it) writes no record and counts for nothing in
 * its line's reservoir. Several runs of one evaluation (Panharmonicon)
 * confirm it once.
 */
final class TriggerChannel {

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    private final TriggerNegativeSampler negatives;
    /** Each acting line's fired records, held until the game ends. */
    private final RecordReservoir positives;
    /**
     * The firings evaluated and not yet run, each with the reservoir slot it
     * drew and, when that took a slot, its record, read on the board at the
     * evaluation. Few: Forge runs a trigger right after the evaluation that
     * fired it.
     */
    private final List<Firing> unconfirmed = new ArrayList<>();
    /**
     * Whether Forge tells a trigger's run: a jar without {@code onTriggerRun}
     * never does, and there an evaluation that fired is held at once.
     */
    private final boolean runsAreTold;

    /** One evaluated condition: its trigger and run parameters, its line, mode and the key its caps are kept under. */
    private record Evaluation(
            Object trait, ProvenanceKey.Resolved line, Object runParams, String mode, String key) {
    }

    /** An evaluation that fired, waiting for its run. */
    private record Firing(
            Object trigger, Object runParams, String key, String mode, int slot,
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
        this.runsAreTold = PatchHooks.TRIGGER_LISTENER.listenerDeclares(Callback.ON_TRIGGER_RUN);
    }

    /**
     * The condition hook's listener.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     *
     * <p>What the collector's own enumeration at a priority stop sets off is
     * dropped whole, evaluations and runs alike ({@link
     * LiveGame#enumerate}): its rules queries run Forge's static check, which
     * evaluates every state trigger, and none of that is the game's.
     */
    Listener handler() {
        return PatchHooks.TRIGGER_LISTENER.listener((callback, args) -> {
            if (!live.enumeratingHere()) {
                reported(callback, args);
            }
        });
    }

    /** A callback of the hook's, outside the collector's own enumeration. */
    private void reported(Callback callback, Args args) {
        if (callback == Callback.ON_TRIGGER_RUN) {
            if (!live.foreign(args.get(0))) {
                ran(args.get(0), args.get(1));
            }
            return;
        }
        evaluationReported(args);
    }

    /** One trigger condition evaluated: drawn against its mode's balance, then filed as a firing or a negative. */
    private void evaluationReported(Args args) {
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
        // A not-fired evaluation a directed fork keeps (FR-010c) is offered
        // whatever the mode's balance says, and repays none of it.
        boolean kept = !fired && context.kept().keepsNegative(trait);
        if (!kept && !negatives.offer(mode, fired)) {
            return;
        }
        ProvenanceKey.Resolved line = LiveGame.lineOf(trait);
        Evaluation evaluation =
                new Evaluation(trait, line, args.get(1), mode, lineKey(trait, line, mode));
        if (fired) {
            evaluatedFiring(evaluation);
        } else {
            evaluatedNegative(evaluation, kept);
        }
    }

    /** A not-fired evaluation the sampler took: written unless its line is full or it repeats a record. */
    private void evaluatedNegative(Evaluation evaluation, boolean kept) {
        // The line's flood cap is asked before the snapshot is built, so
        // a line at its cap costs a key lookup rather than a whole board.
        if (!negatives.lineHasRoomForNegative(evaluation.key())) {
            return;
        }
        EffectRecord record =
                record(evaluation.trait(), evaluation.line(), evaluation.runParams(), false);
        if (!context.emitDistinct(record, dedup)) {
            return;
        }
        // Counted only for a record that reaches the shard, so a
        // suppressed duplicate or one the per-game cap withheld spends
        // neither the negatives' quota nor the line's cap, and the ratio
        // holds over what reaches the shard.
        if (kept) {
            negatives.keepOutsideTheBalance(evaluation.key());
        } else {
            negatives.keep(evaluation.mode(), evaluation.key(), false);
        }
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
     * A firing evaluated: drawn for its line's reservoir now, against the
     * count its run would make, and built on this board only when the draw
     * takes a slot; counted in the reservoir only once Forge runs it, so an
     * evaluation that never runs leaves the count alone.
     */
    private void evaluatedFiring(Evaluation evaluation) {
        int slot = positives.drawFor(evaluation.key());
        unconfirmed.add(new Firing(
                evaluation.trait(), evaluation.runParams(), evaluation.key(), evaluation.mode(), slot,
                slot < 0 ? null : record(
                        evaluation.trait(), evaluation.line(), evaluation.runParams(), true)));
        if (!runsAreTold) {
            ran(evaluation.trait(), evaluation.runParams());
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
                hold(firing);
                return;
            }
        }
    }

    /** Count a run firing in its line's reservoir and hold it there if it drew a slot. */
    private void hold(Firing firing) {
        positives.count(firing.key());
        int slot = firing.slot();
        if (slot < 0) {
            return;
        }
        // A slot past the records held is a free place, whatever an earlier
        // build that threw left unfilled (C2 #13).
        boolean fills = slot >= positives.held(firing.key());
        positives.hold(firing.key(), slot, firing.record());
        if (fills) {
            negatives.keep(firing.mode(), firing.key(), true);
        }
    }

    /** How many evaluated firings wait with a built record; for a test. */
    int pendingRecords() {
        return (int) unconfirmed.stream().filter(firing -> firing.record() != null).count();
    }

    /** How many evaluated firings wait for their run; for a test. */
    int pending() {
        return unconfirmed.size();
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
        return live.traitRecord(new LiveGame.ActingTrait(context, RecordKind.TRIGGER, trait, line), event)
                .payload(new RecordPayload(Json.object()
                        .raw("event", event.toJson())
                        .value("fired", fired)
                        .render(), false, null));
    }
}
