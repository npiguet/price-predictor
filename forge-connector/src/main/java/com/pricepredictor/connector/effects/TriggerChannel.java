package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;

/**
 * One {@code trigger} record per evaluated trigger condition, fired or not.
 *
 * <p>The condition hook is the busiest in the collector, and most of what it
 * reports is a condition that did not fire; {@link TriggerNegativeSampler}
 * decides which of those are worth a record.
 */
final class TriggerChannel {

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    private final TriggerNegativeSampler negatives;

    TriggerChannel(
            LiveGame live, CollectorContext context, RecordDedup dedup,
            TriggerNegativeSampler negatives) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
        this.negatives = negatives;
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
            // The line's flood cap is asked before the snapshot is built, so
            // a line at its cap costs a key lookup rather than a whole board.
            ProvenanceKey.Resolved line = LiveGame.lineOf(trait);
            if (!negatives.lineHasRoom(line.key(), fired)) {
                return;
            }
            if (!context.emitDistinct(record(trait, line, args.get(1), fired), dedup)) {
                return;
            }
            // Counted only for a record that is actually written, so a
            // suppressed duplicate spends neither the negatives' quota nor
            // the line's cap, and the ratio holds over what reaches the shard.
            negatives.keep(mode, line.key(), fired);
        });
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
