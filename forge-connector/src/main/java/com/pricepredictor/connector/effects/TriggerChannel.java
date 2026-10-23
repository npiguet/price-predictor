package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;


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
            // real host; anything this cannot identify is kept, as only a
            // positively identified other game is dropped.
            CardTraitBase trait = args.get(0, CardTraitBase.class);
            if (trait != null && !live.owns(trait)) {
                return;
            }
            boolean fired = Boolean.TRUE.equals(args.get(2));
            String mode = RunParamNormalizer.modeOf(args.get(0));
            if (mode == null) {
                mode = RunParamNormalizer.UNKNOWN_MODE;
            }
            if (!negatives.offer(mode, fired)) {
                return;
            }
            EffectRecord record = record(args.get(0), args.get(1), fired);
            if (!dedup.allowDistinct(
                    RecordKind.TRIGGER.wireName(), record.payloadJson(), record.stateJson())
                    || !negatives.allowLine(record.abilityJson(), fired)) {
                return;
            }
            // Counted only for a record that is actually written, so a
            // suppressed duplicate does not spend the negatives' quota and the
            // ratio holds over what reaches the shard.
            negatives.keep(mode, fired);
            context.emit(record);
        });
    }

    /**
     * One evaluated condition's record, built from the trigger itself.
     *
     * <p>Built after the sampling gate, so a non-fired evaluation that will not
     * be written never pays for resolving its key — which matters, because this
     * is the hook that fires most often in the whole collector.
     */
    EffectRecord record(Object trait, Object runParams, boolean fired) {
        EffectEvent event = RunParamNormalizer.describeParams(trait, runParams);
        return live.traitRecord(context, RecordKind.TRIGGER, trait, event)
                .payload(Json.object()
                        .raw("event", event.toJson())
                        .value("fired", fired)
                        .render());
    }
}
