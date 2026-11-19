package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.event.GameEventCardChangeZone;

import java.util.List;

/**
 * Everything a damage-step probe's fork did, in order.
 *
 * <p>The bracket collector sorts events into a resolution bracket or a combat
 * one, because a real game interleaves them. A probe is simpler: it is
 * created, one damage step is dealt, and it is thrown away, so the damage
 * step is the whole fork and the sink files every event, as the observed
 * combat bracket files its step's. A forced resolution is not recorded here:
 * the live bracket's own code records it ({@link ForcedResolution}).
 *
 * <p>The events are read by {@link BusEventReader}, the same reader the bracket
 * collector uses, and filed into an {@link EventBuffer}, as its brackets are,
 * so a probe record and an observed one describe the same outcome the same
 * way: the same subscriptions, the same shaping, the same cause the bus names
 * for a damage, a poison or a radiation counter, a state transition repeated
 * within the step filed once.
 *
 * <p>Every event is stamped by {@link EventAttribution} on its way in, for the
 * same reason: {@code attributed_to} and {@code duration} are how a reader asks
 * which clause did this and how long it lasts. A probe forces no ability, so
 * there is no root, and its events attribute exactly the way the observed
 * combat record's do — from the pointer alone, which for a turn-based action
 * names nothing.
 */
final class ForkEventSink extends BusEventReader {

    /** What the probe's damage step filed, a repeat recognised within it. */
    private final EventBuffer filed = new EventBuffer();

    /** What the sink heard and no record holds: a change its baseline already showed. */
    private final DroppedEvents dropped = new DroppedEvents();

    private ForkEventSink(Game fork) {
        super(fork);
        String phase = DroppedEvents.phaseOf(fork);
        for (EffectEvent unreported : rebaselineStats()) {
            dropped.count(unreported.type(), phase);
        }
    }

    /** A sink for a damage step dealt in a fork, with no ability resolving. */
    static ForkEventSink forProbe(Game fork) {
        return new ForkEventSink(fork);
    }

    /** The events the sink heard that reached no record. */
    DroppedEvents dropped() {
        return dropped;
    }

    /** Every event filed, in order. */
    List<EffectEvent> events() {
        return List.copyOf(filed.events());
    }

    /** A probe deals one combat damage step, and its damage is combat damage. */
    @Override
    boolean cardDamageIsCombat() {
        return true;
    }

    @Override
    void fileDamage(EffectEvent damage, String namedCause, boolean combat) {
        fileOutcome(damage, namedCause);
    }

    /**
     * File one event, attributed.
     *
     * <p>The single door in, so no event can miss the stamp. The bracket
     * collector's {@code fileOutcome} is the sibling of this.
     */
    @Override
    void fileOutcome(EffectEvent event, String namedCause) {
        filed.file(EventAttribution.stamp(event, null, null, namedCause));
    }

    /**
     * A move, a state-based death included, is an ordinary event of the
     * step, checked for a repeat there: a damage step is one record, as the
     * observed combat bracket is.
     */
    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        fileOutcome(moved, null);
    }
}
