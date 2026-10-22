package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.event.GameEventCardChangeZone;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Everything a forked resolution did, in order.
 *
 * <p>The bracket collector sorts events into a resolution bracket or a combat
 * one, because a real game interleaves them. A fork has no such problem: it is
 * created, one ability is forced to resolve or one damage step is dealt, and it
 * is thrown away, so every event the bus publishes in that window belongs to
 * that one thing. Appending them flat is the whole job.
 *
 * <p>The events are read by {@link BusEventReader}, the same reader the bracket
 * collector uses, so an interventional or probe record and an observed one
 * describe the same outcome the same way: the same subscriptions, the same
 * shaping, and the same cause the bus names for a damage, a poison or a
 * radiation counter.
 *
 * <p>Every event is stamped by {@link EventAttribution} on its way in, for the
 * same reason: {@code attributed_to} and {@code duration} are how a reader asks
 * which clause of the line did this and how long it lasts, and a fork that
 * answered null to both was not describing the outcome the same way after all.
 */
final class ForkEventSink extends BusEventReader {

    /**
     * The line whose resolution is being watched, or null when nothing resolves.
     *
     * <p>An interventional fork forces one ability, and that ability is the
     * root the sub-ability pointer is read against. A damage-step probe forces
     * no ability at all: it deals combat damage directly, so there is no root,
     * and its events attribute exactly the way the observed combat record's do
     * — from the pointer alone, which for a turn-based action names nothing.
     */
    private final SpellAbility acting;
    /**
     * Whether a card damaged in this fork is damaged in combat: true for a
     * damage-step probe, which deals one combat damage step, and false for an
     * intervention, whose forced resolution is not one. The flag is what a
     * reader uses to tell the two apart, and a probe's damage is compared
     * against the observed combat record's.
     */
    private final boolean combatDamage;
    private final List<EffectEvent> events = new ArrayList<>();
    /** Positions in {@link #events} of moves off the battlefield during a state-based check. */
    private final Set<Integer> stateBasedLeaves = new LinkedHashSet<>();

    private ForkEventSink(Game fork, SpellAbility acting, boolean combatDamage) {
        super(fork);
        this.acting = acting;
        this.combatDamage = combatDamage;
        primeBattlefield();
    }

    /** A sink for a forced resolution of {@code acting}. */
    static ForkEventSink forIntervention(Game fork, SpellAbility acting) {
        return new ForkEventSink(fork, acting, false);
    }

    /** A sink for a damage step dealt in a fork, with no ability resolving. */
    static ForkEventSink forProbe(Game fork) {
        return new ForkEventSink(fork, null, true);
    }

    List<EffectEvent> events() {
        return List.copyOf(events);
    }

    /**
     * Where in {@link #events()} a permanent left the battlefield while a
     * state-based check was running, read as the fork's view tracker being
     * frozen, the way {@code BusBracketCollector.holdAsConsequence} reads it.
     */
    Set<Integer> stateBasedLeaves() {
        return Set.copyOf(stateBasedLeaves);
    }

    boolean isEmpty() {
        return events.isEmpty();
    }

    @Override
    boolean cardDamageIsCombat() {
        return combatDamage;
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
        events.add(EventAttribution.stamp(event, acting, namedCause));
    }

    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        if (event.from() != null
                && event.from().zoneType() == ZoneType.Battlefield
                && game != null && game.getTracker().isFrozen()) {
            // A state-based check is running -- the same reading the live
            // collector takes -- so this is a death the forced resolution
            // left behind, which a modal fork files by the live rule.
            stateBasedLeaves.add(events.size());
        }
        fileOutcome(moved, null);
    }
}
