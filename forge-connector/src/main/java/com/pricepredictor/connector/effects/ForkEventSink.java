package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventZone;

import java.util.ArrayList;
import java.util.List;

/**
 * Everything a forked resolution did, in order.
 *
 * <p>The bracket collector sorts events into a resolution bracket or a combat
 * one, because a real game interleaves them. A fork has no such problem: it is
 * created, one ability is forced to resolve, and it is thrown away, so every
 * event the bus publishes in that window belongs to that resolution. Appending
 * them flat is the whole job.
 *
 * <p>Shapes each event through {@link BusEvents}, the same reading the bracket
 * collector uses, so an interventional record and an observed one describe the
 * same outcome the same way.
 */
final class ForkEventSink {

    private final List<EffectEvent> events = new ArrayList<>();

    List<EffectEvent> events() {
        return List.copyOf(events);
    }

    boolean isEmpty() {
        return events.isEmpty();
    }

    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        // Not combat: a forced resolution is not a combat damage step, and the
        // flag is what a reader uses to tell the two apart.
        events.add(BusEvents.cardDamaged(event, false));
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        events.add(BusEvents.playerDamaged(event));
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        events.add(BusEvents.lifeChanged(event));
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        events.add(BusEvents.poisoned(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        events.add(BusEvents.counters(event));
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        events.add(BusEvents.tapped(event));
    }

    @Subscribe
    public void onZone(GameEventZone event) {
        EffectEvent moved = BusEvents.zone(event);
        if (moved != null) {
            events.add(moved);
        }
    }
}
