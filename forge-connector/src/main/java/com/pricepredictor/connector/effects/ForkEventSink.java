package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardStatsChanged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSurveil;
import forge.game.zone.ZoneType;

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
 * same outcome the same way. <b>That claim is only true if this subscribes to
 * the same events</b>, and for the first collected corpus it did not: the sink
 * listened to seven of the sixteen the bracket collector listens to, so a fork
 * of "draw a card" or "gets +2/+2 and gains flying" recorded an empty event
 * list and the counterfactual read as "the ability did nothing". The whole fork
 * vocabulary measured across that corpus was {@code zone_change} and
 * {@code counter_change}, against seventeen types on real resolutions.
 *
 * <p>Its own {@link StatDiffer}, primed by nothing: a fork lives for one
 * resolution, so the first computed characteristics it sees are the baseline
 * the resolution moved away from.
 */
final class ForkEventSink {

    private final Game fork;
    private final StatDiffer stats = new StatDiffer();
    private final List<EffectEvent> events = new ArrayList<>();

    ForkEventSink(Game fork) {
        this.fork = fork;
    }

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

    /**
     * A completed move, and the draw, discard or mill it may also be.
     *
     * <p>One event per move, from the event that carries both ends of it. The
     * per-zone-list notification the sink used to read fires three times for one
     * cast and names the wrong destination on the first, which is why the fork
     * event lists were both duplicated and wrong.
     */
    @Subscribe
    public void onCardChangeZone(GameEventCardChangeZone event) {
        EffectEvent moved = BusEvents.cardMoved(event);
        if (moved != null) {
            events.add(moved);
        }
        EffectEvent named = BusEvents.libraryMovement(event);
        if (named != null) {
            events.add(named);
        }
        if (event.card() != null && event.from() != null
                && event.from().zoneType() == ZoneType.Battlefield) {
            // Off the battlefield a card's computed characteristics stop
            // meaning anything, and one that returns is a new object.
            stats.forget(event.card().getId());
        }
    }

    /**
     * The engine says a card's stats moved; this says what moved.
     *
     * <p>The bus publishes one event for a pump, an anthem recompute, an
     * animation and a colour change alike, and names only the card — so without
     * the difference the whole "gets +2/+2 and gains flying" family, which is
     * most of what a limited deck does, forks into an empty event list.
     */
    @Subscribe
    public void onStatsChanged(GameEventCardStatsChanged event) {
        if (event.cards() == null) {
            return;
        }
        for (CardView view : event.cards()) {
            if (view == null) {
                continue;
            }
            Card card = cardById(view.getId());
            if (card != null) {
                events.addAll(stats.diff(card));
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        EffectEvent attached = BusEvents.attachment(event);
        if (attached != null) {
            events.add(attached);
        }
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        events.addAll(BusEvents.scry(event));
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        events.addAll(BusEvents.surveil(event));
    }

    /**
     * The fork's battlefield card an id names, or null.
     *
     * <p>The bus hands out views rather than cards, and a view carries the id
     * and not the object. The battlefield is small enough that a scan costs less
     * than an index kept in step with every zone change — and a fork's is
     * shorter still, because it lives for one resolution.
     */
    private Card cardById(int id) {
        if (fork == null) {
            return null;
        }
        for (Card card : fork.getCardsIn(ZoneType.Battlefield)) {
            if (card.getId() == id) {
                return card;
            }
        }
        return null;
    }
}
