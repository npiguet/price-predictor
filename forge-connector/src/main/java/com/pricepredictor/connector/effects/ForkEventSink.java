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
import forge.game.event.GameEventDayTimeChanged;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSpeedChanged;
import forge.game.event.GameEventSurveil;
import forge.game.spellability.SpellAbility;
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
 *
 * <p>Every event is stamped by {@link EventAttribution} on its way in, for the
 * same reason: {@code attributed_to} and {@code duration} are how a reader asks
 * which clause of the line did this and how long it lasts, and a fork that
 * answered null to both was not describing the outcome the same way after all.
 * Across the smoke corpus that silence was most of the 30.9% of events with no
 * attribution at all.
 */
final class ForkEventSink {

    private final Game fork;
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
    private final StatDiffer stats = new StatDiffer();
    private final List<EffectEvent> events = new ArrayList<>();

    ForkEventSink(Game fork, SpellAbility acting) {
        this.fork = fork;
        this.acting = acting;
    }

    List<EffectEvent> events() {
        return List.copyOf(events);
    }

    boolean isEmpty() {
        return events.isEmpty();
    }

    /**
     * File one event, attributed.
     *
     * <p>The single door in, so a new subscriber cannot forget the stamp —
     * which is exactly how the fork paths came to write events with a null
     * {@code attributed_to} while the observed path stamped every one of its
     * own. The bracket collector's {@code record} is the sibling of this.
     */
    private void file(EffectEvent event) {
        if (event != null) {
            events.add(EventAttribution.stamp(event, acting));
        }
    }

    /** File a reading that produced several events, all from the same clause. */
    private void fileAll(List<EffectEvent> produced) {
        for (EffectEvent event : produced) {
            file(event);
        }
    }

    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        // Not combat: a forced resolution is not a combat damage step, and the
        // flag is what a reader uses to tell the two apart.
        file(BusEvents.cardDamaged(event, false));
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        file(BusEvents.playerDamaged(event));
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        file(BusEvents.lifeChanged(event));
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        file(BusEvents.poisoned(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        file(BusEvents.counters(event));
    }

    /**
     * Energy, read the same way the bracket collector reads it — see
     * {@code BusBracketCollector.onPlayerCounters}. Kept in step by {@link
     * ForkEventSinkTest#aForkHearsEveryOutcomeAnObservedRecordHears}, so this
     * one cannot drift from that one the way the sink's whole subscription
     * list once did.
     */
    @Subscribe
    public void onPlayerCounters(GameEventPlayerCounters event) {
        file(BusEvents.playerCounter(event));
    }

    @Subscribe
    public void onRadiation(GameEventPlayerRadiation event) {
        file(BusEvents.radiation(event));
    }

    @Subscribe
    public void onSpeed(GameEventSpeedChanged event) {
        file(BusEvents.speed(event));
    }

    @Subscribe
    public void onDayTime(GameEventDayTimeChanged event) {
        file(BusEvents.dayTime(event));
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        file(BusEvents.tapped(event));
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
        file(BusEvents.cardMoved(event));
        file(BusEvents.libraryMovement(event));
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
                fileAll(stats.diff(card));
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        file(BusEvents.attachment(event));
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        fileAll(BusEvents.scry(event));
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        fileAll(BusEvents.surveil(event));
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
