package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventCardStatsChanged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventDayTimeChanged;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventShuffle;
import forge.game.event.GameEventSpeedChanged;
import forge.game.event.GameEventSurveil;
import forge.game.spellability.SpellAbility;
import forge.game.zone.CostPaymentStack;
import forge.game.zone.ZoneType;

/**
 * The outcome events of the public bus, read once for every collector.
 *
 * <p>The bracket collector and the fork sink both describe what happened
 * through {@link BusEvents}, and a forced resolution must be described the
 * way an observed one is: the same outcome events, each with the cause the
 * bus names for a damage, a poison or a radiation counter. Two subscription
 * lists drift apart, so the subscriptions live here, once, and a subclass
 * decides only what it alone can decide: whether a card
 * damage is combat damage, where an event goes, and what a permanent leaving
 * the battlefield means to it.
 *
 * <p>An abstract class rather than a reader handed a sink, because Forge's bus
 * finds {@code @Subscribe} methods on a subscriber's superclasses too: every
 * caller keeps subscribing the collector itself.
 */
abstract class BusEventReader {

    /** The game whose bus this reads, or null for a sink with no game to read. */
    final Game game;

    /**
     * Computed characteristics, so a stats-changed event can say what changed.
     *
     * <p>Lives as long as the reader: a pump and its wearing off are both
     * differences from what was there before.
     */
    private final StatDiffer stats = new StatDiffer();

    BusEventReader(Game game) {
        this.game = game;
    }

    /**
     * Whether a card damaged now is damaged in combat.
     *
     * <p>{@code GameEventCardDamaged} carries no combat flag, unlike its player
     * sibling, so the reader asks: the bracket collector answers from the
     * step, a damage-step probe always yes, an intervention always no.
     */
    abstract boolean cardDamageIsCombat();

    /**
     * File one damage, card or player.
     *
     * @param namedCause the permanent the bus says dealt it, or null
     * @param combat     whether it is combat damage, which routes it
     */
    abstract void fileDamage(EffectEvent damage, String namedCause, boolean combat);

    /**
     * File any other outcome.
     *
     * @param namedCause the ref the bus named as the causer, which outranks
     *                   the bracket's, or null to take the bracket's
     */
    abstract void fileOutcome(EffectEvent event, String namedCause);

    /** File a completed move; leaving the battlefield can mean a death. */
    abstract void fileMove(GameEventCardChangeZone event, EffectEvent moved);

    /**
     * Whether this move is a permanent leaving the battlefield while a
     * state-based check runs: a death some earlier action left behind.
     *
     * <p>The check is read as the game's view tracker being frozen, which
     * {@code GameAction.checkStateEffects} holds for its whole loop. One
     * reading for the bracket collector and the fork sink alike, so an
     * observed death and a forced one are recognised by the same rule.
     */
    final boolean isStateBasedLeave(GameEventCardChangeZone event) {
        return event.from() != null
                && event.from().zoneType() == ZoneType.Battlefield
                && game != null && game.getTracker().isFrozen();
    }

    /**
     * Whether a player is paying, now, for a spell that {@code resolving}
     * casts — the mana and tapped lands of a madness cast, the sacrifice,
     * discard or life of a cascaded spell's additional cost.
     *
     * <p>That payment is the cast spell's cost, which its own cost half
     * records, and not the resolving ability's effect. Read off {@code
     * game.costPaymentStack}, which {@code CostPayment} keeps for every part
     * being paid, so a mana ability activated to pay is recognised by the
     * spell beneath it. Only a spell cast from a {@code Play} effect reaches
     * the stack in the middle of a resolution; the resolving ability's own
     * payments — an optional cost, an unless cost, which Forge pays through
     * the resolving ability or an empty one — are its effect and stay. One
     * reading for the bracket collector and the fork sink alike, so an
     * observed cascade and a forced one leave the same payment out.
     */
    final boolean payingForACast(SpellAbility resolving) {
        if (game == null) {
            return false;
        }
        for (CostPaymentStack.Entry entry : game.costPaymentStack) {
            var paid = entry.payment() == null ? null : entry.payment().getAbility();
            if (paid != null && paid.isSpell() && paid.isCastFromPlayEffect()
                    && paid != resolving) {
                return true;
            }
        }
        return false;
    }

    /**
     * One damage, and the permanent that dealt it.
     *
     * <p>The cause comes off the event rather than off the bracket, because in
     * a damage step there is no bracket: nothing is resolving, and the
     * attacking creature is the only answer to "what caused this". It is also
     * what makes two attackers each dealing 1 damage to the same blocker two
     * distinguishable events instead of one line rendered twice.
     */
    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        boolean combat = cardDamageIsCombat();
        fileDamage(BusEvents.cardDamaged(event, combat), BusEvents.causeOf(event), combat);
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        fileDamage(BusEvents.playerDamaged(event), BusEvents.causeOf(event), event.combat());
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        file(BusEvents.lifeChanged(event), null);
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        file(BusEvents.poisoned(event), BusEvents.causeOf(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        file(BusEvents.counters(event), null);
    }

    /** Energy, and any other player counter the vocabulary learns to name. */
    @Subscribe
    public void onPlayerCounters(GameEventPlayerCounters event) {
        file(BusEvents.playerCounter(event), null);
    }

    @Subscribe
    public void onRadiation(GameEventPlayerRadiation event) {
        file(BusEvents.radiation(event), BusEvents.causeOf(event));
    }

    @Subscribe
    public void onSpeed(GameEventSpeedChanged event) {
        file(BusEvents.speed(event), null);
    }

    @Subscribe
    public void onDayTime(GameEventDayTimeChanged event) {
        file(BusEvents.dayTime(event), null);
    }

    /**
     * A regeneration shield doing its job. No named cause: see {@link
     * BusEvents#regenerated}.
     */
    @Subscribe
    public void onRegenerated(GameEventCardRegenerated event) {
        file(BusEvents.regenerated(event), null);
    }

    /** A library shuffled. No named cause: see {@link BusEvents#libraryShuffled}. */
    @Subscribe
    public void onShuffle(GameEventShuffle event) {
        file(BusEvents.libraryShuffled(event), null);
    }

    /**
     * The game ended. No named cause: see {@link BusEvents#gameOutcome}. Nothing
     * is filed on a draw, or when the winning name resolves to no player in
     * this game.
     */
    @Subscribe
    public void onGameOutcome(GameEventGameOutcome event) {
        if (game != null) {
            file(BusEvents.gameOutcome(event, game), null);
        }
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        file(BusEvents.tapped(event), null);
    }

    /**
     * A card moving, and the draw, discard or mill it may also be.
     *
     * <p>The workhorse outcome, and it is read here rather than off
     * {@code GameEventZone} because only this event is a <b>move</b>: the
     * per-zone-list notification fires once for the zone left, once for the zone
     * reached and once more for the stack, and it cannot say where a card came
     * from at all.
     *
     * <p>Draws, discards and mills have no event of their own: each is a card
     * moving between two zones and which of the three it is depends on the pair.
     */
    @Subscribe
    public void onCardChangeZone(GameEventCardChangeZone event) {
        EffectEvent moved = BusEvents.cardMoved(event);
        if (moved != null) {
            fileMove(event, moved);
        }
        file(BusEvents.libraryMovement(event), null);
        if (event.card() == null) {
            return;
        }
        if (event.from() != null && event.from().zoneType() == ZoneType.Battlefield) {
            // Off the battlefield a card's computed characteristics stop
            // meaning anything, and one that returns is a new object.
            stats.forget(event.card().getId());
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Battlefield) {
            // Forge publishes a stats change after applying it, so without a
            // reading taken now the first pump finds no earlier value and
            // reports nothing. Read off the battlefield object, which the
            // statics and the counters it entered with have already reached.
            stats.prime(cardById(event.card().getId()));
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Stack) {
            onStackArrival(event.card().getId());
        }
    }

    /**
     * A card has arrived in the stack zone. Published after the static check
     * its move set off, so what that check did has already been filed.
     *
     * <p>A no-op here: only the bracket collector follows a cast from the
     * card's arrival to its cast event.
     */
    void onStackArrival(int cardId) {
    }

    /**
     * Take a reading of every card on the battlefield.
     *
     * <p>For a reader that starts listening to a game already in progress, such
     * as a fork: every card on the board is new to it, and its first change
     * would otherwise be its baseline.
     */
    void primeBattlefield() {
        if (game == null) {
            return;
        }
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            stats.prime(card);
        }
    }

    /**
     * The engine says a card's stats moved; this says what moved.
     *
     * <p>The bus publishes one event for a pump, an anthem recompute, an
     * animation and a colour change alike, and it names only the card. Without
     * the difference, the whole "gets +2/+2 and gains flying" family — most of
     * what a limited deck does — resolved into an empty event list.
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
                for (EffectEvent changed : stats.diff(card)) {
                    fileOutcome(changed, null);
                }
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        file(BusEvents.attachment(event), null);
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        for (EffectEvent looked : BusEvents.scry(event)) {
            fileOutcome(looked, null);
        }
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        for (EffectEvent looked : BusEvents.surveil(event)) {
            fileOutcome(looked, null);
        }
    }

    /** File an outcome a {@link BusEvents} factory may have answered null for. */
    private void file(EffectEvent event, String namedCause) {
        if (event != null) {
            fileOutcome(event, namedCause);
        }
    }

    /**
     * The battlefield card an id names, or null.
     *
     * <p>The bus hands out views rather than cards, and a view carries the id
     * and not the object. The battlefield is small enough that a scan costs
     * less than an index kept in step with every zone change.
     */
    private Card cardById(int id) {
        if (game == null) {
            return null;
        }
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (card.getId() == id) {
                return card;
            }
        }
        return null;
    }
}
