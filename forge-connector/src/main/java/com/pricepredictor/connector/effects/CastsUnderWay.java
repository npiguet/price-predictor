package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.IntPredicate;

/**
 * The live collector's view of a cast on its way onto the stack: from the
 * moment its card is in the stack zone to its cast event.
 *
 * <p>A cast is under way before the card's arrival is published; after it,
 * while no action names anything else — the play path chooses targets before
 * {@code freezeStack} names the cast — and while the action in progress is
 * that card's cast; and from the push to the cast event, while the stack's
 * top is the action in progress and not yet announced. What the static
 * checks in that window set off — the one {@code GameAction.changeZone} runs
 * before the cast is pushed, the one {@code MagicStack.push} runs after — is
 * the cast's own consequence, held here until its cast event opens its
 * bracket.
 *
 * <p>A card in the stack zone that no stack item holds is otherwise a cast
 * given up — {@code ComputerUtil.handlePlayingSpellAbility} returns on a
 * failed charm choice, target check or payment without moving it back — and
 * it stops counting until it leaves the zone, so that what follows routes as
 * if it were not there. Given up is read at the first sign the attempt is
 * over: another action in progress, another card's arrival, a cast event, a
 * priority, a land played, a step boundary.
 */
final class CastsUnderWay {

    private final Game game;
    /** Whether a cast of this ability id has been announced: its cast event seen. */
    private final IntPredicate announced;

    /** Cards whose arrival in the stack zone has been published, while they are there. */
    private final Set<Integer> arrived = new LinkedHashSet<>();

    /** Cards in the stack zone whose cast was given up, while they are there. */
    private final Set<Integer> abandoned = new LinkedHashSet<>();

    /**
     * What a card on its way onto the stack set off before its cast was
     * announced, or null when nothing is held.
     */
    private PreCast preCast;

    /**
     * @param announced whether the live collector holds a cast of an ability
     *                  id, which it does from its cast event on: a pushed
     *                  item it holds is no longer on its way
     */
    CastsUnderWay(Game game, IntPredicate announced) {
        this.game = game;
        this.announced = announced;
    }

    /**
     * A card has arrived in the stack zone. One cast is under way at a time:
     * a new card on its way means any other still in the zone with no stack
     * item was given up.
     */
    void onStackArrival(int cardId) {
        abandonAllBut(cardId);
        // A card keeps its id across zones: a mark from an earlier attempt
        // is not this one's.
        abandoned.remove(cardId);
        arrived.add(cardId);
    }

    /**
     * The card of the cast under way, or null when none is.
     *
     * <p>Reading it keeps the marks current, and is meant to: a card that has
     * left the stack zone loses its marks ({@link #forgetCardsGoneFrom}), and
     * an arrived card the action in progress passes over is given up, since
     * the action names one cast at a time and the one it names is not that
     * card's. Asked again with nothing changed, it answers the same.
     *
     * @param acting the action in progress, or null
     */
    Card cardBeingCast(SpellAbility acting) {
        List<Card> zone = new ArrayList<>(game.getCardsIn(ZoneType.Stack));
        forgetCardsGoneFrom(zone);
        // Pushed and not yet announced: MagicStack.push's own static check.
        var top = game.getStack().peek();
        SpellAbility pushed = top == null ? null : top.getSpellAbility();
        if (acting != null && pushed != null && pushed.getHostCard() != null
                && !announced.test(pushed.getId())
                && (pushed == acting || pushed.getOriginalAbility() == acting)) {
            return pushed.getHostCard();
        }
        Set<Integer> held = heldHostIds();
        // On its way, its arrival not yet published: the newest move.
        for (Card card : zone) {
            int id = card.getId();
            if (!held.contains(id) && !abandoned.contains(id) && !arrived.contains(id)) {
                return card;
            }
        }
        // Arrived: under way until its targets are chosen and the action
        // names it, and for as long as it does. Any other action means it
        // was given up.
        for (Card card : zone) {
            int id = card.getId();
            if (held.contains(id) || abandoned.contains(id)) {
                continue;
            }
            if (acting == null || isCastOf(acting, card)) {
                return card;
            }
            abandoned.add(id);
        }
        return null;
    }

    /** A card that has left the stack zone keeps no mark: back in it, it is a new attempt. */
    private void forgetCardsGoneFrom(List<Card> zone) {
        Set<Integer> inZone = new HashSet<>();
        zone.forEach(card -> inZone.add(card.getId()));
        arrived.retainAll(inZone);
        abandoned.retainAll(inZone);
    }

    /** Whether the action in progress is casting {@code card}. */
    private static boolean isCastOf(SpellAbility acting, Card card) {
        return acting.getHostCard() != null && acting.getHostCard().getId() == card.getId();
    }

    /** Every card in the stack zone that no stack item holds is a cast given up. */
    void abandonAll() {
        abandonAllBut(-1);
    }

    /**
     * The same, but for the card {@code keep}, whose cast is the one now
     * under way, and what its window has held so far.
     */
    private void abandonAllBut(int keep) {
        Set<Integer> held = heldHostIds();
        for (Card card : game.getCardsIn(ZoneType.Stack)) {
            if (!held.contains(card.getId()) && card.getId() != keep) {
                abandoned.add(card.getId());
            }
        }
        if (preCast != null && !preCast.isFor(keep)) {
            preCast = null;
        }
    }

    /**
     * The ids of the cards the stack's items are on. A card in the stack zone
     * that none of them is on has not been pushed: it is on its way, or was
     * given up.
     */
    private Set<Integer> heldHostIds() {
        Set<Integer> held = new HashSet<>();
        for (var instance : game.getStack()) {
            SpellAbility sa = instance.getSpellAbility();
            if (sa != null && sa.getHostCard() != null) {
                held.add(sa.getHostCard().getId());
            }
        }
        return held;
    }

    /**
     * Hold an event the cast of {@code card} set off, for that cast's
     * bracket. What was held for another card is dropped: one cast is under
     * way at a time.
     */
    void hold(Card card, EffectEvent event, String namedCause) {
        if (preCast == null || !preCast.isFor(card)) {
            preCast = new PreCast(card.getId());
        }
        preCast.hold(event, namedCause);
    }

    /**
     * The cast of {@code cast} has been announced: file what its card's move
     * onto the stack set off into its bracket, stamped against its ability,
     * if anything was held for that card.
     */
    void fileHeldInto(SpellAbility cast, EventBuffer bracket) {
        if (preCast != null && preCast.isFor(cast.getHostCard())) {
            preCast.fileInto(cast, bracket);
        }
    }

    /**
     * The cast under way has been announced or given up: what its window
     * held is taken or dropped.
     */
    void dropHeld() {
        preCast = null;
    }

    /**
     * What a card on its way onto the stack set off before its cast was
     * announced, held for that cast's bracket.
     *
     * <p>Held raw and stamped when the cast opens, against the cast's own
     * ability: when they were published there was no bracket to stamp them
     * against. Dropped if the cast is abandoned and no cast event follows.
     */
    private static final class PreCast {
        private record Held(EffectEvent event, String namedCause) {
        }

        private final int cardId;
        private final List<Held> events = new ArrayList<>();

        private PreCast(int cardId) {
            this.cardId = cardId;
        }

        boolean isFor(Card card) {
            return card != null && isFor(card.getId());
        }

        boolean isFor(int id) {
            return id == cardId;
        }

        void hold(EffectEvent event, String namedCause) {
            events.add(new Held(event, namedCause));
        }

        void fileInto(SpellAbility cast, EventBuffer bracket) {
            for (Held held : events) {
                bracket.file(EventAttribution.stampAt(held.event(), cast, null, held.namedCause()));
            }
        }
    }
}
