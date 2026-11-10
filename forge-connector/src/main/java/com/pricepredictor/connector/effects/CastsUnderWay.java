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
 * before the cast is pushed, the one {@code MagicStack.push} runs after —
 * reaches no record: the cost half records the card's move and the costs
 * alone (FR-042a), and the effect half's board, read when the resolution
 * starts, already shows it (FR-042b). The collector counts it as dropped.
 *
 * <p>Where each card came from is kept from its arrival, for the cost half's
 * {@code zone_change} to the stack ({@link #arrivedFrom}).
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
     * The zone each card in the stack zone arrived from, by card id, as the
     * bus published its move: what a spell's cost half says it was cast from.
     */
    private final java.util.Map<Integer, String> arrivedFrom = new java.util.HashMap<>();

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
    void onStackArrival(int cardId, String fromZone) {
        abandonAllBut(cardId);
        // A card keeps its id across zones: a mark from an earlier attempt
        // is not this one's.
        abandoned.remove(cardId);
        arrived.add(cardId);
        arrivedFrom.put(cardId, fromZone);
    }

    /**
     * The zone the card with this id arrived in the stack zone from, as its
     * move was published, or null when no arrival of it was seen.
     */
    String arrivedFrom(int cardId) {
        return arrivedFrom.get(cardId);
    }

    /**
     * Bring the marks up to date with the stack zone and the action in
     * progress, before {@link #cardBeingCast} is asked.
     *
     * <p>A card that has left the stack zone loses its marks ({@link
     * #forgetCardsGoneFrom}), and an arrived card the action in progress
     * passes over is given up, since the action names one cast at a time and
     * the one it names is not that card's. Brought up to date again with
     * nothing changed, the marks stay as they are.
     *
     * @param acting the action in progress, or null
     */
    void keepMarksCurrent(SpellAbility acting) {
        List<Card> zone = stackZone();
        forgetCardsGoneFrom(zone);
        if (pushedUnannounced(acting) != null || onItsWay(zone) != null) {
            return;
        }
        // Arrived cards the action passes over, before the one it names.
        for (Card card : arrivedUnheld(zone)) {
            if (acting == null || isCastOf(acting, card)) {
                return;
            }
            abandoned.add(card.getId());
        }
    }

    /**
     * The card of the cast under way, or null when none is.
     *
     * <p>A pure read of marks {@link #keepMarksCurrent} has brought up to date
     * for the same action.
     *
     * @param acting the action in progress, or null
     */
    Card cardBeingCast(SpellAbility acting) {
        Card pushed = pushedUnannounced(acting);
        if (pushed != null) {
            return pushed;
        }
        List<Card> zone = stackZone();
        Card onItsWay = onItsWay(zone);
        if (onItsWay != null) {
            return onItsWay;
        }
        // Arrived: under way until its targets are chosen and the action
        // names it, and for as long as it does. Any other action means it
        // was given up.
        for (Card card : arrivedUnheld(zone)) {
            if (acting == null || isCastOf(acting, card)) {
                return card;
            }
        }
        return null;
    }

    private List<Card> stackZone() {
        return new ArrayList<>(game.getCardsIn(ZoneType.Stack));
    }

    /**
     * The card of a cast pushed and not yet announced, while it is the action
     * in progress — {@code MagicStack.push}'s own static check — or null.
     */
    private Card pushedUnannounced(SpellAbility acting) {
        SpellAbility pushed = StackActions.topAbility(game.getStack());
        if (acting != null && pushed != null && pushed.getHostCard() != null
                && !announced.test(pushed.getId())
                && (pushed == acting || pushed.getOriginalAbility() == acting)) {
            return pushed.getHostCard();
        }
        return null;
    }

    /** The card on its way, its arrival not yet published — the newest move — or null. */
    private Card onItsWay(List<Card> zone) {
        Set<Integer> held = heldHostIds();
        for (Card card : zone) {
            int id = card.getId();
            if (!held.contains(id) && !abandoned.contains(id) && !arrived.contains(id)) {
                return card;
            }
        }
        return null;
    }

    /** The cards in the zone no stack item holds and no mark gives up, in zone order. */
    private List<Card> arrivedUnheld(List<Card> zone) {
        Set<Integer> held = heldHostIds();
        List<Card> cards = new ArrayList<>();
        for (Card card : zone) {
            int id = card.getId();
            if (!held.contains(id) && !abandoned.contains(id)) {
                cards.add(card);
            }
        }
        return cards;
    }

    /** A card that has left the stack zone keeps no mark: back in it, it is a new attempt. */
    private void forgetCardsGoneFrom(List<Card> zone) {
        Set<Integer> inZone = new HashSet<>();
        zone.forEach(card -> inZone.add(card.getId()));
        arrived.retainAll(inZone);
        abandoned.retainAll(inZone);
        arrivedFrom.keySet().retainAll(inZone);
    }

    /** Whether the action in progress is casting {@code card}. */
    private static boolean isCastOf(SpellAbility acting, Card card) {
        return acting.getHostCard() != null && acting.getHostCard().getId() == card.getId();
    }

    /** Every card in the stack zone that no stack item holds is a cast given up. */
    void abandonAll() {
        abandonAllBut(-1);
    }

    /** The same, but for the card {@code keep}, whose cast is the one now under way. */
    private void abandonAllBut(int keep) {
        Set<Integer> held = heldHostIds();
        for (Card card : game.getCardsIn(ZoneType.Stack)) {
            if (!held.contains(card.getId()) && card.getId() != keep) {
                abandoned.add(card.getId());
            }
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
}
