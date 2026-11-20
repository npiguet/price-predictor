package com.pricepredictor.connector.effects.directed.rules;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;

/** What a trigger on a card's script waits for, read where a rule's precondition needs only that. */
final class Triggers {

    private Triggers() {
    }

    /**
     * Whether {@code trigger} fires as its own card enters the battlefield
     * ({@code ValidCard$ Card.Self}, with the conditions a script adds to
     * it, such as {@code +wasCastByYou}).
     */
    static boolean entersItself(Trigger trigger) {
        return trigger.getMode() == TriggerType.ChangesZone
                && "Battlefield".equals(trigger.getParam("Destination"))
                && trigger.getParam("ValidCard") != null
                && trigger.getParam("ValidCard").startsWith("Card.Self");
    }

    /** Whether {@code trigger} fires at the beginning of an upkeep ({@code Mode$ Phase}, {@code Phase$ Upkeep}). */
    static boolean atUpkeep(Trigger trigger) {
        return trigger.getMode() == TriggerType.Phase && "Upkeep".equals(trigger.getParam("Phase"));
    }

    /**
     * Whether a permanent on the battlefield has a trigger of {@code mode}
     * (Explores, Connives, Discover) that {@code active}'s own action sets
     * off: its {@code ValidPlayer} accepts the player, and its
     * {@code ValidCard} accepts {@code actor} or a permanent the player
     * controls, so a watcher of creatures its controller has does not count
     * on the opponent's side.
     */
    static boolean watchedFor(Iterable<Card> battlefield, TriggerType mode, Player active, Card actor) {
        List<Card> candidates = List.copyOf(active.getCardsIn(ZoneType.Battlefield));
        for (Card host : battlefield) {
            for (Trigger trigger : host.getTriggers()) {
                if (trigger.getMode() == mode && accepts(trigger, active, actor, candidates)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean accepts(Trigger trigger, Player active, Card actor, List<Card> candidates) {
        return trigger.matchesValidParam("ValidPlayer", active) && acceptsCard(trigger, actor, candidates);
    }

    private static boolean acceptsCard(Trigger trigger, Card actor, List<Card> candidates) {
        return !trigger.hasParam("ValidCard") || trigger.matchesValidParam("ValidCard", actor)
                || candidates.stream().anyMatch(card -> trigger.matchesValidParam("ValidCard", card));
    }
}
