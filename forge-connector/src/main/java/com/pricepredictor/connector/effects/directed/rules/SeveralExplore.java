package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 701.44d, permanents instructed to explore at once explore one at a time,
 * inverted (FR-014a) on scenario {@code 701.44d-2}: Hakbal of the Surging
 * Soul's beginning-of-combat trigger has each Merfolk its controller controls
 * explore, Hakbal is the only one, and one card is revealed; Merfolk
 * Branchwalker in the place of A's Grizzly Bears makes two explore, each
 * revealing its own card.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, it controls a permanent whose phase trigger has every creature of a
 * kind it controls explore (Hakbal: each Merfolk), exactly one creature it
 * controls is of that kind, and its library holds at least two cards: the
 * common board, whose trigger reveals one card, the rule's near miss. It
 * binds another creature it controls, not of that kind and not a token, which
 * Merfolk Branchwalker put in its place survives beside.
 *
 * <p>Variant {@code branchwalker_in_the_bears_place} ({@code replace_card}):
 * Merfolk Branchwalker is in that creature's place, so two creatures explore,
 * one at a time, and two cards are revealed: the rule's {@code fires} side,
 * which the counts lack. Nothing moved the Branchwalker there, so its own
 * enters trigger does not explore.
 */
public final class SeveralExplore {

    public static final String RULE = "701.44d";
    static final String SCENARIO = "701.44d-2";
    static final String BRANCHWALKER_IN_PLACE = "branchwalker_in_the_bears_place";
    static final String BRANCHWALKER = "Merfolk Branchwalker";
    private static final String VALID = "Valid ";
    private static final Role<Card> CREATURE = Role.card("creature");

    private SeveralExplore() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SeveralExplore::bind, new DirectedRule.Variant(BRANCHWALKER_IN_PLACE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card(CREATURE), BRANCHWALKER))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.getCardsIn(ZoneType.Library).size() < 2) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(host -> host.getTriggers().stream().map(trigger -> standIn(active, host, trigger)))
                .flatMap(Optional::stream)
                .findFirst()
                .map(standIn -> Binding.empty().with(CREATURE, standIn));
    }

    /**
     * The first creature besides {@code host}, no token, that Merfolk
     * Branchwalker can replace, when the trigger explores for exactly one
     * creature of {@code active}'s; empty for any other trigger.
     */
    private static Optional<Card> standIn(Player active, Card host, Trigger trigger) {
        Map<String, String> explore = Scripts.executed(trigger);
        String defined = explore.getOrDefault("Defined", "");
        if (!exploresForCreatures(trigger, explore, defined)) {
            return Optional.empty();
        }
        String[] kind = defined.substring(VALID.length()).split(",");
        List<Card> explorers = CardLists.getValidCards(active.getCreaturesInPlay(), kind, active, host, trigger);
        if (explorers.size() != 1) {
            return Optional.empty();
        }
        return active.getCreaturesInPlay().stream()
                .filter(creature -> creature != host && !creature.isToken() && !explorers.contains(creature))
                .filter(creature -> Boards.replacementSurvives(creature, BRANCHWALKER))
                .findFirst();
    }

    private static boolean exploresForCreatures(Trigger trigger, Map<String, String> explore, String defined) {
        return trigger.getMode() == TriggerType.Phase && ApiType.Explore.name().equals(Scripts.api(explore))
                && defined.startsWith(VALID);
    }
}
