package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
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

    private SeveralExplore() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SeveralExplore::bind, BRANCHWALKER_IN_PLACE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), BRANCHWALKER)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.getCardsIn(ZoneType.Library).size() < 2) {
            return Optional.empty();
        }
        for (Card host : active.getCardsIn(ZoneType.Battlefield)) {
            for (Trigger trigger : host.getTriggers()) {
                Map<String, String> explore = Boards.executed(trigger);
                String defined = explore.getOrDefault("Defined", "");
                if (trigger.getMode() != TriggerType.Phase || !ApiType.Explore.name().equals(Boards.api(explore))
                        || !defined.startsWith(VALID)) {
                    continue;
                }
                String[] kind = defined.substring(VALID.length()).split(",");
                List<Card> explorers = CardLists.getValidCards(active.getCreaturesInPlay(), kind, active, host,
                        trigger);
                if (explorers.size() != 1) {
                    continue;
                }
                Optional<Card> standIn = active.getCreaturesInPlay().stream()
                        .filter(creature -> creature != host && !creature.isToken() && !explorers.contains(creature))
                        .filter(creature -> Boards.replacementSurvives(creature, BRANCHWALKER))
                        .findFirst();
                if (standIn.isPresent()) {
                    return Optional.of(Binding.empty().with("creature", standIn.get()));
                }
            }
        }
        return Optional.empty();
    }
}
