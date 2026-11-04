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
 * 701.44d, permanents instructed to explore at once explore one at a time, on
 * scenario {@code 701.44d-1}: Hakbal of the Surging Soul's beginning-of-combat
 * trigger has Hakbal and Merfolk Branchwalker explore, each revealing its own
 * card.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, it controls a permanent whose phase trigger has every creature of a
 * kind it controls explore (Hakbal: each Merfolk), exactly two creatures it
 * controls are of that kind, and its library holds at least two cards. It
 * binds one of the two, the trigger's own permanent only when both are
 * others.
 *
 * <p>Variant {@code only_one_merfolk} ({@code replace_card}): Grizzly Bears is
 * in that creature's place, so only one creature explores and one card is
 * revealed.
 */
public final class SeveralExplore {

    public static final String RULE = "701.44d";
    static final String SCENARIO = "701.44d-1";
    static final String ONLY_ONE = "only_one_merfolk";
    static final String BEARS = "Grizzly Bears";
    private static final String VALID = "Valid ";

    private SeveralExplore() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SeveralExplore::bind, ONLY_ONE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("explorer"), BEARS)));
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
                List<Card> explorers = CardLists.getValidCards(active.getCreaturesInPlay(),
                        defined.substring(VALID.length()).split(","), active, host, trigger);
                if (explorers.size() == 2) {
                    Card explorer = explorers.get(0) == host ? explorers.get(1) : explorers.get(0);
                    return Optional.of(Binding.empty().with("explorer", explorer));
                }
            }
        }
        return Optional.empty();
    }
}
