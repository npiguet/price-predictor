package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 702.127a, an aftermath half can be cast only from a graveyard, on scenario
 * {@code 702.127a-1}: Cut // Ribbons lies in A's graveyard in its main phase,
 * and casting Ribbons from there is legal.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins and its graveyard holds a split card whose second half has
 * aftermath, a candidate it may cast from there. It binds that card.
 *
 * <p>Variant {@code from_hand} ({@code move_card}): the card is in the active
 * player's hand instead, where the aftermath half cannot be cast.
 */
public final class Aftermath {

    public static final String RULE = "702.127a";
    static final String SCENARIO = "702.127a-1";
    static final String FROM_HAND = "from_hand";

    private Aftermath() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Aftermath::bind, FROM_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Hand, null)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        return Boards.active(live).getCardsIn(ZoneType.Graveyard).stream()
                .filter(Aftermath::hasAftermathHalf)
                .findFirst()
                .map(card -> Binding.empty().with("card", card));
    }

    static boolean hasAftermathHalf(Card card) {
        return card.hasState(CardStateName.RightSplit)
                && card.getState(CardStateName.RightSplit).hasKeyword(Keyword.AFTERMATH);
    }
}
