package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 702.187b, a card with mayhem can be cast from its owner's graveyard for its
 * mayhem cost only if it was discarded this turn, on scenario
 * {@code 702.187b-1}: A discards Electro's Bolt to Wild Mongrel, then casts it
 * from the graveyard for {1}{R} with two Mountains.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, its graveyard holds a card with a mayhem cost that it discarded this
 * turn, and the permanents it can make mana with are all lands, exactly as
 * many as that cost's mana value. It binds one of those lands.
 *
 * <p>Variant {@code one_mountain} ({@code remove_permanent}): that land is
 * gone, so the mayhem cost is one mana short and the card cannot be cast.
 */
public final class Mayhem {

    public static final String RULE = "702.187b";
    static final String SCENARIO = "702.187b-1";
    static final String ONE_MOUNTAIN = "one_mountain";

    private Mayhem() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Mayhem::bind, ONE_MOUNTAIN,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("land"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card card : active.getCardsIn(ZoneType.Graveyard)) {
            if (!card.wasDiscarded() || !card.enteredThisTurn()) {
                continue;
            }
            for (Cost cost : Boards.keywordCosts(card, Keyword.MAYHEM)) {
                Optional<Card> land = Boards.lastLandFor(active, Boards.manaValue(cost));
                if (land.isPresent()) {
                    return Optional.of(Binding.empty().with("land", land.get()));
                }
            }
        }
        return Optional.empty();
    }
}
