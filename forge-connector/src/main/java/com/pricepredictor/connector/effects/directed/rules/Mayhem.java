package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
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
 * many as that cost's mana value. It binds one of those lands and the card.
 *
 * <p>The active player cannot add a land to its mana this turn
 * ({@link Mana#noLandToPlay}): it holds no land, or its land drop is
 * used, so the turn's land drop cannot undo the mana the variant takes.
 *
 * <p>Variant {@code one_mountain} ({@code remove_permanent}): that land is
 * gone, so the mayhem cost is one mana short and the card cannot be cast.
 *
 * <p>The fork keeps the card's decision candidates whatever the decision
 * sample draws (FR-010c): the collector reads their verdicts at each of the
 * fork's priority stops itself, whether or not a seat plays them.
 */
public final class Mayhem {

    public static final String RULE = "702.187b";
    static final String SCENARIO = "702.187b-1";
    static final String ONE_MOUNTAIN = "one_mountain";
    private static final Role<Card> LAND = Role.card("land");
    private static final Role<Card> CARD = Role.card("card");

    private Mayhem() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Mayhem::bind, new DirectedRule.Variant(ONE_MOUNTAIN,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card(LAND)))
                        .keepingCandidatesOf(CARD)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!Mana.noLandToPlay(active)) {
            return Optional.empty();
        }
        for (Card card : active.getCardsIn(ZoneType.Graveyard)) {
            Optional<Card> land = card.wasDiscarded() && card.enteredThisTurn()
                    ? lastLandForMayhem(active, card) : Optional.empty();
            if (land.isPresent()) {
                return Optional.of(Binding.empty().with(LAND, land.get()).with(CARD, card));
            }
        }
        return Optional.empty();
    }

    /** The land one mana short of paying one of the card's mayhem costs, the first cost's that has one. */
    private static Optional<Card> lastLandForMayhem(Player active, Card card) {
        return Mana.keywordCosts(card, Keyword.MAYHEM).stream()
                .map(cost -> Mana.lastLandFor(active, Mana.manaValue(cost)))
                .flatMap(Optional::stream)
                .findFirst();
    }
}
