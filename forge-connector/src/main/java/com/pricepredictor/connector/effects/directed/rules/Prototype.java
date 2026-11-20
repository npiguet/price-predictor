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
 * 718.3a, a prototyped spell is judged for casting by its prototype mana cost
 * alone, on scenario {@code 718.3a-1}: A can cast Blitz Automaton prototyped
 * for {2}{R} with three Mountains, though not for its own {7}.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with prototype whose own mana value is more than
 * the mana it can make, and the permanents it can make mana with are all
 * lands, exactly as many as the prototype cost's mana value. It binds one of
 * those lands and the card.
 *
 * <p>The active player cannot add a land to its mana this turn
 * ({@link Mana#noLandToPlay}): it holds no land, or its land drop is
 * used, so the turn's land drop cannot undo the mana the variant takes.
 *
 * <p>Variant {@code two_mountains} ({@code remove_permanent}): that land is
 * gone, so the prototype cost is one mana short and neither cast is legal.
 *
 * <p>The fork keeps the card's decision candidates whatever the decision
 * sample draws (FR-010c): the collector reads their verdicts at each of the
 * fork's priority stops itself, whether or not a seat plays them.
 */
public final class Prototype {

    public static final String RULE = "718.3a";
    static final String SCENARIO = "718.3a-1";
    static final String TWO_MOUNTAINS = "two_mountains";

    private Prototype() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Prototype::bind, TWO_MOUNTAINS,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("land")))
                        .keepingCandidatesOf("card"));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!Mana.noLandToPlay(active)) {
            return Optional.empty();
        }
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            for (Cost cost : Mana.keywordCosts(card, Keyword.PROTOTYPE)) {
                int value = Mana.manaValue(cost);
                Optional<Card> land = Mana.lastLandFor(active, value);
                if (land.isPresent() && card.getCMC() > value) {
                    return Optional.of(Binding.empty().with("land", land.get()).with("card", card));
                }
            }
        }
        return Optional.empty();
    }
}
