package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.71a, transfigure sacrifices the creature and puts a creature card of the
 * same mana value from the library onto the battlefield, on scenario
 * {@code 702.71a-1}: A transfigures Fleshwrither (mana value 4) and Hill Giant
 * enters.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls a creature with transfigure whose transfigure cost it
 * has the lands for, and its library holds exactly one creature card of that
 * creature's mana value. It binds the creature and that library card.
 *
 * <p>Variant {@code no_matching_value} ({@code move_card}): a Craw Wurm (mana
 * value 6; a Hill Giant, mana value 4, when the creature's is 6) is in the
 * library card's place, so the search finds nothing to put onto the
 * battlefield.
 */
public final class Transfigure {

    public static final String RULE = "702.71a";
    static final String SCENARIO = "702.71a-1";
    static final String NO_MATCHING_VALUE = "no_matching_value";
    static final String WURM = "Craw Wurm";
    static final String GIANT = "Hill Giant";
    private static final int WURM_VALUE = 6;

    private Transfigure() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Transfigure::bind, NO_MATCHING_VALUE,
                new Application(ForkOperation.MOVE_CARD, fork -> MoveCardOperation.replace(fork.card("card"),
                        fork.card("creature").getCMC() == WURM_VALUE ? GIANT : WURM)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card creature : active.getCreaturesInPlay()) {
            if (!Boards.canPayKeyword(active, creature, Keyword.TRANSFIGURE)) {
                continue;
            }
            List<Card> matching = active.getCardsIn(ZoneType.Library).stream()
                    .filter(card -> card.isCreature() && card.getCMC() == creature.getCMC())
                    .toList();
            if (matching.size() == 1) {
                return Optional.of(Binding.empty().with("creature", creature).with("card", matching.get(0)));
            }
        }
        return Optional.empty();
    }
}
