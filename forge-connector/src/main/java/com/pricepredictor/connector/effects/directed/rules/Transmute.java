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
 * 702.53a, transmute discards the card and puts a card of the same mana value
 * from the library into the hand, on scenario {@code 702.53a-1}: A transmutes
 * Dimir Infiltrator (mana value 2) and finds Grizzly Bears.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with transmute whose transmute cost it has the lands
 * for, and its library holds exactly one card of that card's mana value. It
 * binds the transmuting card and that library card.
 *
 * <p>Variant {@code no_match} ({@code move_card}): a Centaur Courser (mana
 * value 3; a Grizzly Bears, mana value 2, when the transmuting card's is 3) is
 * in the library card's place, so the search finds nothing.
 */
public final class Transmute {

    public static final String RULE = "702.53a";
    static final String SCENARIO = "702.53a-1";
    static final String NO_MATCH = "no_match";
    static final String COURSER = "Centaur Courser";
    static final String BEARS = "Grizzly Bears";
    private static final int COURSER_VALUE = 3;

    private Transmute() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Transmute::bind, NO_MATCH,
                new Application(ForkOperation.MOVE_CARD, fork -> MoveCardOperation.replace(fork.card("card"),
                        fork.card("transmuter").getCMC() == COURSER_VALUE ? BEARS : COURSER)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card transmuter : active.getCardsIn(ZoneType.Hand)) {
            if (!Boards.canPayKeyword(active, transmuter, Keyword.TRANSMUTE)) {
                continue;
            }
            List<Card> matching = active.getCardsIn(ZoneType.Library).stream()
                    .filter(card -> card.getCMC() == transmuter.getCMC())
                    .toList();
            if (matching.size() == 1) {
                return Optional.of(Binding.empty().with("transmuter", transmuter).with("card", matching.get(0)));
            }
        }
        return Optional.empty();
    }
}
