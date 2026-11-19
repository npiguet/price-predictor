package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.EmptyLibraryOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 701.44b, a permanent explores even when no card can be revealed, inverted
 * (FR-014a) on scenario {@code 701.44b-2}: A's Merfolk Branchwalker explores
 * and reveals A's top card, and Wildgrowth Walker triggers; with A's library
 * empty nothing is revealed, the Branchwalker still explores and gets its
 * counter, and Wildgrowth Walker still triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the lands for that explores (Merfolk
 * Branchwalker), a permanent on the battlefield triggers when a creature
 * explores (Wildgrowth Walker), and its library holds a card: the common
 * board, whose explore reveals one. It binds the active player and the
 * exploring card, whose cast the fork forces (FR-010b).
 *
 * <p>Variant {@code empty_library} ({@code move_card}): the library is
 * empty, so the explore reveals nothing, the rule's point; the explores
 * trigger fires as before.
 */
public final class ExploreEmptyLibrary {

    public static final String RULE = "701.44b";
    static final String SCENARIO = "701.44b-2";
    static final String EMPTY_LIBRARY = "empty_library";

    /**
     * A spell of the card that explores as it resolves or as its permanent
     * enters: what the precondition finds and the fork casts.
     */
    static final ForcedPlay.Line EXPLORING = Boards.acting("Explore");

    private ExploreEmptyLibrary() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ExploreEmptyLibrary::bind, EMPTY_LIBRARY,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> new EmptyLibraryOperation(fork.player("player")))
                        .forcing(ForcedPlay.cast("player", "card").playing(EXPLORING)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> explorer = Boards.acting(active, EXPLORING)
                .filter(card -> Boards.watched(live, TriggerType.Explores, active, card));
        if (active.getCardsIn(ZoneType.Library).isEmpty()) {
            return Optional.empty();
        }
        return explorer.map(card -> Binding.empty().with("player", active).with("card", card));
    }
}
