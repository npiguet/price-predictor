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
 * 701.57b, a player has discovered even when no card was found, and discover
 * triggers fire, inverted (FR-014a) on scenario {@code 701.57b-2}: A's
 * Geological Appraiser discovers 3 and finds Grizzly Bears, and Val, Marooned
 * Surveyor triggers; with A's library empty nothing is found, and Val still
 * triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the lands for that discovers (Geological
 * Appraiser), a permanent on the battlefield triggers when a player
 * discovers (Val), and its library holds a card: the common board, whose
 * discovery exiles what it holds until it finds a card or runs out. It binds
 * the active player and the discovering card, whose cast the fork forces
 * (FR-010b).
 *
 * <p>Variant {@code empty_library} ({@code move_card}): the library is
 * empty, so the discovery finds nothing, the rule's point; the discover
 * trigger fires as before.
 */
public final class DiscoverNothingFound {

    public static final String RULE = "701.57b";
    static final String SCENARIO = "701.57b-2";
    static final String EMPTY_LIBRARY = "empty_library";

    /**
     * A spell of the card that discovers as it resolves or as its permanent
     * enters: what the precondition finds and the fork casts.
     */
    static final ForcedPlay.Line DISCOVERING = Boards.acting("Discover");

    private DiscoverNothingFound() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DiscoverNothingFound::bind, EMPTY_LIBRARY,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> new EmptyLibraryOperation(fork.player("player")))
                        .forcing(ForcedPlay.cast("player", "discoverer").playing(DISCOVERING)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> discoverer = Boards.acting(active, DISCOVERING)
                .filter(card -> Boards.watched(live, TriggerType.Discover, active, card));
        if (active.getCardsIn(ZoneType.Library).isEmpty()) {
            return Optional.empty();
        }
        return discoverer.map(card -> Binding.empty().with("player", active).with("discoverer", card));
    }
}
