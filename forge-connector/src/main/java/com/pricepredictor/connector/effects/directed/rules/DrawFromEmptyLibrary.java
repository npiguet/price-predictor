package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 704.5b, a player who attempted to draw from an empty library since the
 * last state-based check loses the game (121.4 restates it), on scenario
 * {@code 704.5b-2}: A's library holds one card as its upkeep begins, and its
 * draw step draws it.
 *
 * <p>The point is the draw step's turn-based draw (504.1), which no player
 * chooses: a draw spell (scenario {@code 704.5b-1}'s Divination) would wait
 * on the AI casting it, and Forge's AI casts no draw spell drawing at least
 * its library's size less three ({@code DrawAi}, "don't deck yourself").
 *
 * <p>The base board, read at a boundary: the active player's upkeep begins,
 * its draw step is not skipped (the first turn of a two-player game skips
 * it), and its library holds exactly one card, which the draw step draws. It
 * binds that card.
 *
 * <p>Variant {@code empty} ({@code move_card}): the card is in its owner's
 * graveyard (a move is the operation's only way to empty a library), so the
 * draw step's draw is attempted from an empty library and the player loses
 * the game at the next state-based check.
 */
public final class DrawFromEmptyLibrary {

    public static final String RULE = "704.5b";
    static final String SCENARIO = "704.5b-2";
    static final String EMPTY = "empty";

    private static final Role<Card> CARD = Role.card("card");

    private DrawFromEmptyLibrary() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DrawFromEmptyLibrary::bind, new DirectedRule.Variant(EMPTY,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card(CARD), ZoneType.Graveyard, null))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP) || drawStepSkipped(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> library = List.copyOf(active.getCardsIn(ZoneType.Library));
        if (library.size() != 1) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with(CARD, library.get(0)));
    }

    /** Whether this turn's draw step is skipped as Forge skips it: a two-player game's first turn. */
    private static boolean drawStepSkipped(Game live) {
        return live.getPhaseHandler().getTurn() == 1 && live.getPlayers().size() == 2;
    }
}
