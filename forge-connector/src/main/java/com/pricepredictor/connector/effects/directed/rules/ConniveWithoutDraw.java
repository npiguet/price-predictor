package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.TriggerType;

import java.util.Optional;

/**
 * 701.50f, a permanent connives even when its controller cannot draw or
 * discard, and connive triggers fire, inverted (FR-014a) on scenario
 * {@code 701.50f-2}: A's Echo Inspector connives, drawing and discarding,
 * and Glorious Purpose triggers; under B's Spirit of the Labyrinth, A having
 * drawn its card this turn, the Inspector connives without drawing, and
 * Glorious Purpose still triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the lands for that connives (Echo
 * Inspector), a permanent on the battlefield triggers when a creature
 * connives (Glorious Purpose), and it has drawn a card this turn and can
 * draw another: the common board, whose connive draws. It binds the active
 * player, the conniving card, whose cast the fork forces (FR-010b), and an
 * opponent.
 *
 * <p>Variant {@code spirit_added} ({@code add_permanent}): the opponent
 * controls Spirit of the Labyrinth, since before this turn, so the player,
 * who has drawn this turn, cannot draw, and the connive draws nothing, the
 * rule's point; the connives trigger fires as before.
 */
public final class ConniveWithoutDraw {

    public static final String RULE = "701.50f";
    static final String SCENARIO = "701.50f-2";
    static final String SPIRIT_ADDED = "spirit_added";
    static final String SPIRIT = "Spirit of the Labyrinth";

    /**
     * A spell of the card that connives as it resolves or as its permanent
     * enters: what the precondition finds and the fork casts.
     */
    static final ForcedPlay.Line CONNIVING = Scan.acting("Connive");

    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> CONNIVER = Role.card("conniver");
    private static final Role<Player> OPPONENT = Role.player("opponent");

    private ConniveWithoutDraw() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ConniveWithoutDraw::bind, SPIRIT_ADDED,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.tapped(SPIRIT, fork.player(OPPONENT)))
                        .forcing(ForcedPlay.cast(PLAYER, CONNIVER).playing(CONNIVING)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> conniver = Scan.acting(active, CONNIVING)
                .filter(card -> Boards.watched(live, TriggerType.Connives, active, card));
        if (conniver.isEmpty() || !spiritWouldStopTheDraw(active)) {
            return Optional.empty();
        }
        return active.getOpponents().stream().findFirst()
                .map(opponent -> Binding.empty().with(PLAYER, active).with(CONNIVER, conniver.get())
                        .with(OPPONENT, opponent));
    }

    /**
     * Whether the Spirit of the Labyrinth added would stop the player's next
     * draw and nothing stops it now: it has drawn a card this turn and can
     * draw another.
     */
    private static boolean spiritWouldStopTheDraw(Player player) {
        return player.getNumDrawnThisTurn() > 0 && player.canDraw();
    }
}
