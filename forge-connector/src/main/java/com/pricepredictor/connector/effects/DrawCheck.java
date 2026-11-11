package com.pricepredictor.connector.effects;

import forge.game.player.Player;

import java.util.ArrayList;
import java.util.List;

/**
 * Whether a card's move from a library to its owner's hand was a draw.
 *
 * <p>Forge publishes no draw event: a draw, a tutor to hand, an explore that
 * reveals a land and a dig to hand all publish the same library-to-hand
 * move. What tells a draw apart is Forge's per-player draw counter, which
 * {@code Player.doDraw} raises right after the move it publishes. So the
 * move is filed as a draw provisionally, and the counter is read once it can
 * have moved: the draw happened when it stands above its value at the move.
 *
 * <p>The counter is read live until the check is settled, and a check is
 * settled at the next move out of the same library (a later draw would raise
 * the counter for itself) and at every step boundary (the cleanup step
 * resets the counter, after its boundary). A record written before either
 * reads the counter live, which is right once the move's own draw, if any,
 * has counted.
 */
final class DrawCheck {

    private final Player player;
    private final int before;
    private Boolean drawn;

    private DrawCheck(Player player) {
        this.player = player;
        this.before = player.getNumDrawnThisTurn();
    }

    /** Whether the move was a draw, as far as the counter says now or said when settled. */
    boolean drawn() {
        return drawn != null ? drawn : player.getNumDrawnThisTurn() > before;
    }

    private void settle() {
        if (drawn == null) {
            drawn = player.getNumDrawnThisTurn() > before;
        }
    }

    /** The checks a reader has not settled yet. */
    static final class Pending {

        private final List<DrawCheck> unsettled = new ArrayList<>();

        /** A check of a move out of {@code player}'s library, read from now on. */
        DrawCheck open(Player player) {
            DrawCheck check = new DrawCheck(player);
            unsettled.add(check);
            return check;
        }

        /** Settle {@code player}'s checks: another card is leaving that library. */
        void settle(Player player) {
            unsettled.removeIf(check -> {
                if (check.player != player) {
                    return false;
                }
                check.settle();
                return true;
            });
        }

        /** Settle every check: a step boundary, past which no move of theirs is still being drawn. */
        void settleAll() {
            unsettled.forEach(DrawCheck::settle);
            unsettled.clear();
        }
    }
}
