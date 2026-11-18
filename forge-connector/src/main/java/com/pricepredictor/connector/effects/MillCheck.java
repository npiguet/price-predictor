package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * Whether a card's move from a library to its owner's graveyard was a mill.
 *
 * <p>Forge publishes no mill event and keeps no per-player mill counter: a
 * mill, an Entomb, a dig's rest and a surveil's bin all publish the same
 * library-to-graveyard move. What tells a mill apart is the card's own flag,
 * which {@code Player.mill} sets with {@code Card.setMilled} right after the
 * move it publishes. So the move is filed as a mill provisionally, and the
 * flag is read once it can have been set: on the card now in the owner's
 * graveyard, live until the check is settled, and settled at the next move
 * out of the same library and at every step boundary, as a {@link DrawCheck}
 * is. A card no longer in the graveyard when the flag is read was not milled
 * as far as this can tell.
 *
 * <p>A surveil's bin counts too: {@code Player.surveil} flags its cards
 * surveilled rather than milled, and the record schema tells it as a mill
 * (see {@link BusEvents#surveil}).
 */
final class MillCheck {

    private final Player player;
    private final int cardId;
    private Boolean milled;

    private MillCheck(Player player, int cardId) {
        this.player = player;
        this.cardId = cardId;
    }

    /** Whether the move was a mill, as far as the card's flag says now or said when settled. */
    boolean milled() {
        return milled != null ? milled : flagged();
    }

    private boolean flagged() {
        for (Card card : player.getCardsIn(ZoneType.Graveyard)) {
            if (card.getId() == cardId) {
                return card.wasMilled() || card.wasSurveilled();
            }
        }
        return false;
    }

    private void settle() {
        if (milled == null) {
            milled = flagged();
        }
    }

    /** The checks a reader has not settled yet. */
    static final class Pending {

        private final List<MillCheck> unsettled = new ArrayList<>();

        /** A check of card {@code cardId}'s move out of {@code player}'s library. */
        MillCheck open(Player player, int cardId) {
            MillCheck check = new MillCheck(player, cardId);
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

        /** Settle every check: a step boundary. */
        void settleAll() {
            unsettled.forEach(MillCheck::settle);
            unsettled.clear();
        }
    }
}
