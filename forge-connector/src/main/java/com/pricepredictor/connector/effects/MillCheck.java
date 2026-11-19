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
 * out of the same library, when the card itself is removed from the graveyard
 * (the moved card has lost the flag, so the check reads it off the card object
 * it kept) and at every step boundary. A card no longer in the graveyard when
 * the flag is read was not milled as far as this can tell.
 *
 * <p>A surveil's bin counts too: {@code Player.surveil} flags its cards
 * surveilled rather than milled, and the record schema tells it as a mill
 * (see {@link BusEvents#surveil}).
 */
final class MillCheck {

    private final Player player;
    private final int cardId;
    private Boolean milled;
    /** The card object in the owner's graveyard when the move was filed, or null; it keeps its flags once the card has moved on. */
    private final Card binned;

    private MillCheck(Player player, int cardId) {
        this.player = player;
        this.cardId = cardId;
        this.binned = inGraveyard();
    }

    /** Whether the move was a mill, as far as the card's flag says now or said when settled. */
    boolean milled() {
        return milled != null ? milled : flagged();
    }

    private boolean flagged() {
        Card card = inGraveyard();
        return card != null && flagOf(card);
    }

    private Card inGraveyard() {
        for (Card card : player.getCardsIn(ZoneType.Graveyard)) {
            if (card.getId() == cardId) {
                return card;
            }
        }
        return null;
    }

    private static boolean flagOf(Card card) {
        return card.wasMilled() || card.wasSurveilled();
    }

    private void settle() {
        if (milled == null) {
            milled = flagged();
        }
    }

    /** Settle on the flag of the card object that sat in the graveyard: it is leaving. */
    private void settleLeaving() {
        if (milled == null) {
            milled = binned != null && flagOf(binned);
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

        /** Settle card {@code cardId}'s check: the card is being removed from its owner's graveyard. */
        void settleCard(int cardId) {
            unsettled.removeIf(check -> {
                if (check.cardId != cardId) {
                    return false;
                }
                check.settleLeaving();
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
