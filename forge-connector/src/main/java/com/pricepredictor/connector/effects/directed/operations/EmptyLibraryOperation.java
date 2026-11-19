package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Objects;

/**
 * {@code move_card} on a whole library: the player's library is empty
 * (FR-011a), the variant's one fact where its point is a library with
 * nothing to reveal or find (701.44b, 701.57b).
 *
 * <p>A raw edit of the library's card list, as {@link MoveCardOperation}'s:
 * every card leaves it for no zone, as a card replaced in place does. No
 * card is milled, exiled or put into a graveyard, so no trigger fires and no
 * per-turn count moves, and no other zone changes: a graveyard the cards
 * joined would change what counts it (threshold, delirium, a Lhurgoyf).
 *
 * @param owner the fork's player whose library is emptied
 */
public record EmptyLibraryOperation(Player owner) implements BoardEdit {

    public EmptyLibraryOperation {
        Objects.requireNonNull(owner, "owner");
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.MOVE_CARD;
    }

    @Override
    public void apply(Game fork) {
        Zone library = owner.getZone(ZoneType.Library);
        if (library.size() == 0) {
            throw new IllegalStateException("the library is empty already: " + owner);
        }
        for (Card card : List.copyOf(library.getCards())) {
            library.remove(card);
            card.setZone(null);
        }
        BoardEdit.settle(fork);
    }
}
