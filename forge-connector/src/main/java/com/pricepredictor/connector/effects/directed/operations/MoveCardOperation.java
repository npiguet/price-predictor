package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.Objects;
import java.util.Set;

/**
 * {@code move_card}: move a card between its owner's hand, library and
 * graveyard, or put a new card from the card database in a card's place in
 * one of them (FR-011a).
 *
 * <p>Raw edits of the zones' card lists: a card leaves one list and joins
 * another at a stated place. No card is drawn, discarded or milled, so no
 * draw, discard or mill trigger fires (no "whenever you draw" or madness),
 * and the per-turn counts stay as the live game left them: cards drawn this
 * turn, cards put into a graveyard this turn, and the descended count a
 * permanent card reaching its owner's graveyard adds to in play. A card
 * replaced in place goes to no zone.
 *
 * <p>Places are indices into Forge's list for the zone: a library's 0 is its
 * top card. A null place puts the card where Forge's list ends (a library's
 * bottom).
 *
 * @param card     the fork's card, in its owner's hand, library or graveyard
 * @param to       the zone it moves to, or null when it is replaced in place
 * @param position its place in {@code to}, or null for the end
 * @param newCard  the name of the card put in its place, or null when it moves
 */
public record MoveCardOperation(Card card, ZoneType to, Integer position, String newCard)
        implements BoardEdit {

    /** The zones a card moves between. */
    public static final Set<ZoneType> ZONES = Set.of(ZoneType.Hand, ZoneType.Library, ZoneType.Graveyard);

    public MoveCardOperation {
        Objects.requireNonNull(card, "card");
        if ((to == null) == (newCard == null)) {
            throw new IllegalArgumentException("a move_card edit either moves the card or replaces it");
        }
        if (to != null && !ZONES.contains(to)) {
            throw new IllegalArgumentException("move_card moves between hand, library and graveyard, not to " + to);
        }
        if (position != null && position < 0) {
            throw new IllegalArgumentException("a place is not negative: " + position);
        }
    }

    /** Move {@code card} to {@code position} of its owner's {@code to}. */
    public static MoveCardOperation move(Card card, ZoneType to, Integer position) {
        return new MoveCardOperation(card, to, position, null);
    }

    /** Put a new {@code newCard} where {@code card} sits, {@code card} going to no zone. */
    public static MoveCardOperation replace(Card card, String newCard) {
        return new MoveCardOperation(card, null, null, newCard);
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.MOVE_CARD;
    }

    @Override
    public void apply(Game fork) {
        Zone from = card.getZone();
        if (from == null || !ZONES.contains(from.getZoneType())) {
            throw new IllegalStateException("not in a hand, library or graveyard: " + card);
        }
        if (newCard == null) {
            Zone target = card.getOwner().getZone(to);
            from.remove(card);
            int place = position == null ? target.size() : position;
            if (place > target.size()) {
                throw new IllegalStateException("no place " + place + " in a " + to
                        + " of " + target.size() + " cards");
            }
            addSilently(target, card, place);
        } else {
            int place = from.getCards().indexOf(card);
            Card fresh = NewCards.create(fork, newCard, card.getOwner());
            fresh.setGameTimestamp(fork.getNextTimestamp());
            from.remove(card);
            card.setZone(null);
            addSilently(from, fresh, place);
        }
        BoardEdit.settle(fork);
    }

    /**
     * Join {@code zone}'s list at {@code place}, as Forge's rollback does: the
     * one form of {@code Zone.add} that leaves the descended count alone,
     * along with the card's "entered this turn" turn (the variant's card was
     * already there).
     */
    private static void addSilently(Zone zone, Card card, int place) {
        zone.add(card, place, null, true);
    }
}
