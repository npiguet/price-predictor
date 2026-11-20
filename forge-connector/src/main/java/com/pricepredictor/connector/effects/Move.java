package com.pricepredictor.connector.effects;

import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.player.PlayerView;
import forge.game.zone.ZoneType;

/**
 * A card's move between zones, as the bus publishes it, answering the
 * questions the readers ask of it: which zone it left, which it reached.
 *
 * <p>Forge leaves either end null for a card that came from or went to no
 * zone; every question answers {@code false} (or null) for a missing end.
 */
record Move(GameEventCardChangeZone event) {

    /** The card that moved, or null when the event names none. */
    CardView card() {
        return event.card();
    }

    /** The card's owner, or null when there is no card or it has none. */
    PlayerView owner() {
        CardView card = event.card();
        return card == null ? null : card.getOwner();
    }

    /** Whether the card left {@code zone}. */
    boolean from(ZoneType zone) {
        return event.from() != null && event.from().zoneType() == zone;
    }

    /** Whether the card reached {@code zone}. */
    boolean to(ZoneType zone) {
        return event.to() != null && event.to().zoneType() == zone;
    }

    /** The zone the card left, or null when it left none. */
    ZoneType fromZone() {
        return event.from() == null ? null : event.from().zoneType();
    }

    /** The zone the card reached, or null when it reached none. */
    ZoneType toZone() {
        return event.to() == null ? null : event.to().zoneType();
    }

    /** The zone the card left as a record spells it, or null. */
    String fromName() {
        return CardSpelling.zoneName(fromZone());
    }

    /** The zone the card reached as a record spells it, or null. */
    String toName() {
        return CardSpelling.zoneName(toZone());
    }

    /** The player whose zone the card reached, or null. */
    PlayerView toPlayer() {
        return event.to() == null ? null : event.to().player();
    }

    /** Whether the card left the stack and it is the card {@code cardId}. */
    boolean leftStackAs(int cardId) {
        return event.card() != null && event.card().getId() == cardId && from(ZoneType.Stack);
    }
}
