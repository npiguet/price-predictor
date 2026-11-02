package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Where every library card sat when a snapshot was captured, for the library
 * cards the record's events name (FR-024, {@code in.library_cards}).
 *
 * <p>A snapshot is taken before its events, so which library cards they will
 * name — the card drawn, the cards milled, revealed or searched out — is
 * known only when the record is written. Rendering every library card into
 * every snapshot would multiply a record's size for the few it needs, so the
 * capture keeps only each card and its position, and a card is rendered when
 * an event names it.
 *
 * <p>Rendered then from the object Forge held in the library. A card moved to
 * any zone but the battlefield is a new object, so the one kept here is still
 * the card as it sat; one put onto the battlefield is the same object, so the
 * characteristics a library card cannot change (name, types, lines) are read
 * right and its board state (tapped, counters) as it is after the move. Its
 * zone and rank are written as captured either way.
 *
 * @param present the ids already among the snapshot's entities
 */
record LibraryCapture(
        Game game, Long withoutStatic, Map<String, Slot> byId, Set<String> present) {

    /** One library card and its position, 0 the top. */
    record Slot(Card card, int position) {
    }

    /** Every card of every library as it sits now. */
    static LibraryCapture of(Game game, Long withoutStatic, Set<String> present) {
        Map<String, Slot> byId = new LinkedHashMap<>();
        for (Player player : game.getPlayers()) {
            int position = 0;
            for (Card card : player.getCardsIn(ZoneType.Library)) {
                byId.put(Refs.of(card), new Slot(card, position++));
            }
        }
        return new LibraryCapture(game, withoutStatic, byId, present);
    }

    /**
     * The entities of the captured library cards {@code events} name and the
     * snapshot does not already hold, comma-joined in the order first named;
     * empty when none.
     */
    String render(Iterable<EffectEvent> events) {
        List<String> rendered = new ArrayList<>();
        Set<String> added = new java.util.HashSet<>(present);
        for (EffectEvent event : events) {
            for (String subject : event.subjects()) {
                Slot slot = byId.get(subject);
                if (slot != null && added.add(subject)) {
                    rendered.add(EntityJson.render(game, slot.card(), withoutStatic,
                            new EntityJson.Beyond(null, "library", slot.position(),
                                    SnapshotBuilder.ROLE_LIBRARY_CARD)));
                }
            }
        }
        return String.join(",", rendered);
    }
}
