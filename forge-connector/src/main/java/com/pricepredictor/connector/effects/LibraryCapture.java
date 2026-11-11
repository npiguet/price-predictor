package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
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
 * <p>Rendered as the card was in the library (FR-042e, root § 8.14): a fresh
 * object built from the card's printing under the same id, as Forge builds a
 * card leaving the library for any zone but the battlefield. So it reads
 * untapped, with no counters, attached to nothing, in no combat, controlled
 * by its owner, face up, not entered this turn and with nothing chosen, its
 * name, types and lines its library face's — whatever the record's events
 * then did to it. Forge keeps the library's own object for a move onto the
 * battlefield, so reading that object at the write would describe it after
 * the move: tapped, sick, face down for a manifest, a copy for a clone. A
 * card with no printing to build from (a token, a copy) is read as it is.
 * Its zone and rank are written as captured.
 *
 * @param present the ids already among the snapshot's entities
 */
record LibraryCapture(
        Game game, StaticSuppression board, Map<String, Slot> byId, Set<String> present) {

    /** One library card and its position, 0 the top. */
    record Slot(Card card, int position) {
    }

    /** Every card of every library as it sits now. */
    static LibraryCapture of(Game game, StaticSuppression board, Set<String> present) {
        Map<String, Slot> byId = new LinkedHashMap<>();
        for (Player player : game.getPlayers()) {
            int position = 0;
            for (Card card : player.getCardsIn(ZoneType.Library)) {
                byId.put(Refs.of(card), new Slot(card, position++));
            }
        }
        return new LibraryCapture(game, board, byId, present);
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
                    rendered.add(EntityJson.render(game, asInLibrary(slot.card()), board,
                            new EntityJson.Beyond(null, "library", slot.position(),
                                    SnapshotBuilder.ROLE_LIBRARY_CARD)));
                }
            }
        }
        return String.join(",", rendered);
    }

    /**
     * The card as it was in the library: a fresh object from its printing,
     * under its own id and owner, or the card itself where it has no
     * printing to build from.
     */
    static Card asInLibrary(Card card) {
        if (card.getPaperCard() == null || card.isToken() || card.isCopiedSpell()
                || card.getCopiedPermanent() != null) {
            return card;
        }
        try {
            Card fresh = CardFactory.getCard(
                    card.getPaperCard(), card.getOwner(), card.getId(), card.getGame());
            // Not entered this turn: a new object takes the current turn as
            // the one it came into its zone, and a library card is in no
            // zone a turn could name.
            fresh.setTurnInZone(-1);
            return fresh;
        } catch (RuntimeException e) {
            return card;
        }
    }
}
