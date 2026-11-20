package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The entities one snapshot renders, as they are built: each rendered entity
 * in order, the ref of each (so a card is written once, in the first tier or
 * beyond-the-tiers place that reaches it) and the cards among them.
 *
 * <p>Built from the four tiers, which are rendered first; the cards beyond
 * the tiers join through {@link #addLibraryCard} and {@link #addBeyond} and a
 * last-known copy, which is no card of the game, through
 * {@link #addLastKnown}.
 */
final class SnapshotEntities {

    private final Game game;
    private final StaticSuppression board;
    private final Set<Card> tiers;
    private final List<String> rendered = new ArrayList<>();
    private final Set<String> present = new HashSet<>();
    /** The card entities rendered so far; a last-known copy is no card of the game, and is not among them. */
    private final List<Card> cards;

    SnapshotEntities(Game game, StaticSuppression board, Set<Card> tiers) {
        this.game = game;
        this.board = board;
        this.tiers = tiers;
        this.cards = new ArrayList<>(tiers);
    }

    Game game() {
        return game;
    }

    StaticSuppression board() {
        return board;
    }

    /** The cards of the four tiers, in order. */
    Set<Card> tiers() {
        return tiers;
    }

    /** Every card entity rendered so far, in order. */
    List<Card> cards() {
        return cards;
    }

    /** The ref of every entity rendered so far. */
    Set<String> present() {
        return present;
    }

    /** The entities rendered so far, each as one JSON element, in order. */
    List<String> elements() {
        return List.copyOf(rendered);
    }

    /** A tier's card, already rendered as {@code json}. */
    void addTier(Card card, String json) {
        rendered.add(json);
        present.add(Refs.of(card));
    }

    /** {@code gone}, a target as it was targeted, as an {@code L<id>} entity, unless one is there. */
    void addLastKnownTarget(Card gone) {
        String ref = ChainTargets.lastKnownRef(gone);
        if (present.add(ref)) {
            rendered.add(EntityJson.render(game, gone, board,
                    EntityJson.Beyond.lastKnown(ref, EntityRole.LAST_KNOWN)));
        }
    }

    /**
     * A library card, read as it is in the library (FR-042e), at its
     * position: the same rendering whether the snapshot holds it as a
     * library's top card or the record's events name it ({@link
     * LibraryCapture}).
     */
    void addLibraryCard(Card card, int position) {
        if (card != null && present.add(Refs.of(card))) {
            cards.add(card);
            rendered.add(EntityJson.render(game, LibraryCapture.asInLibrary(card), board,
                    new EntityJson.Beyond(null, "library", position,
                            EntityRole.LIBRARY_CARD)));
        }
    }

    /** {@code card}, with {@code role}, unless it is already an entity. */
    void addBeyond(Card card, EntityRole role) {
        if (card != null && present.add(Refs.of(card))) {
            cards.add(card);
            rendered.add(EntityJson.render(game, card, board,
                    new EntityJson.Beyond(null, null, null, role)));
        }
    }

    /** {@code lastKnown}, a pending event's subject as it last existed, under {@code L<card id>}. */
    void addLastKnown(Card lastKnown) {
        // A last-known copy is in no zone; it names the one it was in.
        var zone = lastKnown.getZone() != null ? lastKnown.getZone() : lastKnown.getLastKnownZone();
        rendered.add(EntityJson.render(game, lastKnown, board, new EntityJson.Beyond(
                "L" + lastKnown.getId(),
                zone == null ? null : CardSpelling.zoneName(zone.getZoneType()),
                null, EntityRole.LAST_KNOWN)));
    }
}
