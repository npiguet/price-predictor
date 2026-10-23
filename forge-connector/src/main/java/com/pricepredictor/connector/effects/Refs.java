package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.player.Player;
import forge.game.player.PlayerView;

import java.util.Optional;

/**
 * How a record names a game object: {@code E<id>} for a card, {@code P<id>}
 * for a player.
 *
 * <p>The one spelling every subject, cause and target in a record uses, and
 * the one {@code state.entities} and {@code state.players} are keyed by, so a
 * reader can join any ref to the board beside it. The bus hands out views
 * rather than cards and players; both forms name the same object.
 *
 * <p>Public because the root-package workers name players the same way when
 * they stamp a record's actor, and a second spelling there would break the
 * join.
 */
public final class Refs {

    private Refs() {
    }

    public static String of(Card card) {
        return "E" + card.getId();
    }

    public static String of(Player player) {
        return "P" + player.getId();
    }

    public static String of(CardView card) {
        return "E" + card.getId();
    }

    public static String of(PlayerView player) {
        return "P" + player.getId();
    }

    /**
     * The ref, or null for no object: for a slot the engine may leave empty,
     * such as a controller, a cause or an attachment, which the record then
     * writes as an honest JSON {@code null}.
     */
    public static String ofNullable(Card card) {
        return card == null ? null : of(card);
    }

    /** {@link #ofNullable(Card)}, for a player slot. */
    public static String ofNullable(Player player) {
        return player == null ? null : of(player);
    }

    /** {@link #ofNullable(Card)}, for a card the bus hands out as a view. */
    public static String ofNullable(CardView card) {
        return card == null ? null : of(card);
    }

    /** {@link #ofNullable(Card)}, for a player the bus hands out as a view. */
    public static String ofNullable(PlayerView player) {
        return player == null ? null : of(player);
    }

    /**
     * A card's or a player's ref, or empty for anything else: what a target
     * list, an entity parameter or a payload slot holds may be either, or
     * something no ref names.
     */
    static Optional<String> ofAny(Object value) {
        if (value instanceof Card card) {
            return Optional.of(of(card));
        }
        if (value instanceof Player player) {
            return Optional.of(of(player));
        }
        return Optional.empty();
    }

    /** Whether a ref names a card rather than a player. */
    static boolean isEntity(String ref) {
        return ref.startsWith("E");
    }
}
