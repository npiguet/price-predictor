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
 */
final class Refs {

    private Refs() {
    }

    static String of(Card card) {
        return "E" + card.getId();
    }

    static String of(Player player) {
        return "P" + player.getId();
    }

    static String of(CardView card) {
        return "E" + card.getId();
    }

    static String of(PlayerView player) {
        return "P" + player.getId();
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
