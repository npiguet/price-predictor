package com.pricepredictor.connector.effects.directed;

import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.Objects;

/**
 * The name a rule gives one object its precondition binds ({@code "creature"},
 * {@code "attacker"}, {@code "defender"}), with the kind of object it names:
 * declared once per rule, read in its {@code bind}, its edit and its
 * {@link ForcedPlay}, so a misspelt role is a compile error instead of a
 * failure at fork time. The {@link #name} is the string the binding holds the
 * object under, so a typed role and the same name as a string are one role.
 *
 * @param name the role's name
 * @param type the kind of object bound, {@link Card} or {@link Player}
 * @param <T>  the kind of object bound
 */
public record Role<T extends GameObject>(String name, Class<T> type) {

    public Role {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(type, "type");
    }

    /** A role bound to a card. */
    public static Role<Card> card(String name) {
        return new Role<>(name, Card.class);
    }

    /** A role bound to a player. */
    public static Role<Player> player(String name) {
        return new Role<>(name, Player.class);
    }

    @Override
    public String toString() {
        return name;
    }
}
