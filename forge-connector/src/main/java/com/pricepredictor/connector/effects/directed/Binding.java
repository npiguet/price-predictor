package com.pricepredictor.connector.effects.directed;

import forge.ai.simulation.GameCopier;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * What a board precondition found on the live board: the objects a rule's
 * variants act on, each under a role the rule names ({@code "creature"},
 * {@code "attacker"}, {@code "defender"}).
 *
 * <p>A boolean is not enough for a directed fork (D1): the operation needs
 * the bound subject, and a scenario's delta is prose. The precondition binds
 * the live objects; {@link #inFork} maps them into the fork's copy, where the
 * variant's edit applies.
 */
public record Binding(Map<String, Card> cards, Map<String, Player> players) {

    public Binding {
        cards = Map.copyOf(cards);
        players = Map.copyOf(players);
    }

    /** A binding of no object, for a rule whose edit needs none. */
    public static Binding empty() {
        return new Binding(Map.of(), Map.of());
    }

    /** This binding with {@code card} under {@code role}. */
    public Binding with(String role, Card card) {
        Map<String, Card> more = new LinkedHashMap<>(cards);
        more.put(Objects.requireNonNull(role, "role"), Objects.requireNonNull(card, role));
        return new Binding(more, players);
    }

    /** This binding with {@code player} under {@code role}. */
    public Binding with(String role, Player player) {
        Map<String, Player> more = new LinkedHashMap<>(players);
        more.put(Objects.requireNonNull(role, "role"), Objects.requireNonNull(player, role));
        return new Binding(cards, more);
    }

    /** The card bound under {@code role}, failing when there is none. */
    public Card card(String role) {
        Card card = cards.get(role);
        if (card == null) {
            throw new IllegalStateException("no card bound as " + role);
        }
        return card;
    }

    /** The player bound under {@code role}, failing when there is none. */
    public Player player(String role) {
        Player player = players.get(role);
        if (player == null) {
            throw new IllegalStateException("no player bound as " + role);
        }
        return player;
    }

    /**
     * The same roles bound to the fork's copies of the live objects.
     *
     * @throws IllegalStateException when the copy holds no counterpart of a
     *                               bound object, which discards the fork:
     *                               the variant cannot be described on it
     */
    public Binding inFork(GameCopier copier) {
        Map<String, Card> forkCards = new LinkedHashMap<>();
        cards.forEach((role, card) -> forkCards.put(role, mapped(copier, card, role)));
        Map<String, Player> forkPlayers = new LinkedHashMap<>();
        players.forEach((role, player) -> forkPlayers.put(role, mapped(copier, player, role)));
        return new Binding(forkCards, forkPlayers);
    }

    private static <T extends GameObject> T mapped(GameCopier copier, T live, String role) {
        T found;
        try {
            found = copier.find(live);
        } catch (RuntimeException e) {
            throw new IllegalStateException("the fork holds no copy of the " + role, e);
        }
        if (found == null) {
            throw new IllegalStateException("the fork holds no copy of the " + role);
        }
        return found;
    }
}
