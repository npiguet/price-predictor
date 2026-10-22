package com.pricepredictor.connector;

/**
 * Which deck of a match: A or B, as every corpus row spells it.
 *
 * <p>{@link #toString()} is the wire value — the one character a
 * {@code games} / {@code play} sequence or a {@code winner} / {@code starter}
 * field carries — so it must stay the bare constant name.
 */
public enum Side {
    A("p1"),
    B("p2");

    private final String lobbyName;

    Side(String lobbyName) {
        this.lobbyName = lobbyName;
    }

    /** The name the side's Forge lobby player is registered under. */
    public String lobbyName() {
        return lobbyName;
    }

    /**
     * The side a Forge lobby player name belongs to.
     *
     * @throws IllegalArgumentException for a name no side registers: guessing
     *         would credit a game or a card to the wrong deck
     */
    public static Side ofLobbyName(String lobbyName) {
        for (Side side : values()) {
            if (side.lobbyName.equals(lobbyName)) {
                return side;
            }
        }
        throw new IllegalArgumentException("Unknown lobby name: " + lobbyName);
    }
}
