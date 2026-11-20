package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * Which face of a card is up, as an entity's {@code face_state} and a
 * {@code face_change}'s {@code to_state} spell it.
 *
 * <p>{@code FaceState} in {@code state_snapshot.py} is the other half of this
 * vocabulary, in the same order. A split card, a prepared spell and a
 * specialized card are on their {@link #FRONT}.
 */
enum FaceState implements WireNamed {
    FRONT("front"),
    BACK("back"),
    FLIPPED("flipped"),
    MELDED("melded"),
    FACE_DOWN("face_down");

    private final String wireName;

    FaceState(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code face_state} an entity carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
