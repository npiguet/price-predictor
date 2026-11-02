package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.Map;

/**
 * A game's board-only snapshot, parsed, for tests outside this package that
 * compare a board before and after a change.
 */
public final class BoardStates {

    private BoardStates() {
    }

    /** The snapshot a record with no acting line would carry: {@code global}, {@code players}, {@code entities}, {@code links}. */
    public static Map<String, Object> of(Game game) {
        return RecordLines.parse(new SnapshotBuilder(game).ofBoard().render());
    }
}
