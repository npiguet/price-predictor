package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameCopier;
import forge.game.Game;

/** The fork machinery a test outside this package checks its copies with. */
public final class TestForks {

    private TestForks() {
    }

    /**
     * Whether {@code fork}, a copy of {@code live} made by {@code copier},
     * passes the score check every fork passes before it is changed
     * ({@link Forks#scoreCheck}), from the active player's side.
     */
    public static boolean scoreChecks(Game live, GameCopier copier, Game fork) {
        return Forks.scoreCheck(live, copier, fork, live.getPhaseHandler().getPlayerTurn());
    }
}
