package com.pricepredictor.connector;

import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;

import java.util.List;

/**
 * A player-less game to build cards against outside any match.
 *
 * <p>{@code CardFactory} needs a game to set a card's abilities up against, and
 * with none at all a few cards' setup throws. Converting a script and asking
 * whether a card can be cast both build cards with no match running, so both
 * use this one. Created once, on first use.
 */
final class OfflineGame {

    private OfflineGame() {
    }

    static Game instance() {
        return Holder.INSTANCE;
    }

    private static final class Holder {
        static final Game INSTANCE = create();

        private static Game create() {
            GameRules rules = new GameRules(GameType.Constructed);
            return new Game(List.of(), rules, new Match(rules, List.of(), "offline"));
        }
    }
}
