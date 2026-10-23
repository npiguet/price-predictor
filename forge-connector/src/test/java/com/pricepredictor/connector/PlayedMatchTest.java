package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for {@link GamePlayer.PlayedMatch}: the per-game results and win
 * counts a match row is built from.
 *
 * <p>Playing an actual match needs an initialised Forge environment and belongs
 * to the integration profile.
 */
class PlayedMatchTest {

    private static GamePlayer.GameOutcome game(Side winner, Side playFirst) {
        return new GamePlayer.GameOutcome(winner, playFirst, List.of("Shock"), List.of());
    }

    private static GamePlayer.PlayedMatch match(GamePlayer.GameOutcome... games) {
        return new GamePlayer.PlayedMatch(List.of(games), 87, false);
    }

    @Test
    void resultsKeepEachGamesWinnerAndStarterInGameOrder() {
        GamePlayer.PlayedMatch played =
                match(game(Side.A, Side.B), game(Side.B, Side.A), game(Side.A, Side.B));

        assertEquals(MatchFixtures.games("ABA", "BAB"), played.results());
    }

    @Test
    void winCountsDeriveFromTheGames() {
        GamePlayer.PlayedMatch played =
                match(game(Side.A, Side.A), game(Side.B, Side.B), game(Side.A, Side.A));

        assertEquals(2, played.winsA());
        assertEquals(1, played.winsB());
    }
}
