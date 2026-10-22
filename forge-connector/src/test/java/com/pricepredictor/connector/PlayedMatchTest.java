package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Unit tests for {@link GamePlayer.PlayedMatch}: the per-game sequences and
 * win counts a draft-game row is built from, and the empty-match guard.
 *
 * <p>Playing an actual match needs an initialised Forge environment and belongs
 * to the integration profile.
 */
class PlayedMatchTest {

    private static GamePlayer.GameOutcome game(Side winner, Side playFirst) {
        return new GamePlayer.GameOutcome(winner, playFirst, List.of(), List.of());
    }

    private static GamePlayer.PlayedMatch match(GamePlayer.GameOutcome... games) {
        return new GamePlayer.PlayedMatch(List.of(games), 87, false);
    }

    // ── Per-game sequences ──────────────────────────────────────────

    @Test
    void winnerSequenceIsOneCharPerGame() {
        assertEquals("ABA", match(game(Side.A, Side.A), game(Side.B, Side.B), game(Side.A, Side.A)).winnerSequence());
    }

    @Test
    void playSequenceIsOneCharPerGame() {
        assertEquals("BAB", match(game(Side.A, Side.B), game(Side.B, Side.A), game(Side.A, Side.B)).playSequence());
    }

    @Test
    void sequencesHaveEqualLength() {
        GamePlayer.PlayedMatch played = match(game(Side.A, Side.B), game(Side.A, Side.A));

        assertEquals(
                played.winnerSequence().length(),
                played.playSequence().length());
    }

    @Test
    void aBestOfOneMatchIsASingleChar() {
        assertEquals("B", match(game(Side.B, Side.A)).winnerSequence());
    }

    // ── Win counts ──────────────────────────────────────────────────

    @Test
    void winCountsDeriveFromTheSequence() {
        GamePlayer.PlayedMatch played =
                match(game(Side.A, Side.A), game(Side.B, Side.B), game(Side.A, Side.A));

        assertEquals(2, played.winsA());
        assertEquals(1, played.winsB());
    }

    // ── Empty matches ─────────────────────────────────────────────────────────────────────────────────────────

    @Test
    void aMatchWithNoGamesCannotBecomeARow() {
        // GamePlayer skips draws and aborted games, so a match can end up empty.
        // MatchResult rejects it rather than letting a blank row reach the corpus.
        GamePlayer.PlayedMatch empty = new GamePlayer.PlayedMatch(List.of(), 0, false);

        assertThrows(IllegalArgumentException.class, () -> new MatchResult(
                Instant.parse("2026-08-11T00:00:00Z"),
                "eval-1", "BLB", "gen4", "gen1",
                List.of("Forest"), List.of("Island"),
                empty.winnerSequence(),
                empty.playSequence(),
                empty.durationSeconds()));
    }
}
