package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Unit tests for the pieces of {@link DraftGamePlayer} that do not need a live
 * Forge game: the per-game sequences a match-outcome row is built from, and the
 * row's field bindings.
 *
 * <p>Playing an actual match needs an initialised Forge environment and belongs
 * to the integration profile.
 */
class DraftGamePlayerTest {

    private static GamePlayer.GameOutcome game(String winner, String playFirst) {
        return new GamePlayer.GameOutcome(winner, playFirst, List.of(), List.of());
    }

    private static GamePlayer.PlayedMatch match(GamePlayer.GameOutcome... games) {
        return new GamePlayer.PlayedMatch(List.of(games), 87);
    }

    // ── Per-game sequences ──────────────────────────────────────────

    @Test
    void winnerSequenceIsOneCharPerGame() {
        assertEquals("ABA", DraftGamePlayer.winnerSequence(
                match(game("A", "A"), game("B", "B"), game("A", "A"))));
    }

    @Test
    void playSequenceIsOneCharPerGame() {
        assertEquals("BAB", DraftGamePlayer.playSequence(
                match(game("A", "B"), game("B", "A"), game("A", "B"))));
    }

    @Test
    void sequencesHaveEqualLength() {
        GamePlayer.PlayedMatch played = match(game("A", "B"), game("A", "A"));

        assertEquals(
                DraftGamePlayer.winnerSequence(played).length(),
                DraftGamePlayer.playSequence(played).length());
    }

    @Test
    void aBestOfOneMatchIsASingleChar() {
        assertEquals("B", DraftGamePlayer.winnerSequence(match(game("B", "A"))));
    }

    // ── Row construction ────────────────────────────────────────────

    @Test
    void rowCarriesLabelsSetAndSequences() {
        GamePlayer.PlayedMatch played = match(game("A", "A"), game("A", "B"));

        MatchResult result = new MatchResult(
                Instant.parse("2026-08-11T00:00:00Z"),
                "eval-1",
                "BLB",
                "gen4",
                "forge-native",
                List.of("Forest", "Island"),
                List.of("Mountain", "Plains"),
                DraftGamePlayer.winnerSequence(played),
                DraftGamePlayer.playSequence(played),
                played.durationSeconds());

        assertEquals("gen4", result.methodA());
        assertEquals("forge-native", result.methodB());
        assertEquals("BLB", result.setCode());
        assertEquals("AA", result.games());
        assertEquals("AB", result.play());
        assertEquals(87, result.durationSeconds());
    }

    @Test
    void winCountsDeriveFromTheSequence() {
        GamePlayer.PlayedMatch played =
                match(game("A", "A"), game("B", "B"), game("A", "A"));

        MatchResult result = new MatchResult(
                Instant.parse("2026-08-11T00:00:00Z"),
                "eval-1", "BLB", "gen4", "gen1",
                List.of("Forest"), List.of("Island"),
                DraftGamePlayer.winnerSequence(played),
                DraftGamePlayer.playSequence(played),
                played.durationSeconds());

        assertEquals(2, result.winsA());
        assertEquals(1, result.winsB());
    }

    @Test
    void aMatchWithNoGamesCannotBecomeARow() {
        // GamePlayer skips draws and aborted games, so a match can end up empty.
        // MatchResult rejects it rather than letting a blank row reach the corpus.
        GamePlayer.PlayedMatch empty = new GamePlayer.PlayedMatch(List.of(), 0);

        assertThrows(IllegalArgumentException.class, () -> new MatchResult(
                Instant.parse("2026-08-11T00:00:00Z"),
                "eval-1", "BLB", "gen4", "gen1",
                List.of("Forest"), List.of("Island"),
                DraftGamePlayer.winnerSequence(empty),
                DraftGamePlayer.playSequence(empty),
                empty.durationSeconds()));
    }
}
