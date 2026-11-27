package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.zone.ZoneType;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordPicks.withEventOfType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A replaced game loss is not the game's outcome:
 * Lich's Mirror's rewrite record carries {@code player_lost} as its
 * {@code incoming} and as its state's {@code pending_event}, yet the game
 * goes on, so the record is capped like any other and does not let the
 * game-end marker through. Only the record whose own events end the game
 * holds the outcome.
 */
@ExtendWith(ForgeExtension.class)
class ReplacedGameLossTest {

    private static final String WHY = "for the rewrite record and the game-end marker";

    @TempDir
    Path tempDir;

    /**
     * The cap is 0, so every record not holding the outcome is withheld.
     * The loss replaced by the Mirror writes no record; the loss that then
     * really ends the game writes the game-end record, and the marker
     * follows it.
     */
    @Test
    void aReplacedLossIsCappedAndTheMarkerFollowsTheRealOutcomeOnly() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player loser = game.getPlayers().get(0);
        stockLibraryAndPlayTheMirror(game, loser);

        List<String> records = ShardRun.readShard(recorded(game, 0, () -> {
            loser.setLife(0, null);
            game.getAction().checkStateEffects(false);
            assertFalse(game.isGameOver(), "the Mirror replaced the loss");
            loser.setLife(0, null);
            game.getAction().checkStateEffects(false);
            assertTrue(game.isGameOver(), "the loss without the Mirror ended the game");
            game.fireEvent(new GameEventGameFinished());
        }));

        assertEquals(List.of(), records.stream().filter(line -> line.contains("\"kind\":\"rewrite\"")).toList(),
                "the replaced loss's rewrite record is capped: " + records);
        List<String> lost = withEventOfType(records, "player_lost");
        assertEquals(1, lost.size(), records.toString());
        assertEquals(List.of("player_lost", "player_won"), eventTypes(lost.get(0)), lost.get(0));
        assertTrue(isMarker(records.get(records.size() - 1)), "the marker last: " + records);
    }

    /**
     * The loss is replaced and the game never ends: no record holds the
     * outcome, so the game writes no marker and reads as cut.
     */
    @Test
    void aGameWhoseOnlyLossWasReplacedHasNoMarker() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player loser = game.getPlayers().get(0);
        stockLibraryAndPlayTheMirror(game, loser);

        Path shard = recorded(game, 0, () -> {
            loser.setLife(0, null);
            game.getAction().checkStateEffects(false);
            assertFalse(game.isGameOver(), "the Mirror replaced the loss");
            game.fireEvent(new GameEventGameFinished());
        });

        assertFalse(Files.exists(shard), "nothing holds the outcome, so no record is written");
    }

    /**
     * Ten cards in the library for the Mirror's seven-card draw (the Mirror
     * itself goes into it, and a draw from an empty one would lose the game
     * again), and the Mirror in play.
     */
    private static void stockLibraryAndPlayTheMirror(Game game, Player loser) {
        for (int i = 0; i < 10; i++) {
            game.getAction().moveTo(ZoneType.Library, TestBoard.card(game, loser, "Forest"), null, null);
        }
        TestBoard.inPlay(game, loser, "Lich's Mirror");
    }

    private Path recorded(Game game, int maxRecords, ShardRun.Played body) throws Throwable {
        RecordShardWriter writer = new RecordShardWriter(
                tempDir, new ShardId("run", 0, "l1"), maxRecords);
        CollectionCaps noForks = TestCaps.caps().interventions(0).probes(0).build();
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, writer, ShardRun.GAME_ID, noForks).seeded(1L))) {
            recording.open();
            body.run();
        } finally {
            writer.close();
        }
        return writer.path();
    }

    private static boolean isMarker(String line) {
        return line.contains("\"kind\":\"turn_action\"") && line.contains("\"subkind\":\"game_end\"");
    }
}
