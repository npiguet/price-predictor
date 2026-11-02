package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.GamePlayer;
import com.pricepredictor.connector.RandomSeatController;
import com.pricepredictor.connector.RandomSeatLobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A fork played on to the end of the turn inside a real match: a game the
 * engine plays from its first turn to its last, with a random seat, recorded
 * as {@code GamePlayer} records it, forked once at a phase boundary of a
 * later turn the way the directed-fork scheduler will fork it (D4).
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class ForkPlayForwardIntegrationTest {

    @TempDir
    Path tempDir;

    private static Deck deck(String land, Map<String, Integer> spells) {
        Deck deck = new Deck();
        deck.getMain().add(land, 17);
        spells.forEach((name, count) -> deck.getMain().add(name, count));
        return deck;
    }

    /** Forks once, at the second main phase of the first turn from the fifth on. */
    private static final class Forker {
        private final Game live;
        private final GameRecording recording;
        ForkPlayForward.Played played;
        int forkedOnTurn = -1;

        Forker(Game live, GameRecording recording) {
            this.live = live;
            this.recording = recording;
        }

        @Subscribe
        public void onPhase(GameEventTurnPhase event) {
            if (forkedOnTurn >= 0 || event.phase() != PhaseType.MAIN2
                    || live.getPhaseHandler().getTurn() < 5) {
                return;
            }
            forkedOnTurn = live.getPhaseHandler().getTurn();
            played = new Forks.ForkRun(live, () -> { }, () -> live.getPhaseHandler().getPlayerTurn())
                    .take((verified, snapshots) -> recording.playForward().play(
                                    verified.fork(), record -> record.fork(true).interventional(false)),
                            (verified, result) -> result);
        }
    }

    @Test
    void aForkPlaysOnInsideARealMatchAndTheMatchPlaysOn() throws Throwable {
        Deck red = deck("Mountain", Map.of(
                "Goblin Piker", 4, "Hill Giant", 4, "Lightning Bolt", 4, "Gray Ogre", 4,
                "Raging Goblin", 4, "Shock", 3));
        Deck green = deck("Forest", Map.of(
                "Grizzly Bears", 4, "Giant Growth", 4, "Craw Wurm", 4, "Llanowar Elves", 4,
                "Trained Armodon", 4, "Giant Spider", 3));
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(red).setPlayer(new RandomSeatLobbyPlayer("A", 0.5, new Random(3L))),
                new RegisteredPlayer(green).setPlayer(new LobbyPlayerAi("B", null)));
        GameRules rules = new GameRules(GameType.Sealed);
        rules.setGamesPerMatch(1);
        Match match = new Match(rules, players, "fork-play-forward");
        Game game = match.createGame();
        RandomSeatController seat = null;
        for (Player player : game.getPlayers()) {
            if (player.getController() instanceof RandomSeatController random) {
                seat = random;
            }
        }
        assertNotNull(seat);
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).build();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        Forker forker;
        try (GameRecording recording = new GameRecording(
                game, writer, ShardRun.GAME_ID, caps, 11L, Refs.of(seat.getPlayer()),
                GamePlayer::attachForkSeats)) {
            recording.open();
            seat.attach(game, recording.priorityCandidates(), recording.fruitlessActivations());
            forker = new Forker(game, recording);
            game.subscribeToEvents(forker);
            match.startGame(game);
        } finally {
            writer.close();
        }
        List<String> all = ShardRun.readShard(writer.path());

        assertTrue(game.isGameOver(), "the match's game was played to its end");
        assertTrue(forker.forkedOnTurn >= 5, "the game reached a fifth turn and forked there");
        assertNotNull(forker.played, "the fork passed its score check and played out");
        List<String> forked = all.stream().filter(r -> Boolean.TRUE.equals(parse(r).get("fork"))).toList();
        assertEquals(forker.played.records(), forked.size(), "every line the fork held was written");
        assertFalse(forked.isEmpty(), "the fork's turn wrote records");
        for (String record : forked) {
            assertEquals(ShardRun.GAME_ID, parse(record).get("game_id"));
            assertEquals(Boolean.FALSE, parse(record).get("interventional"));
        }
        long live = all.size() - forked.size();
        assertTrue(live > forked.size(), "the live game recorded on around the fork");
    }
}
