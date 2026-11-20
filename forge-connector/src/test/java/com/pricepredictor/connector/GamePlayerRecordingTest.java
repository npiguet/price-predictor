package com.pricepredictor.connector;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.TestCaps;
import forge.ai.PlayerControllerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.Match;
import forge.game.event.GameEventTurnBegan;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Real matches played through {@link GamePlayer#playMatch} with effect
 * records on and a shard writer, as a worker plays them (TC4): the game-end
 * marker each game that ended writes last, the graceful stop at a game end,
 * and a game that throws mid-match. The decks are burn mirrors, so a game
 * lasts a handful of turns.
 */
@ExtendWith(ForgeExtension.class)
class GamePlayerRecordingTest {

    private static final Pattern GAME_ID = Pattern.compile("\"game_id\":\"([^\"]+)\"");

    @TempDir
    Path dir;

    private static Deck burn() {
        Deck deck = new Deck();
        deck.getMain().add("Mountain", 16);
        deck.getMain().add("Lightning Bolt", 24);
        return deck;
    }

    private static CollectionCaps caps() {
        return TestCaps.caps().interventions(0).probes(0).build();
    }

    private RecordShardWriter writer() {
        return new RecordShardWriter(dir, "run", 0, "l1");
    }

    /** Every line of a closed shard, read through each gzip member: a cut one throws. */
    private static List<String> lines(Path shard) throws IOException {
        try (InputStream in = new GZIPInputStream(Files.newInputStream(shard))) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            List<String> lines = new ArrayList<>();
            for (String line : text.split("\n")) {
                if (!line.isEmpty()) {
                    lines.add(line);
                }
            }
            return lines;
        }
    }

    /** Each game's records, by game id, in the order the shard holds them. */
    private static Map<String, List<String>> byGame(List<String> lines) {
        Map<String, List<String>> games = new LinkedHashMap<>();
        for (String line : lines) {
            Matcher id = GAME_ID.matcher(line);
            assertTrue(id.find(), "every record names its game");
            games.computeIfAbsent(id.group(1), g -> new ArrayList<>()).add(line);
        }
        return games;
    }

    private static boolean isMarker(String line) {
        return line.contains("\"kind\":\"turn_action\"") && line.contains("\"subkind\":\"game_end\"");
    }

    /** Creates the stop file at the first turn of a game's third. */
    public static final class StopAtTurn {
        private final Game game;
        private final Path file;

        StopAtTurn(Game game, Path file) {
            this.game = game;
            this.file = file;
        }

        @Subscribe
        public void onTurn(GameEventTurnBegan event) throws IOException {
            if (game.getPhaseHandler().getTurn() == 3 && !Files.exists(file)) {
                Files.createFile(file);
            }
        }
    }

    @Test
    void eachGameThatEndedWritesItsMarkerLast() throws IOException {
        RecordShardWriter writer = writer();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(3, writer, caps(), WorkerStop.NEVER, Match::startGame));

        GamePlayer.PlayedMatch played = player.playMatch(burn(), burn());
        writer.close();

        Map<String, List<String>> games = byGame(lines(writer.path()));
        assertEquals(played.games().size(), games.size(), "one recorded game per game played");
        for (Map.Entry<String, List<String>> game : games.entrySet()) {
            List<String> records = game.getValue();
            assertTrue(isMarker(records.get(records.size() - 1)),
                    game.getKey() + "'s last record is its marker");
            assertEquals(1, records.stream().filter(GamePlayerRecordingTest::isMarker).count(),
                    game.getKey() + " has one marker");
            assertTrue(records.get(records.size() - 1).contains("\"records_withheld\":0"));
        }
    }

    /** Copies the shard's bytes as they stand on disk at the second turn of the game it watches. */
    public static final class SnapshotAtTurnTwo {
        private final Game game;
        private final Path shard;
        private final byte[][] taken;

        SnapshotAtTurnTwo(Game game, Path shard, byte[][] taken) {
            this.game = game;
            this.shard = shard;
            this.taken = taken;
        }

        @Subscribe
        public void onTurn(GameEventTurnBegan event) throws IOException {
            if (game.getPhaseHandler().getTurn() == 2 && taken[0] == null) {
                taken[0] = Files.exists(shard) ? Files.readAllBytes(shard) : new byte[0];
            }
        }
    }

    /** Every line in a shard's bytes as a hard stop leaves them: none is a shard never written. */
    private static List<String> lines(byte[] shard) throws IOException {
        if (shard.length == 0) {
            return List.of();
        }
        try (InputStream in = new GZIPInputStream(new java.io.ByteArrayInputStream(shard))) {
            List<String> lines = new ArrayList<>();
            for (String line : new String(in.readAllBytes(), StandardCharsets.UTF_8).split("\n")) {
                if (!line.isEmpty()) {
                    lines.add(line);
                }
            }
            return lines;
        }
    }

    /**
     * A hard stop (a kill, a crash) early in a match's second game, the
     * writer never closed: the shard as it stands then holds the first game
     * whole, its marker included, so a finished game never reads as cut; the
     * second game has nothing on disk, or a prefix without its marker.
     */
    @Test
    void aFinishedGameIsOnDiskWholeBeforeTheNextGameStarts() throws IOException {
        RecordShardWriter writer = writer();
        byte[][] taken = new byte[1][];
        List<Game> started = new ArrayList<>();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(3, writer, caps(), WorkerStop.NEVER, (match, game) -> {
            started.add(game);
            if (started.size() == 2) {
                game.subscribeToEvents(new SnapshotAtTurnTwo(game, writer.path(), taken));
            }
            match.startGame(game);
        }));

        player.playMatch(burn(), burn());
        writer.close();

        assertTrue(started.size() >= 2 && taken[0] != null, "the second game reached its turn 2");
        Map<String, List<String>> whole = byGame(lines(writer.path()));
        Map<String, List<String>> onDisk = byGame(lines(taken[0]));
        List<String> first = onDisk.get("run.0-l1.0");
        assertEquals(whole.get("run.0-l1.0"), first, "the first game is on disk whole");
        assertTrue(isMarker(first.get(first.size() - 1)), "with its marker last");
        List<String> second = onDisk.getOrDefault("run.0-l1.1", List.of());
        assertTrue(second.stream().noneMatch(GamePlayerRecordingTest::isMarker),
                "the second game, still playing, has no marker on disk");
        assertTrue(java.util.Set.of("run.0-l1.0", "run.0-l1.1").containsAll(onDisk.keySet()),
                "no other game: " + onDisk.keySet());
    }

    /**
     * A game past {@code --max-records-per-game} keeps its outcome record and
     * its marker, the marker counting what the cap withheld: the record
     * holding {@code player_won} / {@code player_lost} passes the cap.
     */
    @Test
    void aGameAtTheCapKeepsItsOutcomeAndItsMarker() throws IOException {
        RecordShardWriter writer = new RecordShardWriter(dir, "run", 0, "l1", 12);
        GamePlayer player = new GamePlayer(new GamePlayer.Config(3, writer, caps(), WorkerStop.NEVER, Match::startGame));

        player.playMatch(burn(), burn());
        writer.close();

        Map<String, List<String>> games = byGame(lines(writer.path()));
        assertFalse(games.isEmpty());
        for (Map.Entry<String, List<String>> game : games.entrySet()) {
            List<String> records = game.getValue();
            String marker = records.get(records.size() - 1);
            assertTrue(isMarker(marker), game.getKey());
            assertFalse(marker.contains("\"records_withheld\":0"), "the cap withheld some: " + marker);
            assertTrue(records.stream().anyMatch(r -> r.contains("\"type\":\"player_won\"")),
                    game.getKey() + " kept the record holding its outcome");
        }
    }

    @Test
    void aStopAskedMidGameEndsTheMatchAtThatGamesEnd() throws IOException {
        RecordShardWriter writer = writer();
        Path stopFile = dir.resolve("worker.stop");
        WorkerStop stop = new WorkerStop(stopFile, -1, System::nanoTime);
        List<Game> started = new ArrayList<>();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(3, writer, caps(), stop, (match, game) -> {
            started.add(game);
            game.subscribeToEvents(new StopAtTurn(game, stopFile));
            match.startGame(game);
        }));

        WorkerStop.Requested requested = assertThrows(
                WorkerStop.Requested.class, () -> player.playMatch(burn(), burn()));

        assertEquals("stop file " + stopFile, requested.reason());
        assertEquals(1, started.size(), "no game starts after the stop");
        assertEquals(1, player.gamesPlayed(), "the game it stopped after was played to its end");
        assertTrue(started.get(0).isGameOver(), "the game in progress ended normally");
        writer.close();
        Map<String, List<String>> games = byGame(lines(writer.path()));
        assertEquals(1, games.size());
        List<String> records = games.values().iterator().next();
        assertTrue(isMarker(records.get(records.size() - 1)), "its marker is its last record");
    }

    @Test
    void aStoppedWorkerClosesItsShardAndSaysSoWithoutAFailureLine() throws IOException {
        RecordShardWriter writer = writer();
        writer.write("{\"record_id\":\"r\",\"game_id\":\"run.0-l1.0\"}");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        WorkerLoop.Progress progress = new WorkerLoop.Progress(
                new PrintStream(out, true, StandardCharsets.UTF_8), 10, "matches generated",
                "Error generating match");

        ToolMain.Exit exit = MatchWorkerMain.stopGracefully(
                new WorkerLoop.Ending(4, "stop file x"), progress, 9, writer);

        assertEquals(ExitCode.OK, exit.code());
        assertEquals(1, lines(writer.path()).size(), "the block in flight is written");
        String line = out.toString(StandardCharsets.UTF_8).trim();
        assertEquals("Worker: stopped at a game end (stop file x); 4 matches generated in this JVM;"
                + " 9 games played; output closed, exiting 0", line);
        assertFalse(line.contains("Error generating match") || line.contains("giving up"));
    }

    /** A controller that throws once its game reaches its fourth turn. */
    private static final class ThrowsMidGame extends PlayerControllerAi {
        ThrowsMidGame(Game game, Player player) {
            super(game, player, player.getLobbyPlayer());
        }

        @Override
        public List<SpellAbility> chooseSpellAbilityToPlay() {
            if (getGame().getPhaseHandler().getTurn() >= 4) {
                throw new IllegalStateException("a game that throws mid-match");
            }
            return super.chooseSpellAbilityToPlay();
        }
    }

    @Test
    void aGameThatThrowsMidMatchWritesNoMarkerAndTheNextRecordsNormally() throws IOException {
        RecordShardWriter writer = writer();
        List<Game> started = new ArrayList<>();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(1, writer, caps(), WorkerStop.NEVER, (match, game) -> {
            started.add(game);
            if (started.size() == 1) {
                match.startGame(game, () -> game.getPlayers().get(0).dangerouslySetController(
                        new ThrowsMidGame(game, game.getPlayers().get(0))));
            } else {
                match.startGame(game);
            }
        }));

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class, () -> player.playMatch(burn(), burn()));
        assertEquals("a game that throws mid-match", thrown.getMessage());
        assertFalse(started.get(0).isGameOver());
        player.playMatch(burn(), burn());
        writer.close();

        Map<String, List<String>> games = byGame(lines(writer.path()));
        List<String> cut = games.get("run.0-l1.0");
        List<String> next = games.get("run.0-l1.1");
        assertTrue(cut != null && !cut.isEmpty(), "the cut game wrote records before it threw");
        assertTrue(cut.stream().noneMatch(GamePlayerRecordingTest::isMarker), "and no marker");
        assertTrue(isMarker(next.get(next.size() - 1)), "the next game records to its marker");
    }

    /**
     * An Error in a random-seat match is never caught and ends the worker;
     * the match is named a random-seat one on stderr first, in the worker
     * loop's own words.
     */
    @Test
    void aRandomSeatMatchEndedByAnErrorSaysItWasOne() {
        RecordShardWriter writer = writer();
        CollectionCaps randomSeat = TestCaps.caps().interventions(0).probes(0)
                .randomSeat(1.0, 0.5).build();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(1, writer, randomSeat, WorkerStop.NEVER,
                (match, game) -> {
                    throw new StackOverflowError("deep");
                }));
        java.io.PrintStream err = System.err;
        ByteArrayOutputStream printed = new ByteArrayOutputStream();
        System.setErr(new PrintStream(printed, true, StandardCharsets.UTF_8));
        try {
            assertThrows(StackOverflowError.class, () -> player.playMatch(burn(), burn()));
        } finally {
            System.setErr(err);
            writer.close();
        }

        assertTrue(printed.toString(StandardCharsets.UTF_8).contains(
                "Error generating match: [random-seat match] java.lang.StackOverflowError: deep"),
                printed.toString(StandardCharsets.UTF_8));
    }

    @Test
    void aRandomSeatMatchThatThrowsSaysItWasOne() {
        RecordShardWriter writer = writer();
        CollectionCaps randomSeat = TestCaps.caps().interventions(0).probes(0)
                .randomSeat(1.0, 0.5).build();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(1, writer, randomSeat, WorkerStop.NEVER,
                (match, game) -> {
                    throw new IllegalStateException("boom");
                }));

        GamePlayer.RandomSeatMatchFailed failed = assertThrows(
                GamePlayer.RandomSeatMatchFailed.class, () -> player.playMatch(burn(), burn()));

        assertEquals("[random-seat match] boom", failed.getMessage());
        writer.close();
    }
}
