package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.ShardId;
import com.pricepredictor.connector.effects.TestCaps;
import forge.deck.Deck;
import forge.game.Match;
import forge.util.MyRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The game's outcome in a real match, played through {@link
 * GamePlayer#playMatch} as a worker plays it (finding 9.5): a random seat
 * that never passes while a non-mana ability is playable pays Mischievous
 * Poltergeist's "Pay 1 life" until it reaches 0 life, with the ability still
 * on the stack, as the pilot's seat did until decision D12 stopped the seat
 * drawing a payment that leaves it at 0 life, so these games now end in
 * combat. Each game still writes its loss and its win once and its marker
 * last. The match draws from a seeded Forge random stream, restored after, so
 * both games are the same whatever ran before in the JVM and each ends by the
 * life payment.
 */
@ExtendWith(ForgeExtension.class)
class GameOutcomeRecordingTest {

    private static final Pattern GAME_ID = Pattern.compile("\"game_id\":\"([^\"]+)\"");
    private static final Pattern EVENT_TYPE = Pattern.compile("\"type\":\"([a-z_]+)\"");

    /** Forge's global random stream for the match: the same two games whatever ran before in the JVM. */
    private static final long MATCH_SEED = 1L;

    @TempDir
    Path dir;

    @Test
    void aSeatPayingLifeToZeroStillWritesTheOutcomeOnceAndTheMarkerLast() throws IOException {
        RecordShardWriter writer = new RecordShardWriter(dir, new ShardId("run", 0, "l1"));
        CollectionCaps randomSeat = TestCaps.caps().interventions(0).probes(0)
                .randomSeat(1.0, 1.0).build();
        GamePlayer player = new GamePlayer(new GamePlayer.Config(
                2, writer, randomSeat, WorkerStop.NEVER, Match::startGame, () -> 1L));

        Random was = MyRandom.getRandom();
        MyRandom.setRandom(new Random(MATCH_SEED));
        try {
            player.playMatch(poltergeists(), poltergeists());
        } finally {
            MyRandom.setRandom(was);
        }
        writer.close();

        Map<String, List<String>> games = byGame(writer.path());
        assertFalse(games.isEmpty());
        for (Map.Entry<String, List<String>> game : games.entrySet()) {
            List<String> records = game.getValue();
            List<String> outcome = records.stream()
                    .filter(record -> record.contains("\"type\":\"player_lost\"")).toList();
            assertEquals(1, outcome.size(), game.getKey() + " writes its loss once");
            assertTrue(eventTypes(outcome.get(0)).containsAll(List.of("player_lost", "player_won")),
                    game.getKey() + "'s game-end record holds the outcome: " + outcome.get(0));
            assertEquals(1, records.stream()
                    .filter(record -> record.contains("\"type\":\"player_won\"")).count(),
                    game.getKey() + " writes its win once");
            assertTrue(isMarker(records.get(records.size() - 1)),
                    game.getKey() + "'s last record is its marker");
        }
    }

    /** Swamps and Mischievous Poltergeists: "Pay 1 life: Regenerate", playable at any life. */
    private static Deck poltergeists() {
        Deck deck = new Deck();
        deck.getMain().add("Swamp", 17);
        deck.getMain().add("Mischievous Poltergeist", 23);
        return deck;
    }

    /** Each game's records, by game id, in the order the closed shard holds them. */
    private static Map<String, List<String>> byGame(Path shard) throws IOException {
        Map<String, List<String>> games = new LinkedHashMap<>();
        try (InputStream in = new GZIPInputStream(Files.newInputStream(shard))) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            for (String line : Arrays.stream(text.split("\n")).filter(l -> !l.isEmpty()).toList()) {
                Matcher id = GAME_ID.matcher(line);
                assertTrue(id.find(), "every record names its game");
                games.computeIfAbsent(id.group(1), g -> new java.util.ArrayList<>()).add(line);
            }
        }
        return games;
    }

    /** The types of a record's events, in order: the payload's, after its board. */
    private static List<String> eventTypes(String record) {
        String payload = record.substring(record.indexOf("\"payload\":"));
        return EVENT_TYPE.matcher(payload).results().map(m -> m.group(1)).toList();
    }

    private static boolean isMarker(String line) {
        return line.contains("\"kind\":\"turn_action\"") && line.contains("\"subkind\":\"game_end\"");
    }
}
