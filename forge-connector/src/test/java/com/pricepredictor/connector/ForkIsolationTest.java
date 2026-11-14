package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.TestCaps;
import com.pricepredictor.connector.effects.directed.rules.ReplacedSacrificeCost;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousDamage;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A fork never touches the live game: the same seeded match played with
 * every fork off and with all four kinds on (interventions, probes,
 * protection and directed forks) writes the same live records, byte for
 * byte. A fork that drew from the live game's random stream, took one of its
 * timestamps or ids, or changed its board shows here as a live record that
 * differs, from the first record it touched on.
 */
@ExtendWith(ForgeExtension.class)
class ForkIsolationTest {

    private static final long START = 1_700_000_000_000L;

    /** When a record was written, the one member a replay cannot repeat. */
    private static final java.util.regex.Pattern WRITTEN_AT =
            java.util.regex.Pattern.compile("\"timestamp\":\"[^\"]*\",");

    @TempDir
    Path dir;

    /** Every fork record the matches wrote. */
    private final List<String> forkRecords = new ArrayList<>();

    private static Deck deck(Object... cardsAndCounts) {
        Deck deck = new Deck();
        for (int i = 0; i < cardsAndCounts.length; i += 2) {
            deck.getMain().add((String) cardsAndCounts[i], (Integer) cardsAndCounts[i + 1]);
        }
        return deck;
    }

    private static Deck burn() {
        return deck("Mountain", 16, "Lightning Bolt", 24);
    }

    /**
     * Creatures, first and double strikers, burn, tokens, an aura, a fog, a
     * sacrifice outlet: what forks reach, a directed fork forcing a play
     * (118.11, a Siege-Gang Commander made to sacrifice a Goblin) among them.
     */
    private static Deck boros() {
        return deck("Mountain", 9, "Plains", 8, "Pacifism", 2, "Serra Angel", 2, "Arc Trail", 2,
                "Siege-Gang Commander", 2, "Youthful Knight", 2, "Lightning Bolt", 2, "Boros Charm", 2, "Fireball", 1,
                "Shock", 1, "Raise the Alarm", 1, "Prodigal Pyromancer", 1, "Holy Day", 1,
                "Fencing Ace", 2, "Mogg Fanatic", 1, "Hill Giant", 1);
    }

    private static CollectionCaps off() {
        return TestCaps.caps().interventions(0).probes(0).build();
    }

    private static CollectionCaps allForks() {
        return TestCaps.caps().interventions(2).probes(4).probeKeywords("first_strike", "double_strike")
                .directed(2, 1.0, SimultaneousDamage.RULE, ReplacedSacrificeCost.RULE).build();
    }

    /** The live records one seeded match writes, in order. */
    private List<String> liveRecords(String name, Deck a, Deck b, long seed, CollectionCaps caps)
            throws IOException {
        Path out = Files.createDirectory(dir.resolve(name));
        RecordShardWriter writer = new RecordShardWriter(out, "run", 0, "l1");
        Random was = MyRandom.getRandom();
        MyRandom.setRandom(new Random(seed));
        try {
            new GamePlayer(1, writer, caps, WorkerStop.NEVER, Match::startGame, () -> START + seed)
                    .playMatch(a, b);
        } finally {
            MyRandom.setRandom(was);
            writer.close();
        }
        List<String> written = new ArrayList<>();
        int forked = 0;
        try (Stream<Path> files = Files.walk(out)) {
            for (Path shard : files.filter(p -> p.toString().endsWith(".jsonl.gz")).sorted().toList()) {
                for (String line : lines(shard)) {
                    if (line.contains("\"fork\":true")) {
                        forked++;
                        forkRecords.add(line);
                    } else {
                        written.add(WRITTEN_AT.matcher(line).replaceFirst(""));
                    }
                }
            }
        }
        return renumbered(written);
    }

    /**
     * The live records with each record id that names one of them (its own,
     * a link's, a mirror's) replaced by the record's place among them: a
     * fork's records take ids from the same sequence, which is no change to
     * the live game.
     */
    private static List<String> renumbered(List<String> written) {
        java.util.Map<String, String> place = new java.util.HashMap<>();
        for (String line : written) {
            java.util.regex.Matcher own = RECORD_ID.matcher(line);
            assertTrue(own.find(), "every record has an id");
            place.put(own.group(1), "#" + place.size());
        }
        List<String> live = new ArrayList<>();
        for (String line : written) {
            java.util.regex.Matcher id = ID_MEMBER.matcher(line);
            StringBuilder out = new StringBuilder();
            while (id.find()) {
                String to = id.group(1).equals("game_id") ? id.group(2)
                        : place.getOrDefault(id.group(2), "fork:" + id.group(2));
                id.appendReplacement(out, java.util.regex.Matcher.quoteReplacement(
                        "\"" + id.group(1) + "\":\"" + to + "\""));
            }
            id.appendTail(out);
            live.add(out.toString());
        }
        // Link ids too: a fork's cost and effect halves take them from the same sequence.
        java.util.Map<String, String> links = new java.util.HashMap<>();
        List<String> relinked = new ArrayList<>();
        for (String line : live) {
            java.util.regex.Matcher link = LINK_ID.matcher(line);
            StringBuilder out = new StringBuilder();
            while (link.find()) {
                String to = links.computeIfAbsent(link.group(), l -> "link#" + links.size());
                link.appendReplacement(out, java.util.regex.Matcher.quoteReplacement(to));
            }
            link.appendTail(out);
            relinked.add(out.toString());
        }
        return relinked;
    }

    private static final java.util.regex.Pattern LINK_ID =
            java.util.regex.Pattern.compile("run\\.0-l1\\.\\d+\\.link\\.\\d+");

    private static final java.util.regex.Pattern RECORD_ID =
            java.util.regex.Pattern.compile("\"record_id\":\"([^\"]+)\"");

    /** A member whose value is an id of this run's sequence. */
    private static final java.util.regex.Pattern ID_MEMBER =
            java.util.regex.Pattern.compile("\"(\\w+)\":\"(run\\.0-l1\\.\\d+)\"");

    private static List<String> lines(Path shard) throws IOException {
        try (InputStream in = new GZIPInputStream(Files.newInputStream(shard))) {
            List<String> lines = new ArrayList<>();
            for (String line : new String(in.readAllBytes(), StandardCharsets.UTF_8).split("\n")) {
                if (!line.isEmpty()) {
                    lines.add(line);
                }
            }
            return lines;
        }
    }

    private void assertForksLeaveTheLiveGame(String name, Deck a, Deck b, long seed) throws IOException {
        List<String> off = liveRecords(name + "-off", a, b, seed, off());
        List<String> on = liveRecords(name + "-on", a, b, seed, allForks());
        assertTrue(off.size() > 10, name + ": the match was played");
        for (int i = 0; i < Math.min(off.size(), on.size()); i++) {
            assertEquals(off.get(i), on.get(i), name + ": live record #" + i);
        }
        assertEquals(off.size(), on.size(), name + ": live records");
    }

    @Test
    void forksLeaveABurnMirrorsLiveRecordsAsTheyWere() throws IOException {
        assertForksLeaveTheLiveGame("burn", burn(), burn(), 11);
    }

    @Test
    void forksLeaveAMixedMatchsLiveRecordsAsTheyWere() throws IOException {
        for (long seed : new long[] {11, 12, 13, 14, 15, 16}) {
            assertForksLeaveTheLiveGame("mixed" + seed, boros(), boros(), seed);
        }
        // The matches reach every kind of fork: forced resolutions and protection
        // forks (interventional), probes (a mirror) and directed forks.
        assertTrue(forkRecords.stream().anyMatch(r -> r.contains("\"interventional\":true")), "interventions");
        assertTrue(forkRecords.stream().anyMatch(r -> r.contains("\"interventional\":false")
                && !r.contains("\"mirror_of\":null")), "probes");
        assertTrue(forkRecords.stream().anyMatch(r -> r.contains("\"directed\":{")), "directed forks");
        assertTrue(forkRecords.stream().anyMatch(r -> r.contains("\"aimed_at\":[\"118.11\"]")),
                "a directed fork whose variant forces a play");
        assertTrue(forkRecords.stream().anyMatch(r -> r.contains("\"interventional\":true")
                && !r.contains("\"kind\":\"resolution\"")), "protection forks, which write every kind");
    }
}
