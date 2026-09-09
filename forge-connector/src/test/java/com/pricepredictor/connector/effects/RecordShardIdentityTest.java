package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Record identity across worker lifetimes.
 *
 * <p>Separate from {@code CollectorTest}'s writer group because this is a
 * regression suite for one defect rather than a description of the writer: the
 * first collected corpus reused 71-75 % of its record ids and held 804 distinct
 * game ids for 31,662 games, because both counters live in one JVM's heap and
 * the supervisor replaces that JVM roughly 530 times over an eight-hour run.
 * Every test here fails against the pre-lifetime writer.
 *
 * <p>The restarts are not the bug and are not going away: the small heap and the
 * timed recycle are deliberate containment for Forge misbehaving late in a JVM's
 * life. The ids have to be correct under them, including for a JVM killed
 * mid-game, which runs no shutdown hook.
 */
class RecordShardIdentityTest {

    @TempDir
    Path tempDir;

    @Test
    void aShardIsNamedForItsRunWorkerAndLifetime() {
        try (RecordShardWriter writer =
                new RecordShardWriter(tempDir, "run-uuid", 4, "l1")) {
            assertEquals(
                    "run-uuid.4-l1.jsonl.gz", writer.path().getFileName().toString());
        }
    }

    @Test
    void aRestartedWorkerReusesNoId() {
        // The scenario the analysed run hit ~530 times: same run id, same pool
        // slot, same directory, a new JVM. Both counters start at zero again, so
        // only the shard token can keep the ids apart. Record ids and game ids
        // are checked separately because they are separate namespaces — the two
        // counters are independent and a game id may read like a record id.
        Set<String> firstRecords = new HashSet<>();
        Set<String> firstGames = new HashSet<>();
        Set<String> secondRecords = new HashSet<>();
        Set<String> secondGames = new HashSet<>();
        Path firstPath;
        Path secondPath;
        try (RecordShardWriter writer =
                new RecordShardWriter(tempDir, "run", 3, "l1")) {
            firstPath = writer.path();
            firstGames.add(writer.nextGameId());
            firstGames.add(writer.nextGameId());
            firstRecords.add(writer.nextRecordId());
            firstRecords.add(writer.nextRecordId());
            firstRecords.add(writer.nextRecordId());
        }
        try (RecordShardWriter writer =
                new RecordShardWriter(tempDir, "run", 3, "l2")) {
            secondPath = writer.path();
            secondGames.add(writer.nextGameId());
            secondGames.add(writer.nextGameId());
            secondRecords.add(writer.nextRecordId());
            secondRecords.add(writer.nextRecordId());
            secondRecords.add(writer.nextRecordId());
        }
        assertNotEquals(firstPath, secondPath);
        assertEquals(3, firstRecords.size());
        assertEquals(3, secondRecords.size());
        assertEquals(2, firstGames.size());
        assertEquals(2, secondGames.size());
        assertTrue(java.util.Collections.disjoint(firstRecords, secondRecords),
                "a restarted worker reissued record ids: "
                        + firstRecords + " vs " + secondRecords);
        assertTrue(java.util.Collections.disjoint(firstGames, secondGames),
                "a restarted worker reissued game ids: "
                        + firstGames + " vs " + secondGames);
    }

    @Test
    void aWorkerKilledMidGameLeavesTheNextLifetimeAWholeShardOfItsOwn() throws IOException {
        // taskkill /F runs no shutdown hook, so the killed JVM's last partial
        // block is simply never written. What must not happen is the next JVM
        // continuing that file and recounting from zero into it.
        RecordShardWriter killed = new RecordShardWriter(tempDir, "run", 0, "l1");
        List<String> killedIds = new ArrayList<>();
        for (int i = 0; i < RecordShardWriter.BLOCK_RECORDS + 7; i++) {
            String id = killed.nextRecordId();
            killedIds.add(id);
            killed.write("{\"record_id\":\"" + id + "\"}");
        }
        // No close(): the process was killed mid-game.

        RecordShardWriter restarted = new RecordShardWriter(tempDir, "run", 0, "l2");
        List<String> restartedIds = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            String id = restarted.nextRecordId();
            restartedIds.add(id);
            restarted.write("{\"record_id\":\"" + id + "\"}");
        }
        restarted.close();

        assertNotEquals(killed.path(), restarted.path());
        assertTrue(java.util.Collections.disjoint(
                        new HashSet<>(killedIds), new HashSet<>(restartedIds)),
                "the restarted lifetime reissued the killed one's ids");

        // The killed lifetime keeps every complete block; only the tail is lost.
        List<String> survived = readShard(killed.path());
        assertEquals(RecordShardWriter.BLOCK_RECORDS, survived.size());
        assertEquals(10, readShard(restarted.path()).size());

        // And the corpus as a whole holds no id twice.
        Set<String> all = new HashSet<>();
        for (Path shard : shards()) {
            for (String line : readShard(shard)) {
                assertTrue(all.add(line), "duplicate record across shards: " + line);
            }
        }
    }

    @Test
    void oneLifetimeKeepsOneShardAndKeepsCounting() throws IOException {
        // The other half of the guarantee: the token must not split a single
        // JVM's output, or the append and gzip-member behaviour stops applying.
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            writer.write("{\"a\":1}");
        }
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            writer.write("{\"a\":2}");
        }
        assertEquals(1, shards().size());
        assertEquals(List.of("{\"a\":1}", "{\"a\":2}"), readShard(shards().get(0)));
    }

    @Test
    void idsKeepThreeDotSegmentsSoParsesFromTheRightStillWork() {
        // EffectRecord.worker is record_id.split(".")[-2]; link_id appends
        // ".link.{n}" to a game id. The token rides inside the middle segment
        // precisely so neither has to change.
        try (RecordShardWriter writer =
                new RecordShardWriter(tempDir, "3f2a-uuid", 7, "mk3p9x2q")) {
            String recordId = writer.nextRecordId();
            assertEquals("3f2a-uuid.7-mk3p9x2q.0", recordId);
            assertEquals("3f2a-uuid.7-mk3p9x2q.0", writer.nextGameId());

            String[] segments = recordId.split("\\.");
            assertEquals(3, segments.length);
            assertEquals("7-mk3p9x2q", segments[segments.length - 2]);
            assertEquals("7", segments[segments.length - 2].split("-")[0]);
        }
    }

    @Test
    void aLifetimeTokenIsUniqueAndSafeToEmbed() {
        String one = RecordShardWriter.mintLifetime();
        String two = RecordShardWriter.mintLifetime();
        assertNotEquals(one, two);
        assertTrue(RecordShardWriter.isValidLifetime(one), one);
        assertTrue(RecordShardWriter.isValidLifetime(two), two);

        // A dot would add a segment and a hyphen would hide the worker index,
        // so both are rejected along with anything a filename cannot hold.
        assertFalse(RecordShardWriter.isValidLifetime(null));
        assertFalse(RecordShardWriter.isValidLifetime(""));
        assertFalse(RecordShardWriter.isValidLifetime("a.b"));
        assertFalse(RecordShardWriter.isValidLifetime("a-b"));
        assertFalse(RecordShardWriter.isValidLifetime("UPPER"));
        assertFalse(RecordShardWriter.isValidLifetime("01234567890123456"));
    }

    @Test
    void mintedTokensStayDistinctUnderABurstOfRestarts() {
        // Six slots recycling every 60 s for eight hours is ~530 mints, and two
        // of them can land in the same millisecond. The clock alone would not
        // separate those.
        Set<String> tokens = new HashSet<>();
        for (int i = 0; i < 2000; i++) {
            tokens.add(RecordShardWriter.mintLifetime());
        }
        assertTrue(tokens.size() > 1990,
                "minted tokens collided " + (2000 - tokens.size()) + " times");
    }

    /** Every shard in the temp directory, sorted by name as a reader would list them. */
    private List<Path> shards() throws IOException {
        try (var listing = Files.list(tempDir)) {
            return listing
                    .filter(p -> p.getFileName().toString().endsWith(RecordShardWriter.SUFFIX))
                    .sorted()
                    .toList();
        }
    }

    /** The shard's lines, decompressed across every complete gzip member. */
    private static List<String> readShard(Path path) throws IOException {
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }
}
