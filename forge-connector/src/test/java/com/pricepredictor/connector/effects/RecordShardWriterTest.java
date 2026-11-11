package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The shard writer: the names and ids it issues, the file it appends to, and
 * record identity across worker lifetimes.
 */
class RecordShardWriterTest {

    @TempDir
    Path tempDir;

    // ── names and ids ───────────────────────────────────────────────────

    @Test
    void aShardIsNamedForItsRunWorkerAndLifetime() {
        try (RecordShardWriter writer =
                new RecordShardWriter(tempDir, "run-uuid", 4, "l1")) {
            assertEquals(
                    "run-uuid.4-l1.jsonl.gz", writer.path().getFileName().toString());
        }
    }

    @Test
    void recordIdsCarryTheWorkerIndexAndCountUp() {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 7, "l1")) {
            assertEquals("run.7-l1.0", writer.nextRecordId());
            assertEquals("run.7-l1.1", writer.nextRecordId());
        }
    }

    @Test
    void gameIdsCountSeparatelyFromRecordIds() {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            writer.nextRecordId();
            writer.nextRecordId();
            assertEquals("run.0-l1.0", writer.nextGameId());
        }
    }

    @Test
    void theDirectoryIsCreated() {
        Path nested = tempDir.resolve("output/effects/records");
        try (RecordShardWriter writer = new RecordShardWriter(nested, "run", 0, "l1")) {
            assertTrue(Files.isDirectory(nested));
            assertNotNull(writer.path());
        }
    }

    // ── the file ────────────────────────────────────────────────────────

    @Test
    void aShardIsAConcatenationOfCompleteGzipMembers() throws IOException {
        // Each writer contributes its own member, so appending is valid gzip
        // rather than a second stream glued onto the first. That is what lets a
        // killed worker truncate the last member and leave the rest readable.
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            writer.write("{\"a\":1}");
        }
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            writer.write("{\"a\":2}");
        }
        byte[] bytes = Files.readAllBytes(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        int members = 0;
        for (int i = 0; i + 1 < bytes.length; i++) {
            // gzip's magic number, which starts every member.
            if ((bytes[i] & 0xFF) == 0x1F && (bytes[i + 1] & 0xFF) == 0x8B) {
                members++;
            }
        }
        assertTrue(members >= 2, "expected one member per writer, saw " + members);

        // And the lifetime token must not split a single JVM's output, or the
        // append and gzip-member behaviour stops applying.
        assertEquals(1, shards().size());
        assertEquals(List.of("{\"a\":1}", "{\"a\":2}"), ShardRun.readShard(shards().get(0)));
    }

    @Test
    void aFullBlockIsFlushedWithoutClosing() throws IOException {
        // The worker loops until it is killed, so a shard that only wrote on
        // close would be empty for the whole run.
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        for (int i = 0; i < RecordShardWriter.BLOCK_RECORDS; i++) {
            writer.write("{\"a\":" + i + "}");
        }
        assertEquals(RecordShardWriter.BLOCK_RECORDS, ShardRun.readShard(writer.path()).size());
    }

    @Test
    void aWriterThatWritesNothingCreatesNoFile() {
        // Many collector tests read "no shard on disk" as "nothing was
        // recorded"; that reading holds only while the file is created by the
        // first record, not by the writer.
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        writer.close();

        assertFalse(Files.exists(writer.path()));
    }

    // ── identity across worker lifetimes ────────────────────────────────
    //
    // Both counters live in one JVM's heap, and the supervisor replaces that
    // JVM many times over a run, so without the lifetime token in the shard
    // name the record ids and game ids of successive JVMs collide.
    //
    // The restarts are not the problem and are not going away: the small heap and
    // the timed recycle are deliberate containment for Forge misbehaving late
    // in a JVM's life. The ids have to be correct under them, including for a
    // JVM killed mid-game, which runs no shutdown hook.

    @Test
    void aRestartedWorkerReusesNoId() {
        // The scenario every recycle produces: same run id, same pool slot,
        // same directory, a new JVM. Both counters start at zero again, so
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
        List<String> survived = ShardRun.readShard(killed.path());
        assertEquals(RecordShardWriter.BLOCK_RECORDS, survived.size());
        assertEquals(10, ShardRun.readShard(restarted.path()).size());

        // And the corpus as a whole holds no id twice.
        Set<String> all = new HashSet<>();
        for (Path shard : shards()) {
            for (String line : ShardRun.readShard(shard)) {
                assertTrue(all.add(line), "duplicate record across shards: " + line);
            }
        }
    }

    @Test
    void idsKeepThreeDotSegmentsSoParsesFromTheRightStillWork() {
        // The worker slot is record_id.split(".")[-2]; link_id appends
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

        // No dot and no hyphen, so an id still splits cleanly and the worker
        // index stays readable.
        assertTrue(one.matches("[0-9a-z]{1,16}"), one);
        assertTrue(two.matches("[0-9a-z]{1,16}"), two);
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

    // ── the per-game record cap (FR-041a) ───────────────────────────────

    /** A record line of {@code game} on {@code turn}, acting through {@code script}. */
    private static String fakeRecord(String id, String game, int turn, String script) {
        return "{\"record_id\":\"" + id + "\",\"generation\":3,\"game_id\":\"" + game
                + "\",\"kind\":\"resolution\",\"ability\":[{\"script_file\":\"" + script
                + "\",\"face\":0}],\"state\":{\"global\":{\"turn\":" + turn
                + ",\"phase\":\"main1\"},\"players\":[{\"this_turn\":{}}]}}";
    }

    /**
     * A game writing past the cap: its first records are written, every one
     * after is withheld, another game's are not touched, and when the game
     * ends one line names it, how many were withheld and the turn and acting
     * card most of them shared.
     */
    @Test
    void aGamePastTheCapIsWithheldAndNamed() throws Throwable {
        Path shard;
        String report;
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1", 3)) {
            shard = writer.path();
            writer.write(fakeRecord("r0", "g1", 4, "cardsfolder/h/hill_giant.txt"));
            for (int i = 1; i < 8; i++) {
                writer.write(fakeRecord("r" + i, "g1", 9, "cardsfolder/c/chain_of_smog.txt"));
            }
            writer.write(fakeRecord("r8", "g1", 9, "cardsfolder/f/forest.txt"));
            writer.write(fakeRecord("r9", "g2", 1, "cardsfolder/f/forest.txt"));
            report = com.pricepredictor.connector.StdStreams.captureStderr(() -> {
                writer.endGame("g1");
                writer.endGame("g2");
            });
        }

        List<String> written = ShardRun.readShard(shard);
        assertEquals(List.of("r0", "r1", "r2", "r9"), written.stream()
                .map(line -> (String) RecordLines.parse(line).get("record_id")).toList());
        assertEquals("Effect records: game g1 reached --max-records-per-game 3; withheld 6 more"
                        + " records, most often (5) on turn 9, acting card"
                        + " cardsfolder/c/chain_of_smog.txt",
                report.strip());
    }
}
