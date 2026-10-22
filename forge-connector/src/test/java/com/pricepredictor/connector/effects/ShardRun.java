package com.pricepredictor.connector.effects;

import forge.game.Game;
import org.junit.jupiter.api.Assumptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * One game's bracket and patched collectors writing one shard, and the records
 * they wrote.
 *
 * <p>The patched collectors file their clause and outcome events through the
 * bracket, as a worker wires them. The shard is closed before it is read, so
 * every record the body caused is on disk.
 */
final class ShardRun {

    private ShardRun() {
    }

    /** What a test does with the two collectors. */
    @FunctionalInterface
    interface Body {
        void run(BusBracketCollector bracket, PatchedCollectors collectors) throws Throwable;
    }

    /**
     * Run {@code body} with neither collector on the game's bus nor any hook
     * installed: the body drives them by hand.
     */
    static List<String> records(Path dir, Game game, Body body) throws Throwable {
        return run(dir, game, body, null);
    }

    /**
     * Run {@code body} with the bracket on the game's bus and every hook the
     * checkout offers installed, as a game in progress has them; skipped on a
     * checkout without the clause hook.
     *
     * @param why what the test needs the clause hook for, shown when it skips
     */
    static List<String> recordsWithClauseHook(Path dir, Game game, String why, Body body)
            throws Throwable {
        return run(dir, game, body, why);
    }

    private static List<String> run(Path dir, Game game, Body body, String hookNeededFor)
            throws Throwable {
        RecordShardWriter writer = new RecordShardWriter(dir, "run", 0, "l1");
        try {
            BusBracketCollector bracket = new BusBracketCollector(game, writer, "run.0-l1.0");
            try (PatchedCollectors collectors = new PatchedCollectors(
                    game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
                if (hookNeededFor != null) {
                    game.subscribeToEvents(bracket);
                    Assumptions.assumeTrue(
                            collectors.install() > 0
                                    && PatchHooks.CLAUSE_LISTENER.present(),
                            "../forge has no clause hook"
                                    + (hookNeededFor.isEmpty() ? "" : " " + hookNeededFor));
                }
                collectors.withBracket(bracket);
                body.run(bracket, collectors);
            }
        } finally {
            writer.close();
        }
        return readShard(writer.path());
    }

    /** Every record line of a shard, or none when nothing was written. */
    static List<String> readShard(Path path) throws IOException {
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }
}
