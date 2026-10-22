package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.spellability.SpellAbility;
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
 * One game's collectors writing one shard, and the records they wrote.
 *
 * <p>Wired as a worker wires them: the patched collectors file their clause
 * and outcome events through the bracket, and a run with the hooks installed
 * records through {@link GameRecording} itself. The shard is closed before it
 * is read, so every record the body caused is on disk.
 *
 * <p>The writer creates its file only when it writes a record, so a run that
 * wrote nothing leaves no shard, and reading one fails. A test that expects
 * nothing says so with {@link #shardOf} and asserts the file is absent.
 */
final class ShardRun {

    private static final String RUN = "run";
    private static final String LIFETIME = "l1";
    private static final String GAME_ID = RUN + ".0-" + LIFETIME + ".0";
    /** The sampler's seed: what a worker's first game would draw from. */
    private static final long SEED = 1L;

    private ShardRun() {
    }

    /** What a test does with the two collectors. */
    @FunctionalInterface
    interface Body {
        void run(BusBracketCollector bracket, PatchedCollectors collectors) throws Throwable;
    }

    /** What a test does with a bracket the game's bus is feeding. */
    @FunctionalInterface
    interface Observed {
        void run(BusBracketCollector bracket) throws Throwable;
    }

    /** What a test plays while a game is recorded the way a worker records it. */
    @FunctionalInterface
    interface Played {
        void run() throws Throwable;
    }

    /** A forced resolution: whether it wrote a record, and the shard it wrote to. */
    record Fork(boolean wrote, Path shard) {

        List<String> records() throws IOException {
            return readShard(shard);
        }
    }

    // ── driven by hand ──────────────────────────────────────────────────

    /**
     * Run {@code body} with neither collector on the game's bus nor any hook
     * installed: the body drives them by hand.
     */
    static List<String> records(Path dir, Game game, Body body) throws Throwable {
        return records(dir, game, CollectionCaps.defaults(), body);
    }

    /** The same, under caps of the test's choosing. */
    static List<String> records(Path dir, Game game, CollectionCaps caps, Body body)
            throws Throwable {
        return readShard(shardOf(dir, game, caps, body));
    }

    /** The same, returning the shard unread, for a test about whether there is one. */
    static Path shardOf(Path dir, Game game, Body body) throws Throwable {
        return shardOf(dir, game, CollectionCaps.defaults(), body);
    }

    static Path shardOf(Path dir, Game game, CollectionCaps caps, Body body) throws Throwable {
        return shardOf(dir, game, caps, null, body);
    }

    /** The same, for a match whose random seat is {@code randomSeatPlayerId}. */
    static List<String> recordsWithRandomSeat(
            Path dir, Game game, String randomSeatPlayerId, Body body) throws Throwable {
        return readShard(shardOf(dir, game, CollectionCaps.defaults(), randomSeatPlayerId, body));
    }

    private static Path shardOf(
            Path dir, Game game, CollectionCaps caps, String randomSeatPlayerId, Body body)
            throws Throwable {
        RecordShardWriter writer = new RecordShardWriter(dir, RUN, 0, LIFETIME);
        try {
            BusBracketCollector bracket = new BusBracketCollector(
                    game, TestCollectors.context(writer, GAME_ID, randomSeatPlayerId),
                    recordId -> { });
            try (PatchedCollectors collectors = new PatchedCollectors(
                    game, TestCollectors.context(writer, GAME_ID, randomSeatPlayerId),
                    caps, SEED, bracket)) {
                body.run(bracket, collectors);
            }
        } finally {
            writer.close();
        }
        return writer.path();
    }

    /**
     * Play {@code body} with the game recorded as a worker records it: the
     * bracket on the game's bus and every hook the checkout offers installed;
     * skipped on a checkout without the clause hook.
     *
     * <p>No fork budget, so the phase boundaries the body crosses copy nothing.
     *
     * @param why what the test needs the clause hook for, shown when it skips
     */
    static List<String> recordsWithClauseHook(Path dir, Game game, String why, Played body)
            throws Throwable {
        assumeClauseHook(why);
        CollectionCaps defaults = CollectionCaps.defaults();
        CollectionCaps noForks = new CollectionCaps(
                defaults.manaCap(), defaults.playabilityRate(), 0, 0, List.of(),
                defaults.randomSeatShare(), defaults.randomSeatProbability());
        RecordShardWriter writer = new RecordShardWriter(dir, RUN, 0, LIFETIME);
        try (GameRecording recording =
                new GameRecording(game, writer, GAME_ID, noForks, SEED, null)) {
            recording.open();
            body.run();
        } finally {
            writer.close();
        }
        return readShard(writer.path());
    }

    // ── observed off the bus ────────────────────────────────────────────

    /**
     * Run {@code body} with the bracket subscribed to the game's bus and no
     * patched collector at all: what a worker on a stock checkout records.
     * Nothing skips it.
     */
    static List<String> observed(Path dir, Game game, Observed body) throws Throwable {
        RecordShardWriter writer = new RecordShardWriter(dir, RUN, 0, LIFETIME);
        try {
            BusBracketCollector bracket = TestCollectors.bracket(game, writer, GAME_ID);
            game.subscribeToEvents(bracket);
            body.run(bracket);
        } finally {
            writer.close();
        }
        return readShard(writer.path());
    }

    // ── forced resolutions ──────────────────────────────────────────────

    /** Fork {@code game} once and force {@code ability} to resolve, with no hook installed. */
    static Fork forked(Path dir, Game game, CollectionCaps caps, long seed, SpellAbility ability) {
        RecordShardWriter writer = new RecordShardWriter(dir, RUN, 0, LIFETIME);
        boolean wrote;
        try {
            wrote = new ForkCollector(game, TestCollectors.context(writer, GAME_ID), caps, seed)
                    .intervene(ability, 1);
        } finally {
            writer.close();
        }
        return new Fork(wrote, writer.path());
    }

    /**
     * The same with every hook the checkout offers installed, so the fork's
     * clauses reach the collectors as a live game's do; skipped on a checkout
     * without the clause hook.
     *
     * @param why what the test needs the clause hook for, shown when it skips
     */
    static Fork forkedWithClauseHook(
            Path dir, Game game, CollectionCaps caps, long seed, SpellAbility ability,
            String why) {
        assumeClauseHook(why);
        RecordShardWriter writer = new RecordShardWriter(dir, RUN, 0, LIFETIME);
        boolean wrote;
        try (PatchedCollectors collectors = new PatchedCollectors(
                game, TestCollectors.context(writer, GAME_ID), caps, SEED,
                TestCollectors.bracket(game, writer, GAME_ID))) {
            collectors.install();
            wrote = new ForkCollector(game, TestCollectors.context(writer, GAME_ID), caps, seed)
                    .intervene(ability, 1);
        } finally {
            writer.close();
        }
        return new Fork(wrote, writer.path());
    }

    private static void assumeClauseHook(String why) {
        Assumptions.assumeTrue(
                AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.CLAUSE_LISTENER.present(),
                "../forge has no clause hook" + (why.isEmpty() ? "" : " " + why));
    }

    // ── reading ─────────────────────────────────────────────────────────

    /**
     * Every record line of a shard, across every complete gzip member.
     *
     * @throws java.nio.file.NoSuchFileException when nothing was ever written to it
     */
    static List<String> readShard(Path path) throws IOException {
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }
}
