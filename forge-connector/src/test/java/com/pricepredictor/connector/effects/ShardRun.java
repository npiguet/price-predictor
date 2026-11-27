package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.player.Player;
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
    /** The id of the first game a test's shard writer records. */
    static final String GAME_ID = RUN + ".0-" + LIFETIME + ".0";
    /** The sampler's seed: what a worker's first game would draw from. */
    private static final long SEED = 1L;

    private ShardRun() {
    }

    /** A writer for a test's shard, under the run, worker and lifetime {@link #GAME_ID} names. */
    static RecordShardWriter writer(Path dir) {
        return new RecordShardWriter(dir, new ShardId(RUN, 0, LIFETIME));
    }

    /**
     * The writer's shard, closed so every record written so far is on disk.
     * The writer creates the file with its first record, so a collector that
     * wrote nothing leaves none.
     */
    static Path closed(RecordShardWriter writer) {
        writer.close();
        return writer.path();
    }

    /** What a test does with the two collectors. */
    @FunctionalInterface
    interface Body {
        void run(BusBracketCollector bracket, Channels collectors) throws Throwable;
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

    /** The same, reading the recording as it plays: what reached no record so far. */
    @FunctionalInterface
    interface Recorded {
        void run(GameRecording recording) throws Throwable;
    }

    /** A game recorded under the hooks, the caps it is recorded under, and why a test needs the hooks. */
    private record Hooked(Game game, CollectionCaps caps, String why) {
    }

    /** A game whose two collectors a test drives by hand, and the match's random seat, if any. */
    private record ByHand(Game game, CollectionCaps caps, String randomSeatPlayerId) {
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
        return records(dir, game, TestCaps.unprobed(), body);
    }

    /** The same, under caps of the test's choosing. */
    static List<String> records(Path dir, Game game, CollectionCaps caps, Body body)
            throws Throwable {
        return readShard(shardOf(dir, game, caps, body));
    }

    /** The same, returning the shard unread, for a test about whether there is one. */
    static Path shardOf(Path dir, Game game, Body body) throws Throwable {
        return shardOf(dir, game, TestCaps.unprobed(), body);
    }

    static Path shardOf(Path dir, Game game, CollectionCaps caps, Body body) throws Throwable {
        return shardOf(dir, new ByHand(game, caps, null), body);
    }

    /** The same, for a match whose random seat is {@code randomSeatPlayerId}. */
    static List<String> recordsWithRandomSeat(
            Path dir, Game game, String randomSeatPlayerId, Body body) throws Throwable {
        return records(dir, game, TestCaps.unprobed(), randomSeatPlayerId, body);
    }

    /** The same, under caps of the test's choosing. */
    static List<String> records(
            Path dir, Game game, CollectionCaps caps, String randomSeatPlayerId, Body body)
            throws Throwable {
        return readShard(shardOf(dir, new ByHand(game, caps, randomSeatPlayerId), body));
    }

    private static Path shardOf(Path dir, ByHand run, Body body) throws Throwable {
        RecordShardWriter writer = writer(dir);
        Game game = run.game();
        try {
            BusBracketCollector bracket = new BusBracketCollector(
                    game, TestCollectors.context(writer, GAME_ID, run.randomSeatPlayerId()),
                    recordId -> { });
            try (Channels collectors = Channels.of(game, new Channels.Setup(
                    TestCollectors.context(writer, GAME_ID, run.randomSeatPlayerId()),
                    run.caps(), SEED, bracket))) {
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
        return hooked(dir, game, why, recording -> body.run());
    }

    /**
     * The same, the body handed the game's recording: the hooked form of
     * {@link #observed}, for a test that reads what the bracket counted as
     * reaching no record ({@link GameRecording#droppedEvents}) as it plays.
     * The game ends as a worker's does, with the bus's game-finished event.
     *
     * @param why what the test needs the clause hook for, shown when it skips
     */
    static List<String> hooked(Path dir, Game game, String why, Recorded body)
            throws Throwable {
        CollectionCaps noForks = TestCaps.caps().interventions(0).probes(0).build();
        return readShard(recorded(dir, new Hooked(game, noForks, why), body).path());
    }

    /**
     * The same under caps of the test's choosing, returning the closed writer
     * unread, for a test about the shard or the files beside it.
     */
    static RecordShardWriter recordedWithClauseHook(
            Path dir, Game game, CollectionCaps caps, String why, Played body) throws Throwable {
        return recorded(dir, new Hooked(game, caps, why), recording -> body.run());
    }

    private static RecordShardWriter recorded(Path dir, Hooked run, Recorded body)
            throws Throwable {
        assumeClauseHook(run.why());
        RecordShardWriter writer = writer(dir);
        try (GameRecording recording =
                new GameRecording(
                        GameRecordingRequest.of(run.game(), writer, GAME_ID, run.caps())
                                .seeded(SEED))) {
            recording.open();
            body.run(recording);
        } finally {
            writer.close();
        }
        return writer;
    }

    // ── observed off the bus ────────────────────────────────────────────

    /**
     * Run {@code body} with the bracket subscribed to the game's bus and no
     * patched collector at all: what a worker on a stock checkout records.
     * Nothing skips it. A worker on the patched checkout records as {@link
     * #hooked} does.
     */
    static List<String> observed(Path dir, Game game, Observed body) throws Throwable {
        RecordShardWriter writer = writer(dir);
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

    /** The player a forced resolution of {@code ability} is for: its activator, else the turn's player. */
    private static Player holderOf(Game game, SpellAbility ability) {
        return ability.getActivatingPlayer() != null
                ? ability.getActivatingPlayer()
                : game.getPhaseHandler().getPlayerTurn();
    }

    /**
     * Fork {@code game} once and force {@code ability} to resolve, with no
     * hook installed: the game's recording is never opened, so the fork's
     * collectors hear its bus alone.
     */
    static Fork forked(Path dir, Game game, CollectionCaps caps, long seed, SpellAbility ability) {
        return forkedUnder(dir, game, caps,
                live -> new InterventionFork(game, live, new ForkBudget(caps), seed)
                        .intervene(ability, 1, holderOf(game, ability)));
    }

    /**
     * The same with the game recorded as a worker records it, every hook the
     * checkout offers installed, so the fork's clauses reach the fork's
     * collectors as a live game's reach its own; skipped on a checkout
     * without the clause hook.
     *
     * @param why what the test needs the clause hook for, shown when it skips
     */
    static Fork forkedWithClauseHook(
            Path dir, Game game, CollectionCaps caps, long seed, SpellAbility ability,
            String why) {
        assumeClauseHook(why);
        return forkedUnder(dir, game, caps, live -> {
            live.open();
            return new InterventionFork(game, live, new ForkBudget(caps), seed)
                        .intervene(ability, 1, holderOf(game, ability));
        });
    }

    /** Run {@code force} under the game's recording, and the shard it wrote to. */
    private static Fork forkedUnder(
            Path dir, Game game, CollectionCaps caps, java.util.function.Predicate<GameRecording> force) {
        RecordShardWriter writer = writer(dir);
        boolean wrote;
        try (GameRecording live = new GameRecording(
                GameRecordingRequest.of(game, writer, GAME_ID, caps)
                        .seeded(SEED))) {
            wrote = force.test(live);
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
