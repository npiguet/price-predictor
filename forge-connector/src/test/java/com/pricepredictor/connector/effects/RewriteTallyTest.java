package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.StdStreams;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.TestCollectors.replacementOfMode;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The worker's rewrite tally, and the summary it prints. */
@ExtendWith(ForgeExtension.class)
class RewriteTallyTest {

    /**
     * A record with no outgoing names the parameters that moved under it.
     *
     * <p>The count alone says the channel filled; this says whether the rows in
     * it are as thin as they look or whether the normaliser cannot express what
     * the replacement changed, and it hands over the key that would close the
     * gap.
     */
    @Test
    void aRecordWithNoOutgoingNamesTheParametersTheNormaliserDidNotRead() {
        RewriteTally tally = new RewriteTally();
        Card card = TestCards.build("Mountain");
        tally.record("Moved", "replaced", false, false,
                Map.of("Card", card, "LibraryPosition", 0),
                Map.of("Card", card, "LibraryPosition", -1));

        String summary = tally.summary();
        assertTrue(summary.contains("Moved.LibraryPosition=1"), summary);
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
    }

    /** And one that moved nothing at all is not counted against the reader. */
    @Test
    void aRecordThatMovedNothingIsNoNormaliserGap() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Untap", "skipped", false, false, same, same);

        String summary = tally.summary();
        assertTrue(summary.contains(
                "0 carried no outgoing while a raw parameter moved: none"), summary);
    }

    /**
     * The headline is the volume and the results, because that is the question.
     *
     * <p>The operator needs to know in the first hour whether the channel
     * filled. A breakdown that is all {@code ?} is a Forge jar whose listener
     * predates the result argument, which reads identically to a wired one on
     * every other line of the summary.
     */
    @Test
    void theSummaryLeadsWithTheVolumeAndTheResults() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", "updated", true, true, same, same);
        for (int i = 0; i < 9; i++) {
            tally.record("Moved", "not_replaced", false, false, same, same);
        }

        String summary = tally.summary();
        assertTrue(summary.contains(
                "10 records, 1 rewrote a parameter in place (10.0%), "
                        + "1 named the ability that ran instead"), summary);
        assertTrue(summary.contains("by result: not_replaced=9, updated=1"), summary);
        assertTrue(summary.contains("by mode and result: Moved.not_replaced=9"),
                summary);
    }

    /** A listener that never learnt the result says so rather than guessing. */
    @Test
    void aResultlessRecordIsItsOwnRowInTheBreakdown() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", null, false, false, same, same);

        assertTrue(tally.summary().contains("by result: ?=1"), tally.summary());
    }

    /**
     * A worker that saw no replacement says nothing.
     *
     * <p>Whether the hook is installed at all is {@code PatchHooks.report()}'s
     * line, and repeating it once a game would be noise in every worker log.
     */
    @Test
    void aGameWithNoReplacementPrintsNothing() throws Throwable {
        RewriteTally tally = new RewriteTally();

        assertEquals("", StdStreams.captureStdout(tally::endOfGame));
    }

    /**
     * The tally separates an unreadable change from no change at all.
     *
     * <p>A mode's records split into those that changed nothing whatever and
     * those whose change has no slot to land in — a {@code Moved} that changed
     * only the entry-counter table, which {@code zone_change} cannot express.
     * Every row is written, so the question the tally answers is which of them
     * a reader is being short-changed on, and it answers it on one line rather
     * than by arithmetic across three.
     */
    @Test
    void theTallySeparatesAnUnreadableChangeFromNoChange() {
        RewriteTally tally = new RewriteTally();

        tally.record("Moved", "replaced", false, true,
                Map.of("Destination", "Battlefield"),
                Map.of("Destination", "Battlefield"));
        tally.record("Moved", "replaced", false, true,
                Map.of("CounterMap", "{}"),
                Map.of("CounterMap", "{P1P1=1}"));

        String summary = tally.summary();
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
        assertTrue(summary.contains("of those, by mode: Moved=1"), summary);
        assertTrue(summary.contains("Moved.CounterMap=1"), summary);
    }

    // ── a capped game ───────────────────────────────────────────────────

    private static final Method ON_REPLACEMENT =
            HookShapes.method(HookShapes.Replacement.class, "onReplacement");

    @TempDir
    Path tempDir;

    private static Map<String, Object> damage(Card target, int amount) {
        return new LinkedHashMap<>(Map.of("DamageAmount", amount, "Affected", target));
    }

    /**
     * One game under a cap of {@code cap} records; the body drives the
     * rewrite channel by hand. Returns the shard's lines and how many
     * records the worker's tally gained.
     */
    private record Capped(List<String> lines, long tallied) {
    }

    private Capped capped(int cap, ShardRun.Body body) throws Throwable {
        long before = RewriteChannel.JVM_REWRITES.records();
        RecordShardWriter writer =
                new RecordShardWriter(tempDir, "run", 0, "l1", cap);
        var game = TestCards.game();
        BusBracketCollector bracket = TestCollectors.bracket(game, writer, ShardRun.GAME_ID);
        try (Channels collectors = Channels.of(game, new Channels.Setup(
                TestCollectors.context(writer, ShardRun.GAME_ID),
                TestCaps.unprobed(), 1L, bracket))) {
            body.run(bracket, collectors);
        } finally {
            writer.close();
        }
        List<String> lines = ShardRun.readShard(writer.path()).stream()
                .filter(line -> line.contains("\"kind\":\"rewrite\"")).toList();
        return new Capped(lines, RewriteChannel.JVM_REWRITES.records() - before);
    }

    private static Object[] replacement(
            ReplacementEffect effect, Card target, int amount, Object buffered) {
        return new Object[]{
                effect, damage(target, amount), damage(target, amount),
                ReplacementResult.Prevented, null, buffered};
    }

    /** A record the cap withholds is not in the tally: it counts what reached the shard. */
    @Test
    void aCappedGamesTallyEqualsTheLinesInTheShard() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        Card target = TestCards.build("Grizzly Bears");

        Capped game = capped(2, (bracket, collectors) -> {
            for (int amount = 1; amount <= 5; amount++) {
                collectors.rewrites().handler().invoke(
                        null, ON_REPLACEMENT, replacement(prevention, target, amount, null));
            }
        });

        assertEquals(2, game.lines().size(), game.lines().toString());
        assertEquals(game.lines().size(), game.tallied());
    }

    /**
     * A record still waiting for its ability's run when the game ends is an
     * end-of-game record: the cap does not withhold it, and the tally counts it.
     */
    @Test
    void aWaitingRecordAtTheGamesEndPassesTheCap() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        Card target = TestCards.build("Grizzly Bears");
        var draw = forge.game.ability.AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You", TestCards.build("Swans of Bryn Argoll"));

        Capped game = capped(1, (bracket, collectors) -> {
            collectors.rewrites().handler().invoke(
                    null, ON_REPLACEMENT, replacement(prevention, target, 1, null));
            collectors.rewrites().handler().invoke(
                    null, ON_REPLACEMENT, replacement(prevention, target, 2, draw));
        });

        assertEquals(2, game.lines().size(), game.lines().toString());
        assertEquals(2, game.tallied());
    }
}
