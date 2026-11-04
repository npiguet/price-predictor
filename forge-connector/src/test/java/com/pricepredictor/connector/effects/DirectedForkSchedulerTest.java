package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.BoardPrecondition;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * When a directed fork is taken and for which rule (FR-010a, FR-012,
 * FR-013; story 3 acceptance 4).
 *
 * <p>The order and budget tests take the forks through a stand-in that only
 * notes which rule it was handed; the score-check test takes a real one.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkSchedulerTest {

    @TempDir
    Path tempDir;

    private static final BoardPrecondition ALWAYS = live -> Optional.of(Binding.empty());
    private static final BoardPrecondition NEVER = live -> Optional.empty();

    private static DirectedRule rule(String id, BoardPrecondition precondition) {
        return DirectedRule.of(id, id + "-1", precondition, "tapped",
                new Application(ForkOperation.TAP, binding -> new TapOperation(binding.card("creature"), true)));
    }

    /** The rules a stand-in fork was handed, in order. */
    private final List<String> forked = new ArrayList<>();

    private DirectedForkScheduler scheduler(
            CollectionCaps caps, DirectedForkTally tally, DirectedRule... rules) {
        return new DirectedForkScheduler(TestCards.game(), caps, DirectedRules.of(rules),
                new ForkBudget(caps), tally, 7L,
                (rule, variant, application, binding) -> forked.add(rule.rule()));
    }

    private DirectedForkScheduler scheduler(CollectionCaps caps, DirectedRule... rules) {
        return scheduler(caps, new DirectedForkTally(), rules);
    }

    @Test
    void aRuleForksAtMostOncePerGame() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(5, 1.0, "A").build(), rule("A", ALWAYS));

        for (int boundary = 0; boundary < 10; boundary++) {
            scheduler.collect(PhaseType.MAIN1);
        }

        assertEquals(List.of("A"), forked);
    }

    /** With probability 1, the first boundary where the precondition holds forks. */
    @Test
    void probabilityOneForksAtTheFirstMatch() {
        AtomicBoolean holds = new AtomicBoolean(false);
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(2, 1.0, "A").build(),
                rule("A", live -> holds.get() ? Optional.of(Binding.empty()) : Optional.empty()));

        assertEquals(0, scheduler.collect(PhaseType.UPKEEP));
        holds.set(true);
        assertEquals(1, scheduler.collect(PhaseType.DRAW));
        assertEquals(List.of("A"), forked);
    }

    /**
     * Below 1, a precondition that holds at every boundary forks once, at a
     * boundary the game's draws choose, and the draws repeat with the seed.
     */
    @Test
    void belowOneTheDrawChoosesTheBoundary() {
        CollectionCaps caps = TestCaps.caps().directed(2, 0.25, "A").build();
        int first = boundaryOfTheFork(scheduler(caps, rule("A", ALWAYS)));
        forked.clear();
        int again = boundaryOfTheFork(scheduler(caps, rule("A", ALWAYS)));

        assertTrue(first >= 0, "a quarter of 200 boundaries forks");
        assertEquals(first, again, "the same game seed draws the same boundary");
        assertEquals(List.of("A"), forked);
    }

    private static int boundaryOfTheFork(DirectedForkScheduler scheduler) {
        int at = -1;
        for (int boundary = 0; boundary < 200; boundary++) {
            if (scheduler.collect(PhaseType.MAIN1) > 0) {
                assertEquals(-1, at, "forked twice");
                at = boundary;
            }
        }
        return at;
    }

    @Test
    void aRuleWhosePreconditionNeverHoldsNeverForks() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(2, 1.0, "A").build(), rule("A", NEVER));

        scheduler.collect(PhaseType.MAIN1);

        assertTrue(forked.isEmpty());
    }

    /** The fewest forks so far in the JVM's lifetime goes first, whatever the list order. */
    @Test
    void theRuleWithTheFewestLifetimeForksGoesFirst() {
        DirectedForkTally tally = new DirectedForkTally();
        tally.count("A");
        tally.count("A");
        tally.count("B");
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(3, 1.0, "A", "B", "C").build(), tally,
                rule("A", ALWAYS), rule("B", ALWAYS), rule("C", ALWAYS));

        scheduler.collect(PhaseType.MAIN1);

        assertEquals(List.of("C", "B", "A"), forked);
        assertEquals(3, tally.taken("A"), "a fork taken counts in the lifetime tally");
    }

    /** A tie goes to the rule listed first in --directed-fork-rules, not registered first. */
    @Test
    void aTieGoesToTheRuleListedFirst() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(1, 1.0, "B", "A").build(),
                rule("A", ALWAYS), rule("B", ALWAYS));

        scheduler.collect(PhaseType.MAIN1);

        assertEquals(List.of("B"), forked);
    }

    /** Once the per-game budget is spent no rule is tried, at this boundary or a later one. */
    @Test
    void theBudgetStopsFurtherRules() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(2, 1.0, "A", "B", "C").build(),
                rule("A", ALWAYS), rule("B", ALWAYS), rule("C", ALWAYS));

        assertEquals(2, scheduler.collect(PhaseType.MAIN1));
        assertEquals(0, scheduler.collect(PhaseType.MAIN2));

        assertEquals(List.of("A", "B"), forked);
    }

    /**
     * Not at the beginning of combat, where the copy's score never agrees:
     * the rule keeps its one fork for a boundary that can keep it.
     */
    @Test
    void noForkIsTakenAtTheBeginningOfCombat() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(2, 1.0, "A").build(), rule("A", ALWAYS));

        assertEquals(0, scheduler.collect(PhaseType.COMBAT_BEGIN));
        assertEquals(1, scheduler.collect(PhaseType.COMBAT_DECLARE_ATTACKERS));
    }

    @Test
    void aRuleTheRegistryLacksTakesNoFork() {
        DirectedForkScheduler scheduler = scheduler(
                TestCaps.caps().directed(2, 1.0, "unregistered", "A").build(), rule("A", ALWAYS));

        scheduler.collect(PhaseType.MAIN1);

        assertEquals(List.of("A"), forked);
    }

    // ── the score check (FR-013, acceptance 4) ──────────────────────────

    /**
     * A fork whose copy already disagrees with the live game's score is
     * discarded before any change, writes nothing, and still counts: against
     * the budget, as the rule's one fork of the game and in the tally.
     *
     * <p>The disagreement is made the one way a test can make it: the
     * perspective the score is read from, which the check reads after the
     * copy, first costs the live opponent three life, so the live game the
     * copy is compared with is no longer the one it copied. Its twin, with
     * nothing changed, is kept.
     */
    @Test
    void aForkWhoseCopyDisagreesWithTheLiveScoreIsDiscardedAndStillCounts() throws Throwable {
        Taken discarded = takeOneFork(true);

        assertFalse(discarded.kept, "the copy disagreed with the live game");
        assertTrue(discarded.forkRecords.isEmpty(), discarded.forkRecords.toString());
        assertFalse(discarded.budgetLeft, "the discarded fork spent the budget");
        assertEquals(1, discarded.tally, "and counts in the lifetime tally");
        assertEquals(0, discarded.retried, "and was the rule's one fork of the game");

        Taken kept = takeOneFork(false);

        assertTrue(kept.kept, "an undisturbed copy agrees and is kept");
        assertFalse(kept.forkRecords.isEmpty(), "and its records are written");
    }

    /**
     * @param budgetLeft whether a budget of two still had room once one more
     *                   fork was spent beside this one
     * @param retried    the forks the same rule took at the next boundary
     */
    private record Taken(boolean kept, List<String> forkRecords, boolean budgetLeft, int tally, int retried) {
    }

    private Taken takeOneFork(boolean disturb) throws Throwable {
        Game game = ForkPlayForwardTest.board();
        ForkPlayForwardTest.sneakAttack(game);
        game.getPhaseHandler().devModeSet(PhaseType.END_OF_TURN, game.getPlayers().get(0), 3);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        DirectedRule rule = DirectedRule.of("A", "A-1", live -> Optional.of(Binding.empty().with("creature", bears)),
                "tapped", new Application(ForkOperation.TAP,
                        binding -> new TapOperation(binding.card("creature"), true)));
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).directed(2, 1.0, "A").build();
        Supplier<Player> perspective = () -> {
            if (disturb) {
                opponent.setLife(opponent.getLife() - 3, null);
            }
            return active;
        };
        Path dir = Files.createDirectories(tempDir.resolve(disturb ? "disturbed" : "undisturbed"));
        RecordShardWriter writer = ShardRun.writer(dir);
        DirectedForkTally tally = new DirectedForkTally();
        ForkBudget budget = new ForkBudget(caps);
        boolean[] kept = new boolean[1];
        int retried;
        boolean budgetLeft;
        try (GameRecording recording = new GameRecording(game, writer, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            DirectedFork fork = new DirectedFork(game, recording.playForward(), () -> null, perspective);
            DirectedForkScheduler scheduler = new DirectedForkScheduler(
                    game, caps, DirectedRules.of(rule), budget, tally, 7L,
                    (r, variant, application, binding) -> kept[0] = fork.take(r, variant, application, binding));
            scheduler.collect(PhaseType.END_OF_TURN);
            retried = scheduler.collect(PhaseType.CLEANUP);
            budget.spendDirected();
            budgetLeft = budget.mayDirect();
        } finally {
            writer.close();
        }
        List<String> written = Files.exists(writer.path()) ? ShardRun.readShard(writer.path()) : List.of();
        List<String> forkRecords = written.stream()
                .filter(record -> Boolean.TRUE.equals(parse(record).get("fork"))).toList();
        return new Taken(kept[0], forkRecords, budgetLeft, tally.taken("A"), retried);
    }
}
