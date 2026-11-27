package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.ForkTestCaps.forkCaps;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork budgets: the per-resolution cap, the probe switch, the held probe
 * branch's turn and how a probe branch reads damage.
 *
 * <p>Forking a live game needs one, which is the integration test's job. What is
 * testable here is the budget arithmetic: it is worth testing precisely,
 * because the failure mode is a run that spends its whole simulation budget on
 * forks of one board.
 */
@ExtendWith(ForgeExtension.class)
class ForkBudgetsTest {

    @TempDir
    Path tempDir;

    private RecordShardWriter writer;
    /** The budget the last fork built here spends. */
    private ForkBudget budget;

    private CollectorContext context(CollectionCaps caps) {
        writer = ShardRun.writer(tempDir);
        budget = new ForkBudget(caps);
        return TestCollectors.context(writer, ShardRun.GAME_ID);
    }

    private DamageStepProbe probes(CollectionCaps caps) {
        return new DamageStepProbe(null, context(caps), budget);
    }

    /**
     * The same, over a game whose turn can be read.
     *
     * <p>No turn is ever taken in the shared test game, so its phase handler
     * reads turn 0 — which is all the pairing guard needs: a branch that says
     * it came from any other turn disagrees with it.
     */
    private DamageStepProbe probesOverAGame(CollectionCaps caps) {
        return new DamageStepProbe(TestCards.game(), context(caps), budget);
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        CollectionCaps caps = forkCaps().interventions(100).probes(2).build();
        ForkBudget spent = new ForkBudget(caps);
        try (RealForks forks = RealForks.interventions(tempDir, spent, caps)) {
            InterventionFork collector = forks.fork;
            boolean bothTaken = collector.intervene(forks.bolt, 7, forks.holder)
                    && collector.intervene(forks.bolt, 7, forks.holder);

            assertTrue(bothTaken, "both forks of the resolution were taken");
            assertFalse(collector.mayIntervene(7));
            assertTrue(collector.mayIntervene(8));
        }
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. An ability of a card in no zone has no counterpart on
        // the copy, so the real fork is discarded after it spent its budget.
        CollectionCaps caps = forkCaps().interventions(1).probes(2).build();
        ForkBudget spent = new ForkBudget(caps);
        try (RealForks forks = RealForks.interventions(tempDir, spent, caps)) {
            long refused = ForkRefusals.counts().getOrDefault("intervention no counterpart", 0L);
            assertFalse(forks.fork.intervene(forks.abilityOfACardInNoZone(), 1, forks.holder));
            assertEquals(refused + 1, ForkRefusals.counts().get("intervention no counterpart"),
                    "the fork was refused on its copy, not skipped before it");
            assertFalse(forks.fork.mayIntervene(1));
        }
    }

    @Test
    void aFailedForkWritesNoRecord() throws IOException {
        CollectionCaps caps = forkCaps().interventions(2).probes(2).build();
        try (RealForks forks = RealForks.interventions(tempDir, new ForkBudget(caps), caps)) {
            assertFalse(forks.fork.intervene(forks.abilityOfACardInNoZone(), 1, forks.holder));
            Path shard = forks.closedShard();
            assertEquals(List.of(), Files.exists(shard) ? ShardRun.readShard(shard) : List.of());
        }
    }

    // ── the probe switch ────────────────────────────────────────────────

    @Test
    void noProbeIsTakenWithTheFlagUnset() {
        // Whatever the build state: the flag is the runtime switch, and it is
        // deliberately separate from the build decision gate 2 drives.
        assertFalse(new ForkBudget(forkCaps().interventions(2).probes(2).build()).mayProbe("wither"));
    }

    @Test
    void onlyTheNamedKeywordsAreProbed() {
        ForkBudget budget = new ForkBudget(forkCaps().interventions(2).probes(2).probeKeywords("wither").build());
        assertTrue(budget.mayProbe("wither"));
        assertFalse(budget.mayProbe("lifelink"));
    }

    @Test
    void theProbeBudgetIsSpentByTakingProbes() {
        // A real probe of a blocked attacker's trample, over a board whose
        // copy is made and played through the damage step.
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card dreadmaw = TestBoard.onBattlefield(game, attacking, "Colossal Dreadmaw");
        Card bears = TestBoard.onBattlefield(game, defending, "Grizzly Bears");
        TestBoard.blockedCombat(new TestBoard.CombatSides(game, attacking, defending), dreadmaw, bears);
        DamageStepProbe collector = new DamageStepProbe(
                game, context(forkCaps().interventions(2).probes(1).probeKeywords("trample").build()),
                budget);

        assertTrue(collector.probe("trample", dreadmaw, DamageStep.REGULAR).isPresent());
        assertFalse(budget.mayProbe("trample"));
    }

    @Test
    void aProbeForAnUnnamedKeywordSpendsNothing() {
        DamageStepProbe collector = probes(forkCaps().interventions(2).probes(1).probeKeywords("wither").build());
        assertTrue(collector.probe("lifelink", TestCards.build("Grizzly Bears"), DamageStep.REGULAR).isEmpty());
        assertTrue(budget.mayProbe("wither"));
        assertFalse(Files.exists(ShardRun.closed(writer)));
    }

    // ── the held branch knows *when* it came from ───────────────────────

    /**
     * A branch and the record it mirrors describe the same turn, or neither is
     * written.
     *
     * <p>The substep alone is not enough. A branch whose own damage step wrote
     * no combat record keeps waiting for the next record of that substep,
     * which can arrive turns later. Because the branch's own snapshot is
     * honest — the copier carries the turn across — relabelling it would make
     * the state block lie, so the branch is dropped instead.
     *
     * <p>Dropped still spends the fork's budget: the simulation was spent
     * either way.
     */
    @Test
    void aBranchFromAnEarlierTurnIsNotWrittenAgainstThisOnesRecord() {
        DamageStepProbe collector = probesOverAGame(forkCaps().interventions(2).probes(2).probeKeywords("trample").build());

        assertFalse(collector.writeHeldProbe(branch(7), "run.0-l1.9"),
                "a branch seven turns older than its mirror is not a pair");
        assertFalse(Files.exists(ShardRun.closed(writer)));
    }

    /** A branch from the turn the record was written on is written. */
    @Test
    void aBranchFromThisTurnIsWritten() throws IOException {
        DamageStepProbe collector = probesOverAGame(forkCaps().interventions(2).probes(2).probeKeywords("trample").build());

        assertTrue(collector.writeHeldProbe(branch(0), "run.0-l1.9"));
        assertEquals(1, ShardRun.readShard(ShardRun.closed(writer)).size());
    }

    /**
     * A branch with no turn to check is written.
     *
     * <p>{@link DamageStepProbe#TURN_UNKNOWN} is what a hand-built branch carries
     * and what a collector with no live game reads, and the guard fires on
     * neither: there is no turn to disagree with.
     */
    @Test
    void aBranchWithNoTurnIsWritten() {
        DamageStepProbe collector = probesOverAGame(forkCaps().interventions(2).probes(2).probeKeywords("trample").build());

        assertTrue(collector.writeHeldProbe(branch(DamageStepProbe.TURN_UNKNOWN), "run.0-l1.9"));
    }

    /** A branch from a named turn, otherwise like every other hand-built one. */
    private static DamageStepProbe.HeldProbe branch(int turn) {
        return new DamageStepProbe.HeldProbe(
                "trample", "E1", RecordState.EMPTY, List.of(), "P0", CombatShape.of(null),
                DamageStep.REGULAR, turn, Map.of());
    }

    // ── a probe branch reads damage as the observed record does ─────────

    /**
     * A probe branch's combat damage carries the combat flag and the named
     * cause the observed combat record carries.
     *
     * <p>Gate 2 reads the branch against the record it mirrors, so the two
     * must render one damage the same way. The blocker's damage to the
     * attacker is the same in both branches — the stripped keyword is the
     * attacker's trample — so its rendered event must match verbatim.
     */
    @Test
    void aProbeBranchsCombatDamageReadsAsTheObservedRecordReadsIt() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card dreadmaw = TestBoard.onBattlefield(game, attacking, "Colossal Dreadmaw");
        Card bears = TestBoard.onBattlefield(game, defending, "Grizzly Bears");
        Combat combat = TestBoard.blockedCombat(new TestBoard.CombatSides(game, attacking, defending), dreadmaw, bears);

        RecordShardWriter writer = ShardRun.writer(tempDir);
        DamageStepProbe.HeldProbe held;
        try {
            DamageStepProbe forks = new DamageStepProbe(
                    game, TestCollectors.context(writer, ShardRun.GAME_ID),
                    new ForkBudget(forkCaps().interventions(0).probes(1).probeKeywords("trample").build()));
            held = forks.probe("Trample", dreadmaw, DamageStep.REGULAR)
                    .orElseThrow(() -> new AssertionError("the probe fork was discarded"));

            BusBracketCollector observed = TestCollectors.bracket(game, writer, ShardRun.GAME_ID);
            game.subscribeToEvents(observed);
            assertTrue(combat.assignCombatDamage(false), "the live step assigned no damage");
            combat.dealAssignedDamage();
            observed.finishGame(Set.of());
        } finally {
            writer.close();
        }

        String record = ShardRun.readShard(writer.path()).stream()
                .filter(r -> r.contains("\"kind\":\"combat\"")).findFirst().orElseThrow();
        // Rendered as the branch's record renders it, which names each
        // damage's dealer as the observed record does (FR-035).
        String branch = held.payload().json();
        String onTheBears = damageTo(held.events(), bears).toJson();
        assertTrue(onTheBears.contains("\"combat\":true"), onTheBears);
        assertTrue(onTheBears.contains("\"cause\":\"E" + dreadmaw.getId() + "\""), onTheBears);
        assertTrue(onTheBears.contains("\"source\":\"E" + dreadmaw.getId() + "\""), onTheBears);
        String onTheDreadmaw = damageTo(held.events(), dreadmaw).toJson();
        assertTrue(branch.contains(onTheDreadmaw), branch);
        assertTrue(record.contains(onTheDreadmaw),
                "the branch renders " + onTheDreadmaw + " and the observed record " + record);
        // The stripped keyword is the branch's input, not something its step did.
        assertTrue(held.events().stream()
                        .noneMatch(e -> EventType.KEYWORD_CHANGE.equals(e.type())),
                held.events().stream().map(EffectEvent::toJson).toList().toString());
    }

    private static EffectEvent damageTo(List<EffectEvent> events, Card card) {
        return events.stream()
                .filter(e -> EventType.DAMAGE_DEALT.equals(e.type())
                        && e.subjects().contains("E" + card.getId()))
                .findFirst().orElseThrow(() -> new AssertionError(
                        "no damage to " + card + " in "
                                + events.stream().map(EffectEvent::toJson).toList()));
    }
}
