package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import forge.game.ability.AbilityFactory;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork budgets and the record shapes they produce.
 *
 * <p>Forking a live game needs one, which is the integration test's job. What is
 * testable here is the budget arithmetic — and it is worth testing precisely,
 * because the failure mode is a run that spends its whole simulation budget on
 * forks of one board.
 */
@ExtendWith(ForgeExtension.class)
class ForkCollectorTest {

    @TempDir
    Path tempDir;

    private static CollectionCaps caps(
            int interventions, int probes, List<String> keywords) {
        return new CollectionCaps(
                2000, 0.1, interventions, probes, keywords, List.of(1, 2, 3), 0.1);
    }

    private ForkCollector collector(CollectionCaps caps) {
        return new ForkCollector(
                null, new RecordShardWriter(tempDir, "run", 0, "l1"), "run.0-l1.0",
                caps, 42L);
    }

    /**
     * The same, over a game whose turn can be read.
     *
     * <p>No turn is ever taken in the shared test game, so its phase handler
     * reads turn 0 — which is all the pairing guard needs: a branch that says
     * it came from any other turn disagrees with it.
     */
    private ForkCollector collectorOverAGame(CollectionCaps caps) {
        return new ForkCollector(
                TestCards.game(), new RecordShardWriter(tempDir, "run", 0, "l1"),
                "run.0-l1.0", caps, 42L);
    }

    // ── the per-game budget ─────────────────────────────────────────────

    @Test
    void aFreshGameMayIntervene() {
        assertTrue(collector(caps(2, 2, List.of())).mayIntervene(1));
    }

    @Test
    void anExhaustedInterventionBudgetStopsFurtherForks() {
        ForkCollector collector = collector(caps(0, 2, List.of()));
        assertFalse(collector.mayIntervene(1));
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void atMostTwoForksMayTargetOneRealResolution() {
        assertEquals(2, ForkCollector.MAX_FORKS_PER_RESOLUTION);
    }

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        ForkCollector collector = collector(caps(100, 2, List.of()));
        collector.intervene(null, 7);
        collector.intervene(null, 7);
        assertFalse(collector.mayIntervene(7));
        assertTrue(collector.mayIntervene(8));
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. A null game makes every copy fail.
        ForkCollector collector = collector(caps(1, 2, List.of()));
        assertFalse(collector.intervene(null, 1));
        assertFalse(collector.mayIntervene(1));
        assertEquals(1, collector.discardedForks());
    }

    @Test
    void aFailedForkWritesNoRecord() {
        ForkCollector collector = collector(caps(2, 2, List.of()));
        collector.intervene(null, 1);
        assertEquals(0L, collector.recordsWritten());
    }

    /**
     * The intervention resolves the ability rather than only forking.
     *
     * <p>The failure this guards against is a collector that copies the game,
     * score-checks it, snapshots it, and writes a record saying an ability was
     * forced without saying what it did — which is what this class did before
     * the resolution was written, and which looks like a working collector.
     *
     * <p>Only the wiring is checkable here, because forking needs a live game;
     * {@code tests/integration/test_effects_forks.py} asserts against a real
     * one that the records carry events.
     */
    @Test
    void interveningResolvesRatherThanOnlyCopying() throws Exception {
        var method = ForkCollector.class.getDeclaredMethod(
                "forceResolution",
                forge.game.Game.class,
                forge.game.spellability.SpellAbility.class,
                forge.game.player.Player.class);
        assertTrue(method != null, "no resolution step on the intervention path");
        // Returns the events it observed, so an empty payload can only mean the
        // ability genuinely did nothing observable.
        assertEquals(List.class, method.getReturnType());
    }

    // ── the probe switch ────────────────────────────────────────────────

    @Test
    void noProbeIsTakenWithTheFlagUnset() {
        // Whatever the build state: the flag is the runtime switch, and it is
        // deliberately separate from the build decision gate 2 drives.
        ForkCollector collector = collector(caps(2, 2, List.of()));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void onlyTheNamedKeywordsAreProbed() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertTrue(collector.mayProbe("wither"));
        assertFalse(collector.mayProbe("lifelink"));
    }

    @Test
    void theProbeBudgetIsSpentByTakingProbes() {
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertTrue(collector.recordProbeBranch(
                "wither", "run.0.1", "{}", "{}", "P0"));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void aProbeBranchIsRecordedAsAnOrdinaryCombatRecord() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertTrue(collector.recordProbeBranch(
                "wither", "run.0.1", "{}", "{}", "P0"));
        assertEquals(1L, collector.recordsWritten());
    }

    @Test
    void aProbeForAnUnnamedKeywordWritesNothing() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertFalse(collector.recordProbeBranch(
                "lifelink", "run.0.1", "{}", "{}", "P0"));
        assertEquals(0L, collector.recordsWritten());
    }


    // ── an intervention chooses before it resolves ──────────────────────

    /**
     * A line with a target and nowhere to point it is abandoned.
     *
     * <p>The feature spec says an intervention resolves "with chosen targets and
     * modes" and the first implementation chose neither: it set the activating
     * player, pushed the ability and resolved it. A "deal 5 damage to target
     * player" that resolves with no target does nothing, which is why none of
     * the 88 sampled interventional records had a target and why the empty ones
     * were Lava Axe, Disintegrate, Drain Life, Mind Control and their like.
     *
     * <p>Abandoning is the point: an empty-because-untargeted record and an
     * empty-because-the-effect-did-nothing record are the same bytes, and a
     * counterfactual nobody can read is worse than one never written.
     */
    @Test
    void aTargetedLineWithNothingToTargetIsAbandoned() {
        SpellAbility bolt = AbilityFactory.getAbility(
                "AB$ DealDamage | Cost$ R | ValidTgts$ Creature | TgtPrompt$ x"
                        + " | NumDmg$ 3",
                TestCards.build("Fountain of Youth"));
        // A game with nothing in it: every candidate list comes back empty,
        // which is the shape of a board where the spell has no legal target.
        bolt.setActivatingPlayer(new Player("nobody", TestCards.game(), 99));

        assertFalse(collector(caps(2, 2, List.of())).chooseTargets(bolt));
    }

    /** A line that targets nothing needs nothing chosen. */
    @Test
    void aLineThatTargetsNothingIsReadyToResolve() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        assertTrue(collector(caps(2, 2, List.of())).chooseTargets(gain));
    }

    /**
     * An X the fork never paid is announced rather than left null.
     *
     * <p>A null X resolves as an X of zero, which is the same silence a missing
     * target produces.
     */
    @Test
    void anXSpellAnnouncesSomethingRatherThanNothing() {
        SpellAbility drain = AbilityFactory.getAbility(
                "AB$ LoseLife | Cost$ X B | Defined$ Player.Opponent | LifeAmount$ X",
                TestCards.build("Fountain of Youth"));
        assertTrue(drain.costHasX(), "the script has to have an X to announce");

        collector(caps(2, 2, List.of())).announceX(drain);

        assertEquals(1, drain.getXManaCostPaid());
    }

    @Test
    void anAbilityWithNoXIsLeftAlone() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        collector(caps(2, 2, List.of())).announceX(gain);

        assertNull(gain.getXManaCostPaid());
    }

    // ── the held branch knows which step it came from ───────────────────

    /**
     * A branch carries the damage step it forked from.
     *
     * <p>Without it the branch is completed against whichever combat record
     * comes next, and in the first corpus that was systematically the wrong one:
     * every first-strike-step fork was empty, its step wrote no real record, and
     * the branch was written against the regular step's.
     */
    @Test
    void aHeldBranchNamesItsDamageStep() {
        ForkCollector.HeldProbe held = new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                ForkCollector.SUBSTEP_FIRST_STRIKE);

        assertEquals("first_strike", held.substep());
        // Spelled as the bracket collector spells it, because the two classes
        // have to agree about which step they are describing.
        assertEquals("regular", ForkCollector.SUBSTEP_REGULAR);
    }

    // ── the held branch knows *when* it came from ───────────────────────

    /**
     * A branch and the record it mirrors describe the same turn, or neither is
     * written.
     *
     * <p>The substep alone was not enough. A branch whose own damage step wrote
     * no combat record kept waiting for the next record of that substep, which
     * in the smoke corpus arrived up to <b>eight turns</b> later: 55 of 1,069
     * probe records mirrored a record from a different turn, and every mirrored
     * pair whose turns disagreed was one of those. Because the branch's own
     * snapshot is honest — the copier carries the turn across, and not one of
     * the 1,788 interventional forks ever named a stale one — relabelling it
     * would make the state block lie, so the branch is dropped instead.
     *
     * <p>Dropped counts as a discarded fork: the simulation was spent either
     * way, and the counter is what makes the loss visible in a run's summary.
     */
    @Test
    void aBranchFromAnEarlierTurnIsNotWrittenAgainstThisOnesRecord() {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertFalse(collector.writeHeldProbe(branch(7), "run.0-l1.9"),
                "a branch seven turns older than its mirror is not a pair");
        assertEquals(0L, collector.recordsWritten());
        assertEquals(1, collector.discardedForks());
    }

    /** A branch from the turn the record was written on is written. */
    @Test
    void aBranchFromThisTurnIsWritten() {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertTrue(collector.writeHeldProbe(branch(0), "run.0-l1.9"));
        assertEquals(1L, collector.recordsWritten());
    }

    /**
     * A branch with no turn to check is written, as it always was.
     *
     * <p>{@link ForkCollector#TURN_UNKNOWN} is what a hand-built branch carries
     * and what a collector with no live game reads, and the guard fires on
     * neither: there is no turn to disagree with.
     */
    @Test
    void aBranchWithNoTurnIsPairedAsBefore() {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));
        ForkCollector.HeldProbe held = new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                ForkCollector.SUBSTEP_REGULAR);

        assertEquals(ForkCollector.TURN_UNKNOWN, held.turn());
        assertTrue(collector.writeHeldProbe(held, "run.0-l1.9"));
    }

    /** A branch from a named turn, otherwise like every other hand-built one. */
    private static ForkCollector.HeldProbe branch(int turn) {
        return new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                ForkCollector.SUBSTEP_REGULAR, turn);
    }

    // ── the seeded source ───────────────────────────────────────────────

    @Test
    void bothBranchesDrawFromOneSeededSource() {
        // A difference between the branches has to be the keyword, not the
        // draw, so the source is seeded per fork rather than per branch.
        ForkCollector first = collector(caps(2, 2, List.of()));
        ForkCollector second = collector(caps(2, 2, List.of()));
        assertEquals(
                first.seededRandom().nextInt(1000),
                second.seededRandom().nextInt(1000));
    }
}
