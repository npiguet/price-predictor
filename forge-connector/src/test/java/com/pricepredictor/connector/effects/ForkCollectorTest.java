package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

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
                2000, 0.1, interventions, probes, keywords, 0.0, 0.0);
    }

    private RecordShardWriter writer;

    private ForkCollector collector(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new ForkCollector(null, TestCollectors.context(writer, "run.0-l1.0"), caps, 42L);
    }

    /**
     * The collector's shard, closed so every record it wrote is on disk. The
     * writer creates the file with its first record, so a collector that
     * wrote nothing leaves none.
     */
    private Path closedShard() {
        writer.close();
        return writer.path();
    }

    /**
     * The same, over a game whose turn can be read.
     *
     * <p>No turn is ever taken in the shared test game, so its phase handler
     * reads turn 0 — which is all the pairing guard needs: a branch that says
     * it came from any other turn disagrees with it.
     */
    private ForkCollector collectorOverAGame(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new ForkCollector(TestCards.game(), TestCollectors.context(writer, "run.0-l1.0"), caps, 42L);
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
    }

    @Test
    void aFailedForkWritesNoRecord() {
        ForkCollector collector = collector(caps(2, 2, List.of()));
        collector.intervene(null, 1);
        assertFalse(Files.exists(closedShard()));
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
        // Spent even by a fork that fails: a null game makes every copy fail.
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("wither", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void aProbeForAnUnnamedKeywordSpendsNothing() {
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("lifelink", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
        assertTrue(collector.mayProbe("wither"));
        assertFalse(Files.exists(closedShard()));
    }

    // ── the held branch knows which step it came from ───────────────────

    /**
     * The damage-step names are a contract with Python: {@code COMBAT_SUBSTEPS}
     * in {@code effect_head_input.py} spells them the same way.
     */
    @Test
    void theDamageStepsAreNamedAsPythonSpellsThem() {
        assertEquals("first_strike", DamageStep.FIRST_STRIKE.wireName());
        assertEquals("regular", DamageStep.REGULAR.wireName());
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
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertFalse(collector.writeHeldProbe(branch(7), "run.0-l1.9"),
                "a branch seven turns older than its mirror is not a pair");
        assertFalse(Files.exists(closedShard()));
    }

    /** A branch from the turn the record was written on is written. */
    @Test
    void aBranchFromThisTurnIsWritten() throws IOException {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertTrue(collector.writeHeldProbe(branch(0), "run.0-l1.9"));
        assertEquals(1, ShardRun.readShard(closedShard()).size());
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
                "trample", "E1", "{}", List.of(), "P0", CombatShape.of(null).fields(),
                DamageStep.REGULAR, ForkCollector.TURN_UNKNOWN);

        assertEquals(ForkCollector.TURN_UNKNOWN, held.turn());
        assertTrue(collector.writeHeldProbe(held, "run.0-l1.9"));
    }

    /** A branch from a named turn, otherwise like every other hand-built one. */
    private static ForkCollector.HeldProbe branch(int turn) {
        return new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", CombatShape.of(null).fields(),
                DamageStep.REGULAR, turn);
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
        Combat combat = new Combat(attacking);
        combat.addAttacker(dreadmaw, defending);
        combat.addBlocker(dreadmaw, bears);
        // What the declare-blockers step leaves behind.
        combat.setBlocked(dreadmaw, true);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        game.getPhaseHandler().setCombat(combat);

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        ForkCollector.HeldProbe held;
        try {
            ForkCollector forks = new ForkCollector(
                    game, TestCollectors.context(writer, "run.0-l1.0"),
                    caps(0, 1, List.of("trample")), 1L);
            held = forks.probe("Trample", dreadmaw, DamageStep.REGULAR);
            assertNotNull(held, "the probe fork was discarded");

            BusBracketCollector observed = TestCollectors.bracket(game, writer, "run.0-l1.0");
            game.subscribeToEvents(observed);
            assertTrue(combat.assignCombatDamage(false), "the live step assigned no damage");
            combat.dealAssignedDamage();
            observed.finishGame(Set.of());
        } finally {
            writer.close();
        }

        String record = ShardRun.readShard(writer.path()).stream()
                .filter(r -> r.contains("\"kind\":\"combat\"")).findFirst().orElseThrow();
        String onTheBears = damageTo(held.events(), bears).toJson();
        assertTrue(onTheBears.contains("\"combat\":true"), onTheBears);
        assertTrue(onTheBears.contains("\"cause\":\"E" + dreadmaw.getId() + "\""), onTheBears);
        String onTheDreadmaw = damageTo(held.events(), dreadmaw).toJson();
        assertTrue(record.contains(onTheDreadmaw),
                "the branch renders " + onTheDreadmaw + " and the observed record " + record);
        // The stripped keyword is the branch's input, not something its step did.
        assertTrue(held.events().stream()
                        .noneMatch(e -> EffectEvent.KEYWORD_CHANGE.equals(e.type())),
                held.events().stream().map(EffectEvent::toJson).toList().toString());
    }

    // ── a forced resolution's state is the board it acted on ────────────

    /**
     * A forced targeted spell's record names the target it resolved at, as
     * an observed resolution record does.
     *
     * <p>The fork draws its targets itself, so a state rendered before that
     * draw holds an ability with no target: Disfigure resolving at the one
     * creature on the board, recorded as aimed at nothing.
     */
    @Test
    void aForcedTargetedSpellsRecordNamesItsTarget() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        // In hand, where the fork's copier can find it.
        Card disfigure = TestBoard.inHand(game, caster, "Disfigure");
        SpellAbility spell = disfigure.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, caps(1, 0, List.of()), 1L, spell);
        assertTrue(fork.wrote(), "the forced resolution observed nothing");

        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String target = "\"E" + bears.getId() + "\"";
        assertTrue(records.get(0).contains("\"refs\":{\"targets\":[" + target + "]"),
                records.get(0));
    }

    /**
     * A forced X spell's record carries the X it resolved with.
     *
     * <p>The fork announces X itself, from its own seeded source, between one
     * and what the caster's lands could pay, so the seed fixes the value:
     * Blaze with five Mountains, which could pay X up to four, and a seed that
     * draws neither bound.
     */
    @Test
    void aForcedXSpellsRecordCarriesItsX() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            TestBoard.inPlay(game, caster, "Mountain");
        }
        Card blaze = TestBoard.inHand(game, caster, "Blaze");
        SpellAbility spell = blaze.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, caps(1, 0, List.of()), 9L, spell);
        assertTrue(fork.wrote(), "the forced resolution observed nothing");

        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"modes\":[],\"x\":3,"), records.get(0));
    }

    // ── a forced resolution's state is read where an observed one's is ──

    /**
     * A forced resolution's state is the board an observed resolution of
     * the same spell records: the spell on the stack, its card in the stack
     * zone rather than in hand, and priority with its caster. The two boards
     * are built alike and one casts Pyroclasm while the other forks and
     * forces it. Pyroclasm has no target, and the observed cast here pays
     * no cost, so nothing separates the two states.
     */
    @Test
    void aForcedResolutionsStateIsTheObservedResolutionsState() throws Throwable {
        Game observedGame = pyroclasmBoard();
        SpellAbility cast = pyroclasmIn(observedGame);
        List<String> observed = ShardRun.observed(tempDir.resolve("observed"), observedGame,
                bracket -> {
                    observedGame.getStack().addAndUnfreeze(cast);
                    observedGame.getStack().resolveStack();
                    observedGame.getAction().checkStateEffects(false);
                    observedGame.fireEvent(new GameEventTurnPhase(
                            observedGame.getPlayers().get(0), PhaseType.MAIN2, ""));
                });
        String observedHalf = observed.stream()
                .filter(r -> r.contains("\"moment\":\"resolution\""))
                .findFirst().orElseThrow(() -> new AssertionError(observed.toString()));

        Game forkedGame = pyroclasmBoard();
        ShardRun.Fork fork = ShardRun.forked(tempDir.resolve("forked"), forkedGame,
                caps(1, 0, List.of()), 1L, pyroclasmIn(forkedGame));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> forced = fork.records();
        assertEquals(1, forced.size(), forced.toString());

        assertTrue(stateOf(forced.get(0)).contains("\"stack_size\":1,"), forced.get(0));
        assertEquals(stateOf(observedHalf), stateOf(forced.get(0)));
    }

    /** A Mountain for its caster, Grizzly Bears for the other seat, and Pyroclasm in hand. */
    private static Game pyroclasmBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Mountain");
        TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Pyroclasm");
        return game;
    }

    private static SpellAbility pyroclasmIn(Game game) {
        Player caster = game.getPlayers().get(0);
        SpellAbility spell = caster.getCardsIn(ZoneType.Hand).get(0).getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        return spell;
    }

    /** A record's {@code state} object, as written. */
    private static String stateOf(String record) {
        int from = record.indexOf("\"state\":");
        int to = record.indexOf(",\"payload\":");
        assertTrue(from >= 0 && to > from, record);
        return record.substring(from, to);
    }

    private static EffectEvent damageTo(List<EffectEvent> events, Card card) {
        return events.stream()
                .filter(e -> EffectEvent.DAMAGE_DEALT.equals(e.type())
                        && e.subjects().contains("E" + card.getId()))
                .findFirst().orElseThrow(() -> new AssertionError(
                        "no damage to " + card + " in "
                                + events.stream().map(EffectEvent::toJson).toList()));
    }
}
