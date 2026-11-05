package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.DirectedTag;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.payloadOf;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.RecordLines.stateOf;
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
class ForkRecordsTest {

    @TempDir
    Path tempDir;

    /** The caps every fork test starts from: room for every mana record. */
    private static TestCaps forkCaps() {
        return TestCaps.caps().manaCap(2000);
    }

    private RecordShardWriter writer;
    /** The budget the last fork built here spends. */
    private ForkBudget budget;

    private CollectorContext context(CollectionCaps caps) {
        writer = ShardRun.writer(tempDir);
        budget = new ForkBudget(caps);
        return TestCollectors.context(writer, ShardRun.GAME_ID);
    }

    private InterventionFork interventions(CollectionCaps caps) {
        return new InterventionFork(null, context(caps), budget, 42L);
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

    // ── the per-game budget ─────────────────────────────────────────────

    @Test
    void aFreshGameMayIntervene() {
        assertTrue(new ForkBudget(forkCaps().interventions(2).probes(2).build()).mayIntervene(1));
    }

    @Test
    void anExhaustedInterventionBudgetStopsFurtherForks() {
        assertFalse(new ForkBudget(forkCaps().interventions(0).probes(2).build()).mayIntervene(1));
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        InterventionFork collector = interventions(forkCaps().interventions(100).probes(2).build());
        collector.intervene(null, 7);
        collector.intervene(null, 7);
        assertFalse(collector.mayIntervene(7));
        assertTrue(collector.mayIntervene(8));
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. A null game makes every copy fail.
        InterventionFork collector = interventions(forkCaps().interventions(1).probes(2).build());
        assertFalse(collector.intervene(null, 1));
        assertFalse(collector.mayIntervene(1));
    }

    @Test
    void aFailedForkWritesNoRecord() {
        InterventionFork collector = interventions(forkCaps().interventions(2).probes(2).build());
        collector.intervene(null, 1);
        assertFalse(Files.exists(ShardRun.closed(writer)));
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
        // Spent even by a fork that fails: a null game makes every copy fail.
        DamageStepProbe collector = probes(forkCaps().interventions(2).probes(1).probeKeywords("wither").build());
        assertNull(collector.probe("wither", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
        assertFalse(budget.mayProbe("wither"));
    }

    @Test
    void aProbeForAnUnnamedKeywordSpendsNothing() {
        DamageStepProbe collector = probes(forkCaps().interventions(2).probes(1).probeKeywords("wither").build());
        assertNull(collector.probe("lifelink", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
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
                "trample", "E1", "{}", List.of(), "P0", CombatShape.of(null),
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
        Combat combat = TestBoard.blockedCombat(game, attacking, defending, dreadmaw, bears);

        RecordShardWriter writer = ShardRun.writer(tempDir);
        DamageStepProbe.HeldProbe held;
        try {
            DamageStepProbe forks = new DamageStepProbe(
                    game, TestCollectors.context(writer, ShardRun.GAME_ID),
                    new ForkBudget(forkCaps().interventions(0).probes(1).probeKeywords("trample").build()));
            held = forks.probe("Trample", dreadmaw, DamageStep.REGULAR);
            assertNotNull(held, "the probe fork was discarded");

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
        String branch = held.payload();
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

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, forkCaps().interventions(1).probes(0).build(), 1L, spell);
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

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, forkCaps().interventions(1).probes(0).build(), 9L, spell);
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
        SpellAbility cast = firstSpellInHand(observedGame);
        List<String> observed = ShardRun.observed(tempDir.resolve("observed"), observedGame,
                bracket -> {
                    observedGame.getStack().addAndUnfreeze(cast);
                    observedGame.getStack().resolveStack();
                    observedGame.getAction().checkStateEffects(false);
                    TestBoard.nextStep(observedGame);
                });
        List<String> observedHalves = records(observed, "resolution");
        assertFalse(observedHalves.isEmpty(), observed.toString());
        String observedHalf = observedHalves.get(0);

        Game forkedGame = pyroclasmBoard();
        ShardRun.Fork fork = ShardRun.forked(tempDir.resolve("forked"), forkedGame,
                forkCaps().interventions(1).probes(0).build(), 1L, firstSpellInHand(forkedGame));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> forced = fork.records();
        assertEquals(1, forced.size(), forced.toString());

        assertTrue(stateOf(forced.get(0)).contains("\"stack_size\":1,"), forced.get(0));
        assertEquals(stateOf(observedHalf), stateOf(forced.get(0)));
    }

    // ── a forced record holds what live attributes to the forced line ───

    /**
     * A trigger the forced spell's resolution causes is not the spell's: live
     * gives it a bracket and a record of its own. Attended Knight's enters
     * trigger makes a Soldier, and the forced record holds the Knight's own
     * arrival and nothing of the token.
     */
    @Test
    void aForcedSpellsRecordLeavesOutTheTriggerItCaused() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card knight = TestBoard.inHand(game, caster, "Attended Knight");
        SpellAbility spell = knight.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String forced = forcedRecord(game, spell, 1L);
        String payload = payloadOf(forced);

        assertEquals(List.of(EventType.ZONE_CHANGE.wireName()), eventTypes(forced), payload);
        assertTrue(payload.contains("\"subjects\":[\"E" + knight.getId() + "\"],"
                + "\"params\":{\"from_zone\":\"stack\",\"to_zone\":\"battlefield\""), payload);
    }

    /**
     * A cast trigger resolves above the forced spell, before it, and is not
     * the spell's either: Monastery Swiftspear's prowess pump is not in the
     * forced Pyroclasm's record, though the damage it took is.
     */
    @Test
    void aForcedSpellsRecordLeavesOutItsCastTrigger() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card swiftspear = TestBoard.inPlay(game, caster, "Monastery Swiftspear");
        SpellAbility spell = TestBoard.inHand(game, caster, "Pyroclasm").getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String payload = payloadOf(forcedRecord(game, spell, 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"E"
                + swiftspear.getId() + "\"]"), payload);
        assertFalse(payload.contains(EventType.PT_CHANGE.wireName()), payload);
        assertFalse(payload.contains(EventType.KEYWORD_CHANGE.wireName()), payload);
    }

    /**
     * After the forced spell has resolved only the deaths of the state-based
     * check that follows join its record, as live holds them. Pyroclasm goes
     * to the graveyard once it has resolved, and Tarmogoyf grows for the
     * sorcery there: that stat change is Tarmogoyf's own static at work, not
     * Pyroclasm's effect. The deaths of the damaged Tarmogoyf and bears in
     * the check that follows are Pyroclasm's.
     */
    @Test
    void afterTheResolutionOnlyTheFollowingChecksDeathsJoin() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card goyf = TestBoard.inPlay(game, game.getPlayers().get(1), "Tarmogoyf");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility spell = TestBoard.inHand(game, caster, "Pyroclasm").getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String payload = payloadOf(forcedRecord(game, spell, 1L));

        assertFalse(payload.contains(EventType.PT_CHANGE.wireName()), payload);
        for (Card died : List.of(goyf, bears)) {
            assertTrue(payload.contains("\"subjects\":[\"E" + died.getId() + "\"],\"params\":"
                    + "{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""), payload);
        }
    }

    /**
     * A death in the state-based check between the cast and the resolution
     * is nobody's, forced as live (FR-029): no resolution is held and no
     * damage step open, so live counts it and files it nowhere, the fork
     * refuses it, and the two records agree. Maro counts its caster's hand,
     * and casting Pyroclasm, the hand's one card, leaves it 0/0 before
     * Pyroclasm resolves; its shrink is still the cast's.
     */
    @Test
    void aDeathInTheCheckBeforeTheResolutionIsNobodys() throws Throwable {
        Game observedGame = maroBoard();
        String observed = payloadOf(
                observedEffectHalf(observedGame, firstSpellInHand(observedGame)));

        Game forkedGame = maroBoard();
        Card maro = forkedGame.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Maro")).findFirst().orElseThrow();
        String forced = payloadOf(forcedRecord(forkedGame, firstSpellInHand(forkedGame), 1L));

        assertTrue(forced.contains("{\"type\":\"pt_change\",\"subjects\":[\"" + Refs.of(maro)
                + "\"]"), forced);
        assertFalse(forced.contains("\"subjects\":[\"" + Refs.of(maro)
                + "\"],\"params\":{\"from_zone\":\"battlefield\""), forced);
        assertEquals(observed, forced);
    }

    /**
     * A cast trigger above the forced spell owns what it does, and the
     * deaths of the check after it: Earthshaker deals 2 to each creature
     * without flying when an Arcane spell is cast, and the opposing bears it
     * kills are nowhere in the forced Lava Spike's record.
     */
    @Test
    void aCastTriggersDamageAndDeathsAreNotTheForcedSpells() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        TestBoard.inPlay(game, caster, "Earthshaker");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, caster, "Lava Spike");

        String payload = payloadOf(forcedRecord(game, firstSpellInHand(game), 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"P"), payload);
        assertFalse(payload.contains("\"E" + bears.getId() + "\""), payload);
    }

    /**
     * Nothing in the aftermath of an item above the forced spell is the
     * spell's, though the spell is the top of the stack again: it has not
     * begun to resolve, and live sends that aftermath to no record.
     * Managorger Hydra's cast trigger puts a +1/+1 counter on a Hydra that
     * carries a -1/-1 counter, above a forced Lava Spike, and the state-based
     * check after the trigger removes the pair: an outcome of that check
     * that is no death, and not Lava Spike's.
     */
    @Test
    void theAftermathOfACastTriggerIsNotTheForcedSpells() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        // The anthem keeps the 1/1 Hydra alive under its -1/-1 counter.
        TestBoard.inPlay(game, caster, "Glorious Anthem");
        Card hydra = TestBoard.inPlay(game, caster, "Managorger Hydra");
        hydra.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
        TestBoard.inHand(game, caster, "Lava Spike");

        String payload = payloadOf(forcedRecord(game, firstSpellInHand(game), 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"P"), payload);
        assertFalse(payload.contains("\"E" + hydra.getId() + "\""), payload);
    }

    /**
     * A death the forced Kavu's enters trigger causes is the trigger's, as
     * live files it: the Kavu's held window is closed by the time it
     * happens. The forced Kavu's record equals the observed Kavu's.
     */
    @Test
    void aDeathTheForcedKavusTriggerCausesIsNotTheKavus() throws Throwable {
        Game observedGame = kavuBoard();
        String observed = payloadOf(
                observedEffectHalf(observedGame, firstSpellInHand(observedGame)));

        Game forkedGame = kavuBoard();
        String forced = forcedRecord(forkedGame, firstSpellInHand(forkedGame), 1L);

        assertEquals(List.of(EventType.ZONE_CHANGE.wireName()), eventTypes(forced), forced);
        assertEquals(observed, payloadOf(forced));
    }

    /** Maro for the caster, whose hand is Pyroclasm alone, and the other seat's bears. */
    private static Game maroBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Maro");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Pyroclasm");
        return game;
    }

    /** Two creatures for the other seat and Flametongue Kavu in the caster's hand. */
    private static Game kavuBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Llanowar Elves");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Flametongue Kavu");
        return game;
    }

    private static SpellAbility firstSpellInHand(Game game) {
        Player caster = game.getPlayers().get(0);
        SpellAbility spell = caster.getCardsIn(ZoneType.Hand).get(0).getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        return spell;
    }

    /**
     * The first effect half an observed cast of {@code spell} writes, played
     * the way a game plays it: the check before priority, the resolution,
     * the check after it, and the triggers until the stack is empty.
     */
    private String observedEffectHalf(Game game, SpellAbility spell) throws Throwable {
        List<String> records = ShardRun.observed(tempDir.resolve("observed"), game, bracket -> {
            // A live collector has watched the board form, as the fork's sink
            // primes it: the stat change the cast's move to the stack makes
            // has a reading to differ from on both sides.
            bracket.rebaselineStats();
            game.getStack().addAndUnfreeze(spell);
            game.getAction().checkStateEffects(false);
            while (!game.getStack().isEmpty()) {
                game.getStack().resolveStack();
                game.getAction().checkStateEffects(false);
                game.getStack().addAllTriggeredAbilitiesToStack();
            }
            TestBoard.nextStep(game);
        });
        List<String> halves = records(records, "resolution");
        assertFalse(halves.isEmpty(), records.toString());
        return halves.get(0);
    }

    /** The one forced record. */
    private String forcedRecord(Game game, SpellAbility spell, long seed) throws IOException {
        ShardRun.Fork fork = ShardRun.forked(
                tempDir.resolve("fork-" + seed), game, forkCaps().interventions(1).probes(0).build(), seed, spell);
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        return records.get(0);
    }

    /** A Mountain for its caster, Grizzly Bears for the other seat, and Pyroclasm in hand. */
    private static Game pyroclasmBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Mountain");
        TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Pyroclasm");
        return game;
    }

    private static EffectEvent damageTo(List<EffectEvent> events, Card card) {
        return events.stream()
                .filter(e -> EventType.DAMAGE_DEALT.equals(e.type())
                        && e.subjects().contains("E" + card.getId()))
                .findFirst().orElseThrow(() -> new AssertionError(
                        "no damage to " + card + " in "
                                + events.stream().map(EffectEvent::toJson).toList()));
    }
    // ── a turn-conditioned line forced on the other player's turn (FR-057) ──

    /**
     * Zealous Display untaps its caster's creatures only when it is not the
     * caster's turn. Held by the player whose turn it is not, the scheduler
     * forces it at this boundary of the other player's turn, for its holder,
     * and the record shows the branch: the holder's tapped bears untap.
     */
    @Test
    void aTurnConditionedLineIsForcedForItsHolderOnTheOtherPlayersTurn() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, holder, "Grizzly Bears");
        bears.tap(true, null, null);
        TestBoard.inHand(game, holder, "Zealous Display");

        List<String> forced = scheduled(game, forkCaps().interventions(2).probes(0).build(), 1);

        assertEquals(1, forced.size(), forced.toString());
        assertEquals(Refs.of(holder), parse(forced.get(0)).get("actor_player"), forced.get(0));
        assertEquals(Boolean.TRUE, parse(forced.get(0)).get("interventional"), forced.get(0));
        assertTrue(eventTypes(forced.get(0)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(0)));
    }

    /**
     * The same line is forced again once its holder's own turn comes, where
     * it takes the other branch: the off-turn forcing and the on-turn one are
     * kept apart, so one game can show both.
     */
    @Test
    void theSameLineIsForcedAgainOnItsHoldersOwnTurnTakingTheOtherBranch() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, holder, "Grizzly Bears");
        bears.tap(true, null, null);
        TestBoard.inHand(game, holder, "Zealous Display");

        List<String> forced = scheduled(game, forkCaps().interventions(2).probes(0).build(), 1,
                () -> game.getPhaseHandler().devModeSet(PhaseType.MAIN1, holder));

        assertEquals(2, forced.size(), forced.toString());
        assertTrue(eventTypes(forced.get(0)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(0)));
        assertFalse(eventTypes(forced.get(1)).contains(EventType.UNTAPPED.wireName()),
                payloadOf(forced.get(1)));
    }

    /** A line whose text does not turn on whose turn it is is not forced off its holder's turn. */
    @Test
    void aLineWithNoTurnConditionIsNotForcedOnTheOtherPlayersTurn() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        TestBoard.inHand(game, holder, "Lightning Bolt");

        assertTrue(scheduled(game, forkCaps().interventions(2).probes(0).build(), 1).isEmpty());
    }

    /** Which lines turn on whose turn it is: a clause of the chain naming either parameter. */
    @Test
    void aTurnConditionAnywhereInTheChainMarksTheLine() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        assertTrue(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Fated Conflagration").getFirstSpellAbility()));
        assertTrue(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Zealous Display").getFirstSpellAbility()));
        assertFalse(TurnConditions.branchesOnWhoseTurn(
                TestBoard.inHand(game, holder, "Lightning Bolt").getFirstSpellAbility()));
    }

    /**
     * The scheduler's records over {@code boundaries} boundaries of {@code
     * game}, {@code between} run after each but the last, read from the shard.
     */
    private List<String> scheduled(Game game, CollectionCaps caps, int boundaries, Runnable... between)
            throws IOException {
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try {
            InterventionScheduler scheduler = new InterventionScheduler(game, new InterventionFork(
                    game, TestCollectors.context(shard, ShardRun.GAME_ID), new ForkBudget(caps), 1L));
            scheduler.collect();
            for (Runnable next : between) {
                next.run();
                scheduler.collect();
            }
        } finally {
            shard.close();
        }
        return Files.exists(shard.path()) ? ShardRun.readShard(shard.path()) : List.of();
    }

    // ── protection forks (FR-058) ───────────────────────────────────────

    /**
     * A protection fork grants a creature a rare protection through the
     * {@code keyword} operation and plays the copy to the end of the turn:
     * every record it writes is an interventional fork joined to the live
     * game, the board shows the granted line, and the grant is no event of
     * the turn. The fork's decisions are recorded too, as only a fork played
     * on has any.
     */
    @Test
    void aProtectionForkGrantsTheQualityAndPlaysTheTurnOn() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        Player active = game.getPlayers().get(0);
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        TestBoard.inPlay(game, game.getPlayers().get(1), "Goblin Piker");
        CollectionCaps caps = forkCaps().interventions(0).probes(0).everyDecision().build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        boolean kept;
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            kept = new ProtectionFork(game, new ForkBudget(caps), recording.playForward(), 7L)
                    .take(new ProtectionFork.Grant(bears, "Protection:Goblin"));
        } finally {
            shard.close();
        }
        List<String> forked = ShardRun.readShard(shard.path());

        assertTrue(kept, "the fork was kept");
        assertEquals(PhaseType.MAIN1, game.getPhaseHandler().getPhase(), "the live game did not move");
        assertTrue(forked.stream().anyMatch(record -> "playability".equals(parse(record).get("kind"))),
                "the fork played on, its decisions recorded");
        assertTrue(forked.stream().anyMatch(record -> record.contains("\"fragment\":\"Protection:Goblin\"")),
                "a fork record shows the granted protection");
        String grantedToBears = "{\"type\":\"keyword_change\",\"subjects\":[\"" + Refs.of(bears) + "\"]";
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            Object id = parsed.get("record_id");
            assertEquals(Boolean.TRUE, parsed.get("fork"), () -> "fork on " + id);
            assertEquals(Boolean.TRUE, parsed.get("interventional"), () -> "interventional on " + id);
            assertNull(parsed.get("directed"), () -> "directed on " + id);
            assertEquals(ShardRun.GAME_ID, parsed.get("game_id"), () -> "game_id on " + id);
            assertFalse(payloadOf(record).contains(grantedToBears), () -> "the grant is an event of " + id);
        }
        assertTrue(active.getCardsIn(ZoneType.Battlefield).contains(bears));
        assertTrue(bears.getKeywords().stream().noneMatch(k -> "Protection:Goblin".equals(k.getOriginal())),
                "the live creature gained nothing");
    }

    /**
     * The qualities a grant draws from are the rare ones of what the
     * creature can meet: the opponent's non-land permanents and hand, each
     * quality once, spelled as rule_outcomes.ProtectionQuality.fragment spells
     * them, then protection from everything. No colour, and nothing of the
     * opponent's lands or of the creature's own side.
     */
    @Test
    void aGrantDrawsFromTheRareQualitiesOfWhatTheCreatureCanMeet() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player own = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, own, "Grizzly Bears");
        TestBoard.inPlay(game, own, "Serra Angel");
        TestBoard.inPlay(game, opponent, "Goblin Piker");
        TestBoard.inPlay(game, opponent, "Forest");
        TestBoard.inHand(game, opponent, "Lightning Bolt");

        assertEquals(List.of(
                "Protection from each color",
                "Protection:Card.MonoColor:monocolored",
                "Protection:Card.cmcLE3:mana value 3 or less",
                "Protection:Creature",
                "Protection:Goblin",
                "Protection:Instant",
                "Protection:Warrior",
                "Protection from everything"), ProtectionFork.qualitiesFor(bears));
    }

    /**
     * One protection fork a game, only on a run taking interventions, and
     * only as a first main phase begins; a boundary that takes none spends
     * nothing.
     */
    @Test
    void aProtectionForkIsTakenOnceAGameOnARunTakingInterventions() {
        assertFalse(new ForkBudget(forkCaps().interventions(0).probes(2).build()).mayProtect());
        ForkBudget budget = new ForkBudget(forkCaps().interventions(2).probes(0).build());
        assertTrue(budget.mayProtect());

        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        assertFalse(new ProtectionFork(game, budget, null, 1L).collect(PhaseType.UPKEEP));
        assertTrue(budget.mayProtect(), "a boundary that took no fork spent nothing");

        budget.spendProtection();
        assertFalse(budget.mayProtect());
    }

    // ── directed forks (T090, D5) ───────────────────────────────────────

    /**
     * 704.5f's rule, for these tests only: as the end step begins with
     * Grizzly Bears on the battlefield, its toughness becomes 5.
     */
    private static DirectedRule tougherBears() {
        return DirectedRule.of("704.5f", "704.5f-1", live -> {
            if (live.getPhaseHandler().getPhase() != PhaseType.END_OF_TURN) {
                return Optional.empty();
            }
            return live.getCardsIn(ZoneType.Battlefield).stream()
                    .filter(card -> card.getName().equals("Grizzly Bears")).findFirst()
                    .map(bears -> Binding.empty().with("creature", bears));
        }, "tougher", new Application(ForkOperation.PT,
                binding -> new PtOperation(binding.card("creature"), null, 5)));
    }

    /** The live turn of the Sneak Attack board played out with the rule enabled; the shard's lines. */
    private List<String> directedTurn(CollectionCaps caps) throws Throwable {
        Game game = ForkPlayForwardTest.board();
        ForkPlayForwardTest.sneakAttack(game);
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(
                game, shard, ShardRun.GAME_ID, caps, 7L, null, ForkSeats.NONE,
                DirectedRules.of(tougherBears()), new DirectedForkTally())) {
            recording.open();
            ForkPlayForwardTest.playTheTurnOut(game);
        } finally {
            shard.close();
        }
        return ShardRun.readShard(shard.path());
    }

    private static List<String> split(List<String> all, boolean fork) {
        List<String> kept = new ArrayList<>();
        for (String record : all) {
            if (Boolean.valueOf(fork).equals(parse(record).get("fork"))) {
                kept.add(record);
            }
        }
        return kept;
    }

    /**
     * Every record of a directed fork names its one rule, is a
     * non-interventional fork and says which variant and operation it
     * applied; taken at a phase boundary, where no live record is open, it
     * mirrors none. The live game's records carry none of it on a
     * match-outcomes run.
     */
    @Test
    void aDirectedForksRecordsNameTheRuleTheVariantAndTheOperation() throws Throwable {
        List<String> all = directedTurn(
                forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build());
        List<String> forked = split(all, true);

        assertFalse(forked.isEmpty(), "the fork at the end step was kept and recorded");
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            assertEquals(List.of("704.5f"), parsed.get("aimed_at"), record);
            assertEquals(Boolean.FALSE, parsed.get("interventional"), record);
            assertEquals(Map.of("variant", "tougher", "operation", "pt"), parsed.get("directed"), record);
            assertNull(parsed.get("mirror_of"), record);
            assertEquals(ShardRun.GAME_ID, parsed.get("game_id"), record);
        }
        for (String record : split(all, false)) {
            assertFalse(parse(record).containsKey("aimed_at"), record);
            assertFalse(parse(record).containsKey("directed"), record);
        }
    }

    /**
     * The edit publishes a stats change on the fork's bus, but the fork's
     * collectors start listening after it: no fork record holds the
     * toughness change as an event the turn caused.
     */
    @Test
    void theEditItselfIsNoEventOfTheFork() throws Throwable {
        List<String> all = directedTurn(
                forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build());

        for (String record : split(all, true)) {
            assertFalse(eventTypes(record).contains("pt_change"), record);
        }
    }

    /**
     * On a run aimed at several rules (collect-variants), the live records
     * carry the run's list and a directed fork's records the one rule it was
     * taken for.
     */
    @Test
    void aDirectedForkOnAnAimedRunNamesItsOwnRule() throws Throwable {
        List<String> all = directedTurn(forkCaps().interventions(0).probes(0)
                .aimedAt("704.5f", "702.19b").directed(1, 1.0, "704.5f").build());

        assertFalse(split(all, true).isEmpty());
        for (String record : split(all, true)) {
            assertEquals(List.of("704.5f"), parse(record).get("aimed_at"), record);
        }
        for (String record : split(all, false)) {
            assertEquals(List.of("704.5f", "702.19b"), parse(record).get("aimed_at"), record);
        }
    }

    /**
     * Story 3 acceptance 1 on the registered 704.5f (T094): the active player
     * holds Disfigure and an untapped Swamp and the opponent's Drudge
     * Skeletons is a 1/1, so as the main phase begins the collector forks,
     * raises the Skeletons' toughness through {@code pt}, plays the fork to
     * the end of the turn (its end step sacrifices Sneak Attack's Hill Giant)
     * and writes the fork's records, each showing the 1/3 the fork made.
     */
    @Test
    void theRegisteredToughnessZeroRuleForksATougherCreatureToTheEndOfTheTurn() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        ForkPlayForwardTest.sneakAttack(game);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, active, "Swamp");
        TestBoard.inHand(game, active, "Disfigure");
        Card skeletons = TestBoard.inPlay(game, opponent, "Drudge Skeletons");
        CollectionCaps caps = forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            // The main phase's boundary, before either seat has had priority
            // in it: given priority earlier, the AI casts Disfigure at once.
            TestBoard.enterStep(game, PhaseType.MAIN1);
            ForkPlayForwardTest.playTheTurnOut(game);
        } finally {
            shard.close();
        }
        List<String> forked = split(ShardRun.readShard(shard.path()), true);

        assertFalse(forked.isEmpty(), "the fork was kept and recorded");
        boolean sawSkeletons = false;
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            assertEquals(List.of("704.5f"), parsed.get("aimed_at"), record);
            assertEquals(Map.of("variant", "tougher", "operation", "pt"), parsed.get("directed"), record);
            for (Object entity : (List<?>) RecordLines.asMap(parsed.get("state")).get("entities")) {
                Map<String, Object> fields = RecordLines.asMap(entity);
                if (("E" + skeletons.getId()).equals(fields.get("id"))) {
                    assertEquals(List.of(1L, 3L), fields.get("base_pt"), record);
                    sawSkeletons = true;
                }
            }
        }
        assertTrue(sawSkeletons, "a fork record shows the Skeletons");
    }

    /** {@code mirror_of} is written only when a live record was open at the fork point (D5). */
    @Test
    void aDirectedForkMirrorsALiveRecordOnlyWhenOneWasOpen() {
        List<String> held = new ArrayList<>();
        CollectorContext live = TestCollectors.context(ShardRun.writer(tempDir), ShardRun.GAME_ID, null);
        DirectedTag tag = new DirectedTag("tougher", ForkOperation.PT);
        CollectorContext opened = live.forFork(DirectedFork.flags("704.5f", tag, "run.0-l1.7"), held::add);
        CollectorContext none = live.forFork(DirectedFork.flags("704.5f", tag, null), held::add);

        opened.emit(opened.newRecord(RecordKind.COMBAT).actor("P0").state("{}").payload("{}"));
        none.emit(none.newRecord(RecordKind.COMBAT).actor("P0").state("{}").payload("{}"));

        assertEquals("run.0-l1.7", parse(held.get(0)).get("mirror_of"));
        assertNull(parse(held.get(1)).get("mirror_of"));
        assertEquals(List.of("704.5f"), parse(held.get(1)).get("aimed_at"));
    }
}
