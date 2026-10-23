package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
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
import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
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
class ForkCollectorTest {

    @TempDir
    Path tempDir;

    private static CollectionCaps caps(
            int interventions, int probes, List<String> keywords) {
        return new CollectionCaps(
                2000, 0.1, interventions, probes, keywords, 0.0, 0.0);
    }

    private RecordShardWriter writer;
    /** The budget the last fork built here spends. */
    private ForkBudget budget;

    private CollectorContext context(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        budget = new ForkBudget(caps);
        return TestCollectors.context(writer, "run.0-l1.0");
    }

    private InterventionFork interventions(CollectionCaps caps) {
        return new InterventionFork(null, context(caps), budget, 42L);
    }

    private DamageStepProbe probes(CollectionCaps caps) {
        return new DamageStepProbe(null, context(caps), budget);
    }

    /**
     * The fork's shard, closed so every record it wrote is on disk. The
     * writer creates the file with its first record, so a fork that wrote
     * nothing leaves none.
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
    private DamageStepProbe probesOverAGame(CollectionCaps caps) {
        return new DamageStepProbe(TestCards.game(), context(caps), budget);
    }

    // ── the per-game budget ─────────────────────────────────────────────

    @Test
    void aFreshGameMayIntervene() {
        assertTrue(new ForkBudget(caps(2, 2, List.of())).mayIntervene(1));
    }

    @Test
    void anExhaustedInterventionBudgetStopsFurtherForks() {
        assertFalse(new ForkBudget(caps(0, 2, List.of())).mayIntervene(1));
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        InterventionFork collector = interventions(caps(100, 2, List.of()));
        collector.intervene(null, 7);
        collector.intervene(null, 7);
        assertFalse(collector.mayIntervene(7));
        assertTrue(collector.mayIntervene(8));
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. A null game makes every copy fail.
        InterventionFork collector = interventions(caps(1, 2, List.of()));
        assertFalse(collector.intervene(null, 1));
        assertFalse(collector.mayIntervene(1));
    }

    @Test
    void aFailedForkWritesNoRecord() {
        InterventionFork collector = interventions(caps(2, 2, List.of()));
        collector.intervene(null, 1);
        assertFalse(Files.exists(closedShard()));
    }

    // ── the probe switch ────────────────────────────────────────────────

    @Test
    void noProbeIsTakenWithTheFlagUnset() {
        // Whatever the build state: the flag is the runtime switch, and it is
        // deliberately separate from the build decision gate 2 drives.
        assertFalse(new ForkBudget(caps(2, 2, List.of())).mayProbe("wither"));
    }

    @Test
    void onlyTheNamedKeywordsAreProbed() {
        ForkBudget budget = new ForkBudget(caps(2, 2, List.of("wither")));
        assertTrue(budget.mayProbe("wither"));
        assertFalse(budget.mayProbe("lifelink"));
    }

    @Test
    void theProbeBudgetIsSpentByTakingProbes() {
        // Spent even by a fork that fails: a null game makes every copy fail.
        DamageStepProbe collector = probes(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("wither", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
        assertFalse(budget.mayProbe("wither"));
    }

    @Test
    void aProbeForAnUnnamedKeywordSpendsNothing() {
        DamageStepProbe collector = probes(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("lifelink", TestCards.build("Grizzly Bears"), DamageStep.REGULAR));
        assertTrue(budget.mayProbe("wither"));
        assertFalse(Files.exists(closedShard()));
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
        DamageStepProbe collector = probesOverAGame(caps(2, 2, List.of("trample")));

        assertFalse(collector.writeHeldProbe(branch(7), "run.0-l1.9"),
                "a branch seven turns older than its mirror is not a pair");
        assertFalse(Files.exists(closedShard()));
    }

    /** A branch from the turn the record was written on is written. */
    @Test
    void aBranchFromThisTurnIsWritten() throws IOException {
        DamageStepProbe collector = probesOverAGame(caps(2, 2, List.of("trample")));

        assertTrue(collector.writeHeldProbe(branch(0), "run.0-l1.9"));
        assertEquals(1, ShardRun.readShard(closedShard()).size());
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
        DamageStepProbe collector = probesOverAGame(caps(2, 2, List.of("trample")));

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

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        DamageStepProbe.HeldProbe held;
        try {
            DamageStepProbe forks = new DamageStepProbe(
                    game, TestCollectors.context(writer, "run.0-l1.0"),
                    new ForkBudget(caps(0, 1, List.of("trample"))));
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
                caps(1, 0, List.of()), 1L, firstSpellInHand(forkedGame));
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
     * is the forced spell's, as live files it into the spell's bracket. Maro
     * counts its caster's hand, and casting Pyroclasm, the hand's one card,
     * leaves it 0/0 before Pyroclasm resolves.
     */
    @Test
    void aDeathInTheCheckBeforeTheResolutionIsTheForcedSpells() throws Throwable {
        Game observedGame = maroBoard();
        String observed = payloadOf(
                observedEffectHalf(observedGame, firstSpellInHand(observedGame)));

        Game forkedGame = maroBoard();
        Card maro = forkedGame.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Maro")).findFirst().orElseThrow();
        String forced = payloadOf(forcedRecord(forkedGame, firstSpellInHand(forkedGame), 1L));

        assertTrue(forced.contains("\"subjects\":[\"" + SnapshotBuilder.entityId(maro)
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""),
                forced);
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
            bracket.primeBattlefield();
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
                tempDir.resolve("fork-" + seed), game, caps(1, 0, List.of()), seed, spell);
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
}
