package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.Contribution;
import forge.card.CardChangedType;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.TestCollectors.caps;
import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static com.pricepredictor.connector.effects.TestCollectors.triggerOfMode;
import static com.pricepredictor.connector.effects.TestCollectors.unwired;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the patched collectors decide around the records they build: which
 * records coalesce, what a continuous contribution carries, when a held
 * probe branch is written, which hooks get installed, and which game a
 * trigger, playability, combat or mana report is admitted from.
 *
 * <p>The record builders have tests of their own: {@link TriggerRecordTest},
 * {@link RewriteContractTest} and {@link OutcomeHandlerTest}.
 */
@ExtendWith(ForgeExtension.class)
class PatchedCollectorTest {

    @TempDir
    Path tempDir;

    private static final Method ON_CONDITION_EVALUATED =
            HookShapes.method(HookShapes.TriggerFire.class, "onConditionEvaluated");
    private static final Method ON_CANDIDATE =
            HookShapes.method(HookShapes.Playability.class, "onCandidate");
    private static final Method ON_ATTACKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onAttackersDeclarable");
    private static final Method ON_BLOCKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onBlockersDeclarable");
    private static final Method ON_MANA_PRODUCED =
            HookShapes.method(HookShapes.Mana.class, "onManaProduced");

    // ── continuous coalescing ───────────────────────────────────────────

    /**
     * A static's record is keyed by the static and the board: one repeating
     * both is refused, and a change in either is a record of its own.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a static is recorded once per stable board, anthem, board-1, false",
            "a changed board gets a new record, anthem, board-2, true",
            "two statics on one board each get a record, crusade, board-1, true"})
    void aContinuousRecordIsKeyedByItsStaticAndBoard(
            String why, String staticKey, String board, boolean allowed) {
        PatchedCollectors collector = unwired(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertEquals(allowed, collector.allowContinuousRecord(staticKey, board), why);
    }

    @Test
    void continuousRecordsTakeNoNumericCap() {
        // Coalescing per stable board is itself the cap (FR-029).
        PatchedCollectors collector = unwired(caps(1, 1.0));
        for (int board = 0; board < 100; board++) {
            assertTrue(collector.allowContinuousRecord("anthem", "board-" + board));
        }
    }

    // ── continuous type and colour channels ─────────────────────────────

    /**
     * A stand-in for the patch's own {@code Card.CardColor}.
     *
     * <p>The real one is nested in {@code Card} and only exists as a public type
     * on a patched checkout, which this module does not compile against. The
     * collector reads both components reflectively, so any public record with
     * the same two accessors exercises the same path — and a test that named the
     * real type would not compile on a stock checkout.
     */
    public record ColorChange(ColorSet color, boolean additional) {
    }

    private static String tokens(Contribution into, String field) {
        String json = into.toJson();
        int at = json.indexOf("\"" + field + "\":[");
        return json.substring(at + field.length() + 4, json.indexOf(']', at));
    }

    @Test
    void anAddedTypeIsBareAndARemovedOneIsDashed() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new CardChangedType(
                new CardType(List.of("Creature", "Elf"), true),
                new CardType(List.of("Land"), true),
                false, Set.of()), into);
        assertEquals("\"creature\",\"elf\",\"-land\"", tokens(into, "types"));
    }

    @Test
    void removingAWholeClassOfTypeIsNamedForTheClass() {
        // "loses all creature types" names no type, so it cannot be spelled as
        // a list of removals.
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new CardChangedType(
                null, null, true,
                Set.of(RemoveType.CreatureTypes, RemoveType.SuperTypes)), into);
        assertEquals(
                Set.of("\"all-creature-types\"", "\"-all-creature-types\"",
                        "\"-all-super-types\""),
                Set.of(tokens(into, "types").split(",")));
    }

    @Test
    void aStateChangeMarksTheTypeLineAsSetRatherThanAdded() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(
                new StateChangedType(new CardType(List.of("Land"), true)), into);
        assertEquals("\"=\",\"land\"", tokens(into, "types"));
    }

    @Test
    void aTextChangeIsARemovalAndAnAdditionInOneEntry() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new WordChangedType("Forest", "Island"), into);
        assertEquals("\"-forest\",\"island\"", tokens(into, "types"));
    }

    @Test
    void anAdditionalColourIsJustTheLetter() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), true), into);
        assertEquals("\"G\"", tokens(into, "colors"));
    }

    @Test
    void aReplacingColourIsMarked() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), false), into);
        assertEquals("\"=\",\"G\"", tokens(into, "colors"));
    }

    @Test
    void replacingTheColourWithNothingIsColourless() {
        // An empty colour set and an empty token list are different answers:
        // one turns the permanent colourless, the other says nothing happened.
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromMask(0), false), into);
        assertEquals("\"=\",\"C\"", tokens(into, "colors"));
    }

    @Test
    void anUnreadableColourEntryContributesNothingRatherThanAReplacement() {
        // Guessing here would report every colour change as an overwrite.
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(new Object(), into);
        assertEquals("", tokens(into, "colors"));
    }

    @Test
    void aStaticThatWritesTheSameTypeTwiceSaysItOnce() {
        // One static can write to more than one type layer.
        Contribution into = new Contribution("E1");
        CardChangedType change = new CardChangedType(
                new CardType(List.of("Creature"), true), null, false, Set.of());
        PatchedCollectors.typeTokens(change, into);
        PatchedCollectors.typeTokens(change, into);
        assertEquals("\"creature\"", tokens(into, "types"));
    }

    // ── continuous name channel ──────────────────────────────────────────

    /** {@code "name"} is the last field {@link Contribution#toJson()} writes. */
    private static String nameField(Contribution into) {
        String json = into.toJson();
        int at = json.indexOf("\"name\":");
        return json.substring(at + "\"name\":".length(), json.length() - 1);
    }

    /**
     * The channel this task exists for: {@code contributions[].name} is
     * declared and listed in {@code KNOWN_CONSTANT_FIELDS} because it had
     * never carried a value in any corpus.
     *
     * <p>Goes through the real {@code Card.addChangedName} and the real,
     * reflectively-read {@code Card.getChangedCardNames} rather than a
     * hand-built stand-in, because the defect this guards against is exactly
     * the gap between "compiles against a plausible shape" and "the engine
     * ever produces this".
     *
     * <p>Two competing entries, not one: with a single rename in the table,
     * {@code card.getName()} -- which walks every entry and returns the last
     * overwrite -- answers the same string as that one static's own cell, so
     * a {@code contribution(...).name(card.getName())} implementation
     * (whole-card, not this cell) would pass unnoticed. The later static
     * ({@code winningStatic}) is what {@code getName()} reports; asserting
     * that the earlier one ({@code supersededStatic}) still carries its own,
     * superseded name is what only a per-cell read can produce. This also
     * covers two statics renaming the same card, which otherwise had no test
     * at all.
     */
    @Test
    void aChangedNameContributesTheNewName() {
        Card card = TestCards.build("Grizzly Bears");
        long supersededStatic = 1111L;
        long winningStatic = 2222L;
        card.addChangedName("Superseded-By-Later-Static", false, 3L, supersededStatic);
        card.addChangedName("Winning-Name", false, 7L, winningStatic);
        // The premise the whole-card shortcut would rely on, and get wrong
        // for the superseded static below.
        assertEquals("Winning-Name", card.getName());

        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        PatchedCollectors.nameContributions(card, byStatic, "E1");

        assertEquals("\"Superseded-By-Later-Static\"",
                nameField(byStatic.get(supersededStatic).get("E1")));
        assertEquals("\"Winning-Name\"",
                nameField(byStatic.get(winningStatic).get("E1")));
    }

    /**
     * The channel's other half: most statics contribute no name, and null is
     * the correct, meaningful answer there -- not an empty string, and not a
     * dropped key.
     */
    @Test
    void aCardWithNoNameChangeRendersNullName() {
        Card card = TestCards.build("Grizzly Bears");
        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        PatchedCollectors.nameContributions(card, byStatic, "E1");
        assertTrue(byStatic.isEmpty(), "no static wrote a name, so nothing is filed");
        assertEquals("null", nameField(new Contribution("E1")));
    }

    @Test
    void anUnreadableNameEntryContributesNoName() {
        // Guessing here would report every unrelated object as a rename.
        Contribution into = new Contribution("E1");
        PatchedCollectors.nameTokens(new Object(), into);
        assertEquals("null", nameField(into));
    }

    // ── coalescing the kinds that had none ─────────────────────────────

    /**
     * A decision is keyed by what it says and the board it was asked on: the
     * same answer on an unchanged board is written once, and a board that
     * moved gets its own record.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a decision on an unchanged board is written once, '{life:20}', false",
            "a board that moved gets its own decision record, '{life:19}', true"})
    void aDecisionIsKeyedByItsPayloadAndBoard(String why, String board, boolean allowed) {
        PatchedCollectors collector = unwired(caps(2000, 1.0));
        assertTrue(collector.allowDistinctRecord("decision", "{payload}", "{life:20}"));
        assertEquals(allowed, collector.allowDistinctRecord("decision", "{payload}", board), why);
    }

    /**
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() {
        PatchedCollectors collector = unwired(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("11", "board-1"));
        assertTrue(collector.allowContinuousRecord("12", "board-1"));
        assertTrue(collector.allowDistinctRecord("continuous", "{anthem}", "{board}"));
        assertFalse(collector.allowDistinctRecord("continuous", "{anthem}", "{board}"));
    }

    /**
     * The legality cap keys on the snapshot as well as the payload (FR-028).
     *
     * <p>The AI's re-asks within one combat still collapse -- the board does
     * not move while it evaluates, so their snapshots render identically --
     * while the same answer on a board that moved is kept, which the
     * payload-only key of gen-1 dropped as a repeat (spec Story 3 scenario 10).
     */
    @Test
    void aLegalityAnswerCoalescesOnPayloadAndSnapshotTogether() {
        PatchedCollectors collector = unwired(caps(2000, 1.0));
        assertTrue(collector.allowDistinctRecord("blockers", "{payload}", "{board-1}"));
        assertFalse(collector.allowDistinctRecord("blockers", "{payload}", "{board-1}"));
        assertTrue(collector.allowDistinctRecord("blockers", "{payload}", "{board-2}"));
        assertTrue(collector.allowDistinctRecord("attackers", "{payload}", "{board-1}"));
    }

    // ── the snapshot depth is the same on every record ─────────────────

    @Test
    void everySnapshotCarriesAllFourTiers() {
        String json = recording(tempDir).triggerRecord(
                triggerOfMode("ChangesZone"), Map.of(), true).toJson();

        assertTrue(json.contains("\"tiers\":[1,2,3,4]"), json);
    }

    // ── held probe branches ────────────────────────────────────────────

    private static ForkCollector.HeldProbe branch(DamageStep substep) {
        return new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", CombatShape.empty().fields(), substep,
                ForkCollector.TURN_UNKNOWN);
    }

    private PatchedCollectors withForks(RecordShardWriter writer) {
        CollectionCaps probing = new CollectionCaps(
                1, 1.0, 2, 2, List.of("trample"), 0.0, 0.0);
        Game game = TestCards.game();
        return new PatchedCollectors(
                game, TestCollectors.context(writer, "run.0-l1.0"), probing, 1L,
                TestCollectors.bracket(game, writer, "run.0-l1.0"),
                new ForkCollector(null, TestCollectors.context(writer, "run.0-l1.0"), probing, 1L));
    }

    /**
     * A branch is completed only against its own damage step.
     *
     * <p>In the first corpus it was not: a branch forked at the first-strike
     * step, where nothing was ever assigned, survived that step and was written
     * against the regular step's record — so the sampled fork record whose phase
     * was {@code combat_first_strike_damage} carried a {@code mirror_of} whose
     * phase was {@code combat_damage}, and gate 2's difference compared a step
     * that did not happen with one that did.
     */
    @Test
    void aBranchIsCompletedOnlyAgainstItsOwnDamageStep() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE));
            collector.hold(branch(DamageStep.REGULAR));

            collector.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR);
            // The first-strike branch must still be waiting for its own step.
            collector.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE);
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.10\""), lines.get(1));
    }

    /** With no substep named, the oldest held step is the one being written. */
    @Test
    void theOldestHeldStepIsTheOneACombatRecordCompletes() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE));
            collector.hold(branch(DamageStep.REGULAR));

            collector.writeHeldProbes("run.0-l1.9");
            // One branch is left, and it is the regular step's.
            collector.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE);
            collector.writeHeldProbes("run.0-l1.11", DamageStep.REGULAR);
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.11\""), lines.get(1));
    }

    /** A branch whose step produced no record is dropped rather than mispaired. */
    @Test
    void aBranchFromAStepThatWroteNoRecordIsDropped() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE));

            assertEquals(1, collector.discardStaleProbes(DamageStep.REGULAR));
            collector.writeHeldProbes("run.0-l1.9", DamageStep.FIRST_STRIKE);
        }

        assertFalse(Files.exists(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)));
    }

    // ── installing ──────────────────────────────────────────────────────

    /**
     * Exactly the listener hooks the checkout offers get installed, and no
     * others -- and none at all in degraded mode, where install() declines
     * before it looks.
     *
     * <p>Asserted against the checkout rather than against a fixed count: the
     * sibling Forge is patched or not independently of this repository. The
     * expected set is derived from {@link PatchHooks#REQUIRED}, so a listener
     * hook added there is expected here without editing the test. The confirm
     * listener is excluded because the recording installs it with the
     * bracket's handler, not this class.
     */
    @Test
    void everyHookTheCheckoutOffersIsInstalledAndNoOthers() {
        long expected = AttributionMode.detect() == AttributionMode.DEGRADED
                ? 0
                : PatchHooks.REQUIRED.stream()
                        .filter(hook -> hook.method().startsWith("setEffectRecord")
                                && hook.method().endsWith("Listener")
                                && !PatchHooks.PLAYER_CONTROLLER_AI.equals(hook.owner()))
                        .filter(PatchHooks.Hook::present)
                        .count();
        try (PatchedCollectors collector = unwired(CollectionCaps.defaults())) {
            assertEquals(expected, collector.install());
        }
    }

    // ── final-fix-3.md item 2: the other four handlers ───────────────────

    /**
     * final-fix-3.md item 2: {@code triggerFireHandler} had no game check at
     * all before this fix. A fork's own trigger evaluations
     * ({@code GameSimulator.resolveStack} runs the forced ability's triggers
     * through the real {@code TriggerHandler}) must not reach the live shard.
     */
    @Test
    void aTriggerEvaluationFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Paralyze");
        Trigger forkTrigger = forkCard.getCurrentState().getTriggers().get(0);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggerFireHandler().invoke(null, ON_CONDITION_EVALUATED,
                        new Object[]{forkTrigger, Map.of(), true}));

        assertFalse(Files.exists(shard),
                "a fork's own trigger evaluation must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code playabilityHandler} had no game check
     * either. {@code caps(0, 1.0)} forces {@code playabilityRate} to 1.0 so
     * the sampling gate ahead of the game check cannot flakily hide it.
     */
    @Test
    void aPlayabilityCandidateFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{forkCandidate}));

        assertFalse(Files.exists(shard),
                "a fork's own playability candidate must not land in the live shard");
    }

    /**
     * {@code combatLegalityHandler} checks the game the defender belongs to.
     * The defender is the acting {@code GameEntity} this hook hands over; the
     * candidate/legal lists may be any real card, since only the defender's
     * own game identity is what the check reads.
     */
    @Test
    void anAttackerLegalityAnswerFromAnotherGamesDefenderDoesNotReachTheShard() throws Throwable {
        Player forkDefender = new Player("fork-defender", TestBoard.anotherGame(), 92301);
        Card candidate = TestCards.build("Grizzly Bears");

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{null, forkDefender, List.of(candidate), List.of(candidate)}));

        assertFalse(Files.exists(shard), "another game's defender must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code manaHandler} had no game check either.
     * Held in {@link PatchedCollectors#manaReservoir} rather than written
     * immediately, so the collector must be closed (which flushes it) before
     * the shard can be read.
     */
    @Test
    void manaProducedByAnotherGamesAbilityDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Llanowar Elves");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", forkCard);
        Player forkPlayer = new Player("fork-player", forkCard.getGame(), 92302);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.manaHandler().invoke(null, ON_MANA_PRODUCED,
                        new Object[]{forkAbility, forkPlayer, "G"}));

        assertFalse(Files.exists(shard),
                "a fork's own mana production must not land in the live shard");
    }

    // ── final-fix-4.md item 3: a positive test behind each of those gates ──
    //
    // triggerFireHandler, playabilityHandler, combatLegalityHandler and
    // manaHandler are invoked nowhere in the test tree except the four
    // fork-rejection tests above -- the existing triggerRecord(...)-style
    // tests call the record builder directly and never reach the handler
    // (InvocationHandler) these gates actually live in. A gate with only a
    // negative test behind it can fail closed for everything, live included,
    // and every one of those tests would still pass. rewriteHandler and
    // confirmHandler have positive coverage in RewriteContractTest and
    // BusBracketCollectorTest respectively, which drive the live handler
    // directly, so only these four need one here.

    @Test
    void aLiveTriggerEvaluationReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Paralyze");
        Trigger liveTrigger = liveCard.getCurrentState().getTriggers().get(0);

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggerFireHandler().invoke(null, ON_CONDITION_EVALUATED,
                        new Object[]{liveTrigger, Map.of(), true}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"trigger\"")),
                "a live trigger evaluation must reach the shard: " + lines);
    }

    @Test
    void aLivePlayabilityCandidateReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);

        // playabilityRate forced to 1.0, matching final-fix-4.md item 5's own
        // fix: at the default 0.1 this record would be sampled out at random
        // roughly nine times in ten, and a flaky pass here is exactly as
        // uninformative as the vacuous one item 3 exists to close.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{liveCandidate}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"playability\"")),
                "a live playability candidate must reach the shard: " + lines);
    }

    @Test
    void aLiveAttackerLegalityAnswerReachesTheShard() throws Throwable {
        Player liveDefender = new Player("live-defender", TestCards.game(), 93101);
        Card candidate = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{null, liveDefender, List.of(candidate), List.of(candidate)}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"subkind\":\"attackers\"")),
                "a live defender's legality answer must reach the shard: " + lines);
    }

    @Test
    void aLiveManaProductionReachesTheShard() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player livePlayer = new Player("live-player", TestCards.game(), 93102);

        // manaHandler holds records in the reservoir rather than writing them
        // immediately (PatchedCollectors#manaReservoir); only flushMana(),
        // called when the run closes the collectors, writes them out.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.manaHandler().invoke(null, ON_MANA_PRODUCED,
                        new Object[]{liveAbility, livePlayer, "G"}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("mana_produced")),
                "live mana production must reach the shard: " + lines);
    }

    /**
     * final-fix-4.md item 5: a fork candidate rejected on identity must not
     * have drawn from the shared {@code sampler} on its way to being
     * rejected -- checking identity before the {@code caps.playabilityRate()}
     * draw is what this pins, not just what the diff shows.
     *
     * <p>Both collectors share one seed (1) and one {@code playabilityRate}
     * (0.5). The sampler is <em>not</em> {@code new Random(1L)}: it is
     * {@code new Random(scramble(seed))}, and {@code scramble} is SplitMix64's
     * finalizer, so the draws are those of {@code Random(scramble(1L))} --
     * {@code 0.9245, 0.1142}, not {@code Random(1L)}'s {@code 0.7309, 0.4101}.
     * An earlier version of this comment cited the latter; the conclusion held
     * either way, which is exactly why the wrong constants survived. Do not
     * try to reproduce them against the real {@code sampler}.
     *
     * <p>A solo live candidate draws only the first (0.9245 &gt; 0.5, sampled
     * out). If the identity check ran <em>after</em> the draw, the fork
     * candidate ahead of it would consume that first draw before being dropped
     * on identity, and the live candidate behind it would draw the second
     * (0.1142 &le; 0.5, sampled in) -- a live record appearing only because a fork candidate
     * happened to precede it. Checked first, the fork candidate never reaches
     * the sampler at all, and the live candidate's own draw -- and therefore
     * its own outcome -- is identical whether or not the fork one ran ahead
     * of it.
     */
    @Test
    void aRejectedForkCandidateDoesNotShiftTheLiveSampleStream() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);
        CollectionCaps halfRate = new CollectionCaps(
                CollectionCaps.defaults().manaCap(), 0.5,
                CollectionCaps.defaults().interventionsPerGame(), CollectionCaps.defaults().probesPerGame(),
                CollectionCaps.defaults().probeKeywords(), 0.0, 0.0);

        Path solo = ShardRun.shardOf(tempDir.resolve("solo"), TestCards.game(), halfRate,
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{liveCandidate}));
        boolean soloAdmitted = admitsAPlayabilityRecord(solo);

        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);
        Path paired = ShardRun.shardOf(tempDir.resolve("paired"), TestCards.game(), halfRate,
                (bracket, collectors) -> {
                    collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                            new Object[]{forkCandidate});
                    collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                            new Object[]{liveCandidate});
                });
        boolean pairedAdmitted = admitsAPlayabilityRecord(paired);

        assertEquals(soloAdmitted, pairedAdmitted,
                "a rejected fork candidate ahead of it must not change whether the "
                        + "live candidate behind it is sampled in: solo=" + soloAdmitted
                        + " paired=" + pairedAdmitted);
    }

    /** Whether a run sampled a playability record in; one that sampled none wrote no shard. */
    private static boolean admitsAPlayabilityRecord(Path shard) throws IOException {
        return Files.exists(shard) && ShardRun.readShard(shard).stream()
                .anyMatch(l -> l.contains("\"kind\":\"playability\""));
    }

    // ── gen-2: declarations and the deciding player ─────────────────────
    //
    // The engine reports legality from the turn-based declaration itself and
    // names the deciding player, so every record is a real declaration
    // (what_if=false) whatever the step, and FR-030a's actor is the player
    // the engine names; FR-030 stamps random_seat off that actor. These
    // build a two-player game so the deciding player is a real seat.

    private static List<String> recordsOf(List<String> lines) {
        return lines.stream().filter(l -> l.contains("\"kind\":\"playability\"")).toList();
    }

    /**
     * An attackers record is a real declaration made by the player the
     * engine names, and a candidate the rules keep out is listed as
     * forbidden rather than left out.
     */
    @Test
    void anAttackDeclarationIsRealAndNamesTheDeciderTheEngineReports() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card ready = TestBoard.inPlay(game, active, "Grizzly Bears");
        Card tapped = TestBoard.inPlay(game, active, "Runeclaw Bear");
        tapped.setTapped(true);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(ready, tapped), List.of(ready)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"attackers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(active) + "\""),
                "the deciding player is the one the engine names: " + real);
        assertTrue(real.contains("\"random_seat\":false"), real);
        assertTrue(real.contains("\"legal_attackers\":[\"" + SnapshotBuilder.entityId(ready) + "\"]"),
                real);
        assertTrue(real.contains("\"forbidden\":[{\"entity\":\""
                + SnapshotBuilder.entityId(tapped) + "\""), real);
    }

    /**
     * Nothing about the step or whose turn it is reclassifies a declaration:
     * the engine reports only real ones, so a report outside the
     * declare-attackers step is still written as real.
     */
    @Test
    void aDeclarationIsRealWhateverTheStep() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{other, active, List.of(theirs), List.of(theirs)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"what_if\":false"), records.get(0));
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + SnapshotBuilder.playerId(other) + "\""), records.get(0));
    }

    /**
     * FR-030a's blockers row: a blockers record names the blocking player
     * the engine reports, with the attacker it anchors on and how many
     * blockers that attacker takes.
     */
    @Test
    void aBlockDeclarationIsRealAndNamesTheBlockingPlayer() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                    new Object[]{blocking, attacker, List.of(blocker), List.of(blocker), 1});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"blockers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(blocking) + "\""),
                "the deciding player is the blocking player: " + real);
        assertTrue(real.contains("\"anchor_attacker\":\"" + SnapshotBuilder.entityId(attacker) + "\""),
                real);
        assertTrue(real.contains("\"min_blockers\":1"), real);
    }

    /**
     * Every declaration on a board that moved is written, and the same
     * declaration on an unchanged board is written once (FR-028).
     */
    @Test
    void everyDeclarationOnAMovedBoardIsWrittenAndARepeatIsNot() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            for (int turn = 0; turn < 20; turn++) {
                // A life total that moves is enough to make each snapshot new.
                active.setLife(20 - turn, null);
                collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{active, other, List.of(mine), List.of(mine)});
            }
            collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        assertEquals(20, recordsOf(lines).size());
    }

    /**
     * Spec Story 3 scenario 6 on the hook path: a record whose actor is the
     * random seat says so.
     */
    @Test
    void randomSeatIsStampedOffTheActor() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.recordsWithRandomSeat(tempDir, game,
                SnapshotBuilder.playerId(active), (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"random_seat\":true"), records.get(0));
    }

    /**
     * FR-030a's decision row: a candidate evaluated on the opponent's turn
     * names the candidate's controller, not the active player.
     */
    @Test
    void aDecisionRecordNamesTheCandidatesController() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");
        SpellAbility candidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", theirs);
        candidate.setActivatingPlayer(other);

        List<String> lines = ShardRun.records(tempDir, game, caps(0, 1.0), (bracket, collectors) -> {
            collectors.playabilityHandler().invoke(null, ON_CANDIDATE,
                    new Object[]{candidate});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + SnapshotBuilder.playerId(other) + "\""), records.get(0));
        assertFalse(records.get(0).contains("what_if"),
                "a decision record carries no what_if: " + records.get(0));
    }
}
