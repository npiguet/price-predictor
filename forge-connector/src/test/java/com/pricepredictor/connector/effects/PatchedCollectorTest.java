package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.Contribution;
import forge.card.CardChangedType;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

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
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() {
        PatchedCollectors collector = unwired(caps(2000, 1.0));
        RecordDedup dedup = new RecordDedup();
        assertTrue(collector.allowContinuousRecord("11", "board-1"));
        assertTrue(collector.allowContinuousRecord("12", "board-1"));
        assertTrue(dedup.allowDistinct("continuous", "{anthem}", "{board}"));
        assertFalse(dedup.allowDistinct("continuous", "{anthem}", "{board}"));
    }

    // ── the snapshot depth is the same on every record ─────────────────

    @Test
    void everySnapshotCarriesAllFourTiers() {
        String json = recording(tempDir).triggers().record(
                triggerOfMode("ChangesZone"), Map.of(), true).toJson();

        assertTrue(json.contains("\"tiers\":[1,2,3,4]"), json);
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
     * final-fix-3.md item 2: {@code TriggerChannel.handler} had no game check at
     * all before this fix. A fork's own trigger evaluations
     * ({@code GameSimulator.resolveStack} runs the forced ability's triggers
     * through the real {@code TriggerHandler}) must not reach the live shard.
     */
    @Test
    void aTriggerEvaluationFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Paralyze");
        Trigger forkTrigger = forkCard.getCurrentState().getTriggers().get(0);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggers().handler().invoke(null, ON_CONDITION_EVALUATED,
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
                (bracket, collectors) -> collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
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
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
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
    // The trigger handler, playabilityHandler, combatLegalityHandler and
    // manaHandler are invoked nowhere in the test tree except the four
    // fork-rejection tests above -- the existing record(...)-style
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
                collectors.triggers().handler().invoke(null, ON_CONDITION_EVALUATED,
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
                (bracket, collectors) -> collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{liveCandidate}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"playability\"")),
                "a live playability candidate must reach the shard: " + lines);
    }

    @Test
    void aLiveAttackerLegalityAnswerReachesTheShard() throws Throwable {
        Player liveDefender = new Player("live-defender", TestCards.game(), 93101);
        Card candidate = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
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
}
