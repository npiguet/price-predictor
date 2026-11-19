package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.phase.PhaseType;
import forge.game.player.GameLossReason;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a bus record has to say, and the wiring that makes it say it.
 *
 * <p>Each of these is a wiring property rather than a format one: the cost
 * half's {@code outcome} is what became of the spell, and its {@code link_id}
 * always has a partner; a move onto the stack is no {@code zone_change}, and
 * no event in a record repeats another; {@code combat} and {@code resolution}
 * records are snapshotted as deep as every other kind; and {@code refs.modes}
 * is read at the resolution, where it is set.
 *
 * <p>And three more: a bus-derived event names what caused it, so two
 * attackers each dealing 1 damage to the same player render apart; the end of
 * the game writes what it would otherwise cut off — the combat that decided
 * the game, and a count of the casts still on the stack; and {@code declined}
 * is written when the controller refuses an optional effect, which otherwise
 * reads exactly like one that did something.
 *
 * <p>The bus itself is not exercised here — driving Forge's event bus needs a
 * running match, which is the integration test's job. What is exercised is
 * everything below the two lines that read the stack, against real abilities
 * built by {@code AbilityFactory} from real cards, which is where all of these
 * properties live.
 */
@ExtendWith(ForgeExtension.class)
class BusBracketCollectorTest {

    @TempDir
    Path tempDir;

    private RecordShardWriter writer;

    private BusBracketCollector collector() {
        return collector(TestCards.game());
    }

    private BusBracketCollector collector(Game game) {
        writer = ShardRun.writer(tempDir);
        return TestCollectors.bracket(game, writer, ShardRun.GAME_ID);
    }

    /** The records written so far, one parsed line each. */
    private List<String> written() throws IOException {
        return ShardRun.readShard(ShardRun.closed(writer));
    }

    /** A targeted burn spell, the shape a partial fizzle happens to. */
    private static SpellAbility damageAbility() {
        Card host = TestCards.build("Lightning Bolt");
        return AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose"
                        + " | NumDmg$ 3", host);
    }

    /** A collector that has seen a bolt cast and resolve; returns the bolt. */
    private SpellAbility resolveBolt(BusBracketCollector collector) {
        SpellAbility bolt = damageAbility();
        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();
        return bolt;
    }

    /** The listener the patched {@code PlayerControllerAi} installs, reading {@code collector}. */
    private static HookShapes.Confirm confirmListener(BusBracketCollector collector) {
        return HookShapes.proxy(HookShapes.Confirm.class, collector.confirmHandler());
    }

    // ── outcome ─────────────────────────────────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    // ── snapshot depth ──────────────────────────────────────────────────

    /**
     * Both halves of a resolution carry all four tiers.
     *
     * <p>Checked where the vector lands rather than where it is decided. A depth
     * hardcoded per collector made {@code state.tiers} say which collector wrote
     * a record — {@code combat} and {@code resolution} at {@code [1,2]} against
     * {@code [1,2,3]} everywhere else — which is collection metadata the schema
     * keeps out of a model's inputs.
     */
    @Test
    void everyRecordCarriesAllFourTiers() throws IOException {
        resolveBolt(collector());

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        for (String record : records) {
            assertTrue(record.contains("\"tiers\":[1,2,3,4]"), record);
        }
    }

    // ── resolution-time refs ────────────────────────────────────────────

    /**
     * The effect half's refs are read when the bracket closes, not at the cast.
     *
     * <p>A charm's chosen mode, an announced X and a named card are set by the
     * engine while the ability resolves. Read at the cast they are all absent;
     * read here they are there, on a board that is still the pre-event one.
     */
    @Test
    void theEffectHalfCarriesRefsReadAtResolution() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        Card victim = TestCards.build("Grizzly Bears");

        collector.beginBracket(bolt);
        // The engine chooses during resolution; this stands in for that.
        bolt.resetTargets();
        bolt.getTargets().add(victim);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertFalse(records.get(0).contains("\"targets\":[\"E" + victim.getId() + "\"]"),
                "the cost half describes the cast, before any of this existed");
        assertTrue(records.get(1).contains("\"targets\":[\"E" + victim.getId() + "\"]"),
                records.get(1));
    }

    // ── what caused it ──────────────────────────────────────────────────

    /**
     * A damage event names the creature that dealt it, not the open bracket.
     *
     * <p>The source is on the bus event. It is what makes two attackers each
     * dealing 1 damage to the same blocker two events rather than one line
     * printed twice — and, on its own, the answer to "which permanent caused
     * this damage", which nothing else in the record carries.
     */
    @Test
    void damageNamesThePermanentThatDealtIt() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        Card attacker = TestCards.build("Grizzly Bears");
        Card victim = TestCards.build("Runeclaw Bear");

        collector.beginBracket(bolt);
        collector.onCardDamaged(new GameEventCardDamaged(
                victim.getView(), attacker.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        String effectHalf = written().get(1);
        assertTrue(effectHalf.contains("\"cause\":\"E" + attacker.getId() + "\""),
                effectHalf);
        assertFalse(
                effectHalf.contains("\"cause\":\"E" + bolt.getHostCard().getId() + "\""),
                "the bus named a source, and it outranks the bracket");
    }

    /**
     * An outcome the bus names no source for takes the resolving card.
     *
     * <p>Most of the channel: a zone change, a life total, a counter, a tap.
     * The engine names nothing on those events, and the honest answer is the
     * card whose line was resolving when they arrived — the same value the
     * trait-derived side writes for a {@code SpellAbility} cause, so a
     * {@code zone_change} from a replacement map and one from the bus name the
     * same card in the same spelling.
     */
    @Test
    void anOutcomeWithNoNamedSourceNamesTheResolvingCard() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.onCardChangeZone(new GameEventCardChangeZone(
                TestCards.build("Grizzly Bears"),
                zone(ZoneType.Battlefield), zone(ZoneType.Graveyard)));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        String effectHalf = written().get(1);
        assertTrue(
                effectHalf.contains("\"cause\":\"E" + bolt.getHostCard().getId() + "\""),
                effectHalf);
    }

    /** A damage event with no source falls back to the bracket like the rest. */
    @Test
    void damageFromNowhereStillNamesTheResolvingCard() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Runeclaw Bear").getView(), null, 1,
                GameEventCardDamaged.DamageType.Normal));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        String effectHalf = written().get(1);
        assertTrue(
                effectHalf.contains("\"cause\":\"E" + bolt.getHostCard().getId() + "\""),
                effectHalf);
    }

    /**
     * Radiation names the player who irradiated this one, not the open bracket.
     *
     * <p>{@code GameEventPlayerRadiation} carries a {@code source} the same way
     * {@code GameEventPlayerPoisoned} does, one line away in
     * {@code Player.setCounters} from the same local — but {@code onRadiation}
     * was calling the single-arg {@code record()}, so every radiation event
     * fell back to the bracket's generic cause instead of the irradiating
     * player Forge actually names.
     */
    @Test
    void radiationNamesThePlayerWhoCausedIt() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        Player source = new Player("irradiator", TestCards.game(), 501);
        Player receiver = new Player("irradiated", TestCards.game(), 502);

        collector.beginBracket(bolt);
        collector.onRadiation(new GameEventPlayerRadiation(receiver, source, 3));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        String effectHalf = written().get(1);
        assertTrue(effectHalf.contains("\"cause\":\"P" + source.getId() + "\""),
                effectHalf);
        assertFalse(
                effectHalf.contains("\"cause\":\"E" + bolt.getHostCard().getId() + "\""),
                "the bus named a source, and it outranks the bracket");
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * A cast still on the stack when the game ends had its cost half written
     * at its cast, and nothing more is: its effect half never comes.
     */
    @Test
    void aCastStillOnTheStackAtTheEndWritesNoEffectHalf() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.finishGame(Set.of(bolt.getId()));

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"moment\":\"activation\""), records.get(0));
    }

    /**
     * The game ends while a bracket is still open on an item that never
     * resolved (its own cost payment reaching 0 life): the outcome the
     * state-based check then publishes is not the item's, and the game-end
     * record holds it, the loss before the win. No record claims the item
     * resolved.
     */
    @Test
    void theOutcomeOfAGameEndingWithABracketOpenIsWrittenInTheGameEndRecord() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        BusBracketCollector collector = collector(game);
        SpellAbility bolt = damageAbility();
        game.getPlayers().get(1).loseConditionMet(GameLossReason.LifeReachedZero, null);

        collector.beginBracket(bolt);
        collector.onGameOutcome(new GameEventGameOutcome(
                1, List.of("a has won"), "a", "a: 1 b: 0 "));
        collector.finishGame(Set.of(bolt.getId()));

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"moment\":\"activation\""), records.get(0));
        String end = records.get(1);
        assertEquals(List.of("player_lost", "player_won"), RecordLines.eventTypes(end), end);
        assertEquals(null, RecordLines.payload(end).get("outcome"),
                "no record may claim bolt resolved: " + end);
    }

    /**
     * A cast that had already left the stack by the end writes nothing more
     * than its cost half: no {@code countered} outcome is invented for it.
     */
    @Test
    void aCastGoneFromTheStackByTheEndWritesNothingMore() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.finishGame(Set.of());

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertFalse(records.get(0).contains("countered"), records.get(0));
    }

    /**
     * The combat that ended the game is written.
     *
     * <p>A damage step's bracket closes at a phase boundary, and a game that
     * ends in combat damage never reaches one: {@code PhaseHandler.mainLoopStep}
     * returns the moment {@code checkStateBasedEffects} reports the game over,
     * so no {@code GameEventTurnPhase} and no {@code GameEventCombatEnded}
     * follow the lethal damage. The record for the most decisive combat of the
     * game was the one no game had.
     */
    @Test
    void theCombatThatEndedTheGameIsWritten() throws IOException {
        Game game = gameInACombatDamageStep();
        BusBracketCollector collector = collector(game);
        Card attacker = TestCards.build("Grizzly Bears");

        collector.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Runeclaw Bear").getView(), attacker.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));
        collector.finishGame(Set.of());

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"kind\":\"combat\""), records.get(0));
        assertTrue(records.get(0).contains("\"cause\":\"E" + attacker.getId() + "\""),
                "and in a damage step the attacker is the only cause there is");
    }

    /**
     * What resolves in a damage step after its damage belongs to its own
     * record, not the combat's.
     *
     * <p>A damage step has a priority window after the damage, and a
     * combat-damage trigger resolves in it. Routed by the step alone, the
     * trigger's events joined the combat record and left its own resolution
     * without them, and its damage to a creature read as combat damage.
     */
    @Test
    void whatResolvesAfterTheDamageIsNotTheCombats() throws IOException {
        Game game = gameInACombatDamageStep();
        BusBracketCollector collector = collector(game);
        Card attacker = TestCards.build("Grizzly Bears");
        collector.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Runeclaw Bear").getView(), attacker.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));

        SpellAbility bolt = damageAbility();
        collector.beginBracket(bolt);
        collector.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Llanowar Elves").getView(),
                bolt.getHostCard().getView(), 3,
                GameEventCardDamaged.DamageType.Normal));
        collector.endBracket(bolt.getId(), false);
        collector.finishGame(Set.of());

        List<String> records = written();
        List<String> combat = records.stream()
                .filter(r -> r.contains("\"kind\":\"combat\"")).toList();
        assertEquals(1, combat.size(), records.toString());
        assertTrue(combat.get(0).contains("\"amount\":2"), combat.get(0));
        assertFalse(combat.get(0).contains("\"amount\":3"),
                "the bolt's damage is not the combat's: " + combat.get(0));
        String bolted = records.stream()
                .filter(r -> r.contains("\"amount\":3")).findFirst().orElseThrow();
        assertTrue(bolted.contains("\"kind\":\"resolution\""), bolted);
        assertTrue(bolted.contains("\"combat\":false"), bolted);
    }

    /**
     * A game in a damage step, the only state a combat bracket opens in.
     *
     * <p>Its own game rather than the shared one every other test here uses:
     * the phase is what routes an event into the combat bracket, and a shared
     * game left in a damage step would route every other test's events there
     * too.
     */
    private static Game gameInACombatDamageStep() {
        return gameInACombatDamageStep(PhaseType.COMBAT_DAMAGE);
    }

    private static Game gameInACombatDamageStep(PhaseType step) {
        return TestBoard.twoPlayerGame(step);
    }

    // ── an offer turned down ────────────────────────────────────────────

    /**
     * A refusal inside a resolution that did something is still resolved.
     *
     * <p>The half of the rule that keeps it from over-claiming. A line whose
     * mandatory clause moved a card and whose optional clause was declined did
     * something, and the schema's {@code declined} — offered, turned down,
     * nothing happened — is not the word for it.
     */
    @Test
    void aRefusalInsideAResolutionThatDidSomethingIsStillResolved()
            throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.noteDeclined();
        collector.onCardChangeZone(new GameEventCardChangeZone(
                TestCards.build("Grizzly Bears"),
                zone(ZoneType.Battlefield), zone(ZoneType.Graveyard)));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertTrue(records.get(1).contains("\"outcome\":\"resolved\""), records.get(1));
    }

    /** A refusal belongs to the bracket it happened in and to no later one. */
    @Test
    void aRefusalDoesNotOutliveItsBracket() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility refused = damageAbility();
        SpellAbility answer = damageAbility();

        collector.beginBracket(refused);
        collector.noteDeclined();
        // A spell cast in response goes on top; the refused spell's bracket
        // stays below it, unresolved.
        collector.beginBracket(answer);
        collector.endBracket(answer.getId(), false);
        collector.onNextAction();

        // The refused spell's cost half, then the answer's two halves: the
        // answer's effect half is resolved.
        List<String> records = written();
        assertEquals(3, records.size(), records.toString());
        assertTrue(records.get(2).contains("\"outcome\":\"resolved\""), records.get(2));
    }

    /**
     * A refusal made between a cast and its resolution's start is the
     * cast's consequence, which precedes the effect half's board: a
     * resolution that then did nothing observable is resolved, not
     * declined.
     */
    @Test
    void aRefusalBeforeTheResolutionStartsIsNotTheResolutions() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.noteDeclined();
        collector.resolutionStarting(bolt, false);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertTrue(records.get(1).contains("\"outcome\":\"resolved\""), records.get(1));
    }

    /** A refusal outside any bracket is nobody's, and is dropped. */
    @Test
    void aRefusalWithNothingResolvingIsNotHeldAgainstTheNextSpell()
            throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.noteDeclined();
        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        assertTrue(written().get(1).contains("\"outcome\":\"resolved\""));
    }

    /**
     * The hook is read for its answer, by the name the patch declares.
     *
     * <p>{@code PatchHooks} finds the listener interface by string and calls it
     * through a proxy, so the method name and the argument positions are the
     * whole contract; a typo in either degrades this channel with no error. The
     * proxy here is the same shape the patched {@code PlayerControllerAi}
     * installs.
     */
    @Test
    void theConfirmHookIsReadForTheAnswerItCarries() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        HookShapes.Confirm listener = confirmListener(collector);

        collector.beginBracket(bolt);
        listener.onConfirm(bolt, true);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        assertTrue(written().get(1).contains("\"outcome\":\"resolved\""),
                "an offer taken is described by what it did, not by an outcome");
    }

    /** And a refusal delivered the same way reaches the record. */
    @Test
    void aRefusalDeliveredThroughTheHookReachesTheRecord() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        HookShapes.Confirm listener = confirmListener(collector);

        collector.beginBracket(bolt);
        listener.onConfirm(bolt, false);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        // Like a fizzle, the half says so, holds no event (FR-042b) and is
        // under its cost half's link.
        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertTrue(records.get(1).contains("\"outcome\":\"declined\""), records.get(1));
        assertEquals(List.of(), RecordLines.eventTypes(records.get(1)), records.get(1));
        assertEquals(RecordLines.linkOf(records.get(0)), RecordLines.linkOf(records.get(1)));
    }

    // ── another game's confirmation ─────────────────────────────────────

    /**
     * A refusal reported for an ability belonging to a
     * DIFFERENT {@code Game} -- exactly {@code InterventionFork.forceResolution}'s
     * shape, since {@code GameSimulator.resolveStack} runs the forced ability
     * through the same {@code PlayerControllerAi.confirmAction}/{@code
     * confirmBidAction} this hook is installed on, regardless of how this
     * game's own two lobby seats were registered -- must not mark the live
     * bracket's own open resolution {@code declined}. {@code GameCopier}
     * builds every copied {@code Card} against a new {@code Game} object, so a
     * fork's ability's own host card already carries a different {@code Game}
     * than this collector's.
     *
     * <p>Companion to {@code OutcomeHandlerTest
     * .anOutcomeFromAnotherGameDoesNotReachThisGamesBracket} and {@code
     * ClauseContractTest.aClauseFromAnotherGameDoesNotReachThisGamesBracket},
     * which cover the same check on {@code ClauseOutcomeRouter}'s two hooks; this
     * one is on {@code BusBracketCollector} because {@code confirmHandler} is
     * declared there instead.
     */
    @Test
    void aDeclinedConfirmationFromAnotherGameDoesNotReachTheLiveBracket() throws IOException {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);

        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        HookShapes.Confirm listener = confirmListener(collector);

        collector.beginBracket(bolt);
        // A fork's own confirmation, declined -- must not touch this bracket.
        listener.onConfirm(forkAbility, false);
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        List<String> written = written();
        assertTrue(written.get(1).contains("\"outcome\":\"resolved\""),
                "a fork's own declined confirmation must not mark the live "
                        + "bracket's ability declined: " + written);
    }

    /**
     * A refusal naming no ability while a fork runs on this thread is the
     * fork's, not the live game's: it leaves the live bracket resolved, as a
     * refusal of the fork's own ability does. A forced resolution hears its
     * own through its fork-bound recording.
     */
    @Test
    void aRefusalNamingNoAbilityInsideAForkIsTheForks() throws IOException {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        HookShapes.Confirm listener = confirmListener(collector);

        collector.beginBracket(bolt);
        Forks.asFork(() -> {
            listener.onConfirm(null, false);
            listener.onConfirm(forkAbility, false);
            return null;
        });
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        assertTrue(written().get(1).contains("\"outcome\":\"resolved\""), written().toString());
    }
}
