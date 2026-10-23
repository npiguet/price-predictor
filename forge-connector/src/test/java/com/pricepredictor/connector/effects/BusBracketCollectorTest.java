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
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
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
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return TestCollectors.bracket(game, writer, "run.0-l1.0");
    }

    /**
     * The shard, closed so every record written so far is on disk. The writer
     * creates the file with its first record, so a collector that wrote
     * nothing leaves none.
     */
    private Path closedShard() {
        writer.close();
        return writer.path();
    }

    /** The records written so far, one parsed line each. */
    private List<String> written() throws IOException {
        return ShardRun.readShard(closedShard());
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

    /**
     * A cast writes nothing yet, because its outcome is not knowable yet.
     *
     * <p>The cost half carries {@code outcome}, and whether a spell resolved,
     * fizzled or was countered does not exist at the moment it is cast — so a
     * record written here could only say a literal.
     */
    @Test
    void aCastIsHeldRatherThanWrittenWithAGuessedOutcome() {
        BusBracketCollector collector = collector();

        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);

        assertFalse(Files.exists(closedShard()), "the cast wrote a record");
        assertEquals(1, collector.finishGame(Set.of(bolt.getId())),
                "the cast is still held, to be abandoned at the end of the game");
    }

    /**
     * A resolution stamps the held half and writes both, joined by one link.
     */
    @Test
    void aResolutionWritesBothHalvesUnderOneLink() throws IOException {
        SpellAbility bolt = resolveBolt(collector());

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        String activation = records.get(0);
        String resolution = records.get(1);
        assertTrue(activation.contains("\"moment\":\"activation\""), activation);
        assertTrue(activation.contains("\"outcome\":\"resolved\""), activation);
        assertTrue(resolution.contains("\"moment\":\"resolution\""), resolution);

        String link = "run.0-l1.0.link." + bolt.getId();
        assertTrue(activation.contains("\"link_id\":\"" + link + "\""), activation);
        assertTrue(resolution.contains("\"link_id\":\"" + link + "\""), resolution);
    }

    /**
     * A fizzle writes the cost half alone, and issues no link at all.
     *
     * <p>{@code MagicStack.resolveStack} does not resolve a spell whose targets
     * are all gone, so there is no effect half to link to — and a {@code link_id}
     * on a fizzled activation is precisely the 306 dangling halves. The costs
     * were still paid, so the record itself must still be written.
     */
    @Test
    void aFizzleWritesThePartnerlessCostHalfAndNoEffectHalf() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), true);
        collector.onNextAction();

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"fizzled\""), records.get(0));
        assertTrue(records.get(0).contains("\"link_id\":null"), records.get(0));
        assertTrue(records.get(0).contains("\"moment\":\"activation\""), records.get(0));
    }

    /**
     * A held cast that has left the stack without resolving was countered.
     *
     * <p>Forge publishes the same removal event for a spell that resolved and
     * for one a counterspell took away, so the question is asked of the stack
     * instead: what is no longer on it and did not resolve was removed.
     */
    @Test
    void aSpellGoneFromTheStackIsWrittenOffAsCountered() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility survives = damageAbility();
        SpellAbility countered = damageAbility();

        collector.beginBracket(survives);
        collector.beginBracket(countered);
        int written = collector.writeOffRemoved(Set.of(survives.getId()));

        assertEquals(1, written);
        assertEquals(1, collector.finishGame(Set.of(survives.getId())),
                "the survivor is still held");
        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"countered\""), records.get(0));
        assertTrue(records.get(0).contains("\"link_id\":null"), records.get(0));
    }

    /**
     * Losing some targets but not all is a partial fizzle.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * before the spell resolves, so the count at resolution against the count at
     * the cast is the engine's own answer. Losing every target without the engine
     * calling it a fizzle is {@code CantFizzle}, where the spell still does what
     * it does.
     */
    @Test
    void losingSomeTargetsButNotAllIsAPartialFizzle() {
        SpellAbility bolt = damageAbility();
        bolt.resetTargets();
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));

        assertEquals(ActivationOutcome.RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.getTargets().remove(bolt.getTargets().get(0));
        assertEquals(ActivationOutcome.PARTIALLY_FIZZLED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.resetTargets();
        assertEquals(ActivationOutcome.RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));
    }

    /** An untargeted spell never reads as partially fizzled. */
    @Test
    void anUntargetedSpellIsSimplyResolved() {
        SpellAbility gain = AbilityFactory.getAbility(
                "SP$ GainLife | Cost$ W | Defined$ You | LifeAmount$ 3",
                TestCards.build("Healing Salve"));

        assertEquals(ActivationOutcome.RESOLVED,
                BusBracketCollector.outcomeOf(0, gain));
    }

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

    /** A draw reaches the collector and writes nothing -- no bracket, no shard growth. */
    @Test
    void aDrawnGameOutcomeWritesNothing() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.onGameOutcome(new GameEventGameOutcome(4, List.of("Draw"), null, ""));
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        String effectHalf = written().get(1);
        assertFalse(effectHalf.contains("player_won"), effectHalf);
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * A cast still on the stack when the game ends is counted, not invented.
     *
     * <p>It neither resolved nor left the stack, and {@code outcome} has no
     * member for a game that stopped. Before, it simply vanished: nothing
     * subscribed {@code GameEventGameFinished}, so one or two casts a game went
     * unrecorded and uncounted.
     */
    @Test
    void aCastStillOnTheStackAtTheEndIsCountedRatherThanGivenAnOutcome() {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        long abandoned = collector.finishGame(Set.of(bolt.getId()));

        assertEquals(1, abandoned);
        assertFalse(Files.exists(closedShard()), "no record claims one of the five");
    }

    /**
     * The sibling
     * {@link #aCastStillOnTheStackAtTheEndIsCountedRatherThanGivenAnOutcome}
     * pins the counting half of this, but never files anything into
     * {@link BusBracketCollector#bracketEvents} while the bracket is open, so
     * it cannot catch {@code flushOrphanedEvents()} writing a record anyway.
     *
     * <p>Reproduces the real shape: {@code openBracket(bolt)} runs at cast
     * time, so the bracket stays open on {@code bolt} for as long as it sits on
     * the stack -- here, until the game ends without it ever resolving (its
     * own cost payment reaching 0 life, in the shape this stands in for). An
     * event fires while that bracket is still open, then the game ends with
     * {@code bolt} still on the stack. The only honest record of {@code bolt}
     * is "abandoned, never resolved" -- exactly what {@code finishGame}
     * reports via {@code abandoned} -- not a second, contradictory
     * {@code kind:resolution} record naming {@code bolt} as having resolved.
     */
    @Test
    void aStillOpenBracketAtTheEndIsNotGivenAResolutionItNeverHad() {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        Card victim = TestCards.build("Runeclaw Bear");

        collector.beginBracket(bolt);
        // Filed while bolt's own bracket is still open -- resolving == bolt --
        // the same as any event its cost payment or its own resolution might
        // produce before the game is discovered to be over.
        collector.onCardDamaged(new GameEventCardDamaged(
                victim.getView(), bolt.getHostCard().getView(), 3,
                GameEventCardDamaged.DamageType.Normal));
        long abandoned = collector.finishGame(Set.of(bolt.getId()));

        assertEquals(1, abandoned, "bolt never resolved and must still be counted so");
        assertFalse(Files.exists(closedShard()),
                "no record may claim bolt resolved -- it never did, and "
                        + "writeOffRemoved's abandoned count already says so correctly");
    }

    /**
     * A cast that had already left the stack is written off as it always was.
     *
     * <p>The two halves of what is held at the end are different facts: this
     * one was removed without resolving, which {@code countered} is the
     * schema's word for, and only the other has no word at all.
     */
    @Test
    void aCastGoneFromTheStackByTheEndIsStillWrittenOff() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        long abandoned = collector.finishGame(Set.of());

        assertEquals(0, abandoned);
        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"countered\""), records.get(0));
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
     * The real sequencing for an ordinary non-combat
     * kill: the killing ability's own bracket opens, resolves and closes --
     * writing and clearing its own effect half -- and only afterwards, with no
     * bracket open, does the state-based check discover the game is over
     * (CR 704.3; {@code MagicStack} fires {@code GameEventSpellResolved}, and
     * this collector's {@code endBracket} runs, before {@code
     * checkStateBasedEffects} ever runs on the next main-loop iteration).
     * {@code GameEventGameOutcome} therefore always arrives with {@link
     * BusBracketCollector#bracketEvents} freshly emptied by the kill's own
     * {@code endBracket}, and no further ability ever resolves to flush it --
     * a sequencing that holding a bracket open around the event's whole life,
     * which the real engine never produces here, would not reproduce. Bound to
     * its own two-player game rather than {@code TestCards.game()} (which
     * registers no players at all), so the winning name has a real seat to
     * resolve against.
     */
    @Test
    void aGameOutcomeAfterTheKillingBracketAlreadyClosedIsStillWritten()
            throws IOException {
        Game outcomeGame = TestBoard.twoPlayerGame("p1", "p2");
        Player p1 = outcomeGame.getPlayers().stream()
                .filter(p -> "p1".equals(p.getLobbyPlayer().getName()))
                .findFirst().orElseThrow();
        Card boltHost = TestBoard.card(outcomeGame, null, "Lightning Bolt");
        SpellAbility bolt = AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose"
                        + " | NumDmg$ 3", boltHost);
        BusBracketCollector outcomeCollector = collector(outcomeGame);

        // Bolt resolves and its own bracket closes -- writing and clearing its
        // own effect half -- before anyone asks whether the game is over.
        outcomeCollector.beginBracket(bolt);
        outcomeCollector.endBracket(bolt.getId(), false);
        // Only now, with no bracket open, does the game turn out to be over.
        outcomeCollector.onGameOutcome(new GameEventGameOutcome(
                4, List.of("p1 has won"), "p1", "p1: 1 p2: 0 "));
        outcomeCollector.finishGame(Set.of());

        List<String> records = written();
        assertTrue(
                records.stream().anyMatch(r -> r.contains("\"type\":\"player_won\"")),
                "the outcome must not be lost just because no bracket was open "
                        + "when it fired: " + records);
        String outcomeRecord = records.stream()
                .filter(r -> r.contains("\"type\":\"player_won\""))
                .findFirst().orElseThrow();
        assertTrue(outcomeRecord.contains(SnapshotBuilder.playerId(p1)), outcomeRecord);
    }

    /**
     * A creature hit in the first-strike damage step opens a combat record.
     *
     * <p>Forge names that step {@code COMBAT_FIRST_STRIKE_DAMAGE}, which does
     * not contain {@code COMBAT_DAMAGE}. A test on that substring alone sent a
     * first striker's damage to a creature, its life changes and its kills to
     * the ordinary bracket, where no record ever read them: only damage to a
     * player, which reads the event's own combat flag, opened the step's
     * record.
     */
    @Test
    void aCreatureHitInTheFirstStrikeStepOpensACombatRecord() throws IOException {
        Game game = gameInACombatDamageStep(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        BusBracketCollector collector = collector(game);
        Card attacker = TestCards.build("White Knight");

        collector.onCardDamaged(new GameEventCardDamaged(
                TestCards.build("Grizzly Bears").getView(), attacker.getView(), 2,
                GameEventCardDamaged.DamageType.Normal));
        collector.finishGame(Set.of());

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"kind\":\"combat\""), records.get(0));
        assertTrue(records.get(0).contains("\"combat_substep\":\"first_strike\""),
                records.get(0));
        assertTrue(records.get(0).contains("\"cause\":\"E" + attacker.getId() + "\""),
                records.get(0));
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
     * An optional effect the controller refused is {@code declined}.
     *
     * <p>There is no bus event for it — an offer nobody took moves no card,
     * deals no damage and changes no life total — which is exactly why it
     * needs saying: the record is otherwise identical to one for an effect
     * that was never offered at all.
     *
     * <p>Partnerless, like a fizzle: {@code declined} is one of the three
     * outcomes the schema says has no effect half, and the Python loader
     * rejects a {@code declined} activation that reaches resolution.
     */
    @Test
    void anOfferTheControllerRefusedIsDeclinedAndStandsAlone() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.noteDeclined();
        collector.endBracket(bolt.getId(), false);
        collector.onNextAction();

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"declined\""), records.get(0));
        assertTrue(records.get(0).contains("\"link_id\":null"), records.get(0));
    }

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
        assertTrue(records.get(0).contains("\"outcome\":\"resolved\""), records.get(0));
    }

    /** A refusal belongs to the bracket it happened in and to no later one. */
    @Test
    void aRefusalDoesNotOutliveItsBracket() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility refused = damageAbility();
        SpellAbility answer = damageAbility();

        collector.beginBracket(refused);
        collector.noteDeclined();
        // A spell cast in response goes on top, and the refused spell's
        // bracket forgets what it gathered while it waited.
        collector.beginBracket(answer);
        collector.endBracket(answer.getId(), false);
        collector.onNextAction();

        // The refused spell is still held: nothing ever settled it. The
        // records are the answer's own two halves, and its cost half is first.
        List<String> records = written();
        assertTrue(records.get(0).contains("\"outcome\":\"resolved\""), records.get(0));
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

        assertTrue(written().get(0).contains("\"outcome\":\"resolved\""));
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

        assertTrue(written().get(0).contains("\"outcome\":\"resolved\""),
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

        assertTrue(written().get(0).contains("\"outcome\":\"declined\""));
    }

    // ── another game's confirmation ─────────────────────────────────────

    /**
     * A refusal reported for an ability belonging to a
     * DIFFERENT {@code Game} -- exactly {@code ForkCollector.forceResolution}'s
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

        assertTrue(written().get(0).contains("\"outcome\":\"resolved\""),
                "a fork's own declined confirmation must not mark the live "
                        + "bracket's ability declined: " + written());
    }
}
