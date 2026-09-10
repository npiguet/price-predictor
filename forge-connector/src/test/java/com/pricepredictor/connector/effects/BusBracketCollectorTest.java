package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The four things a bus record could not say.
 *
 * <p>Each of these was measured on a 55,296-record corpus collected from the
 * previous version of this collector, and each was a wiring gap rather than a
 * format one: {@code outcome} was the literal {@code resolved} on all 4,989
 * activation records and 6.1% of link halves had no partner; 54% of
 * {@code zone_change} events said {@code to_zone=stack} and 13.68% of events in
 * a record repeated another one in it; {@code combat} and {@code resolution}
 * records were snapshotted one tier shallower than every other kind; and
 * {@code refs.modes} was filled on 125 records out of 55,296.
 *
 * <p>Three more were measured on the 463,360-record smoke corpus that followed:
 * not one of the 107,265 bus-derived events named what caused it, so two
 * attackers each dealing 1 damage to the same player rendered byte-identically
 * — 2,193 of the corpus's 3,780 duplicated events; nothing subscribed
 * {@code GameEventGameFinished}, so a cast still on the stack when the game
 * ended vanished with no record and no count, and the combat that decided the
 * game was never flushed at all; and {@code declined} was written on nothing,
 * so an optional effect the controller refused read exactly like one that did
 * something.
 *
 * <p>The bus itself is not exercised here — driving Forge's event bus needs a
 * running match, which is the integration test's job. What is exercised is
 * everything below the two lines that read the stack, against real abilities
 * built by {@code AbilityFactory} from real cards, which is where all of these
 * defects lived.
 */
@ExtendWith(ForgeExtension.class)
class BusBracketCollectorTest {

    @TempDir
    Path tempDir;

    private RecordShardWriter writer;

    private BusBracketCollector collector(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new BusBracketCollector(TestCards.game(), writer, "run.0-l1.0", caps);
    }

    private BusBracketCollector collector() {
        return collector(CollectionCaps.defaults());
    }

    /** The records written so far, one parsed line each. */
    private List<String> written() throws IOException {
        writer.close();
        Path path = tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX);
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new java.util.zip.GZIPInputStream(Files.newInputStream(path));
                var reader = new java.io.BufferedReader(new java.io.InputStreamReader(
                        gzip, java.nio.charset.StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }

    /** A targeted burn spell, the shape a partial fizzle happens to. */
    private static SpellAbility damageAbility() {
        Card host = TestCards.build("Lightning Bolt");
        return AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose"
                        + " | NumDmg$ 3", host);
    }

    // ── outcome ─────────────────────────────────────────────────────────

    /**
     * A cast writes nothing yet, because its outcome is not knowable yet.
     *
     * <p>This is the whole defect in one assertion. The cost half carries
     * {@code outcome}, and whether a spell resolved, fizzled or was countered
     * does not exist at the moment it is cast — so a record written here can
     * only say a literal, which is what all 4,989 activation records said.
     */
    @Test
    void aCastIsHeldRatherThanWrittenWithAGuessedOutcome() throws IOException {
        BusBracketCollector collector = collector();

        collector.beginBracket(damageAbility());

        assertEquals(0, collector.recordsWritten());
        assertEquals(1, collector.unresolvedActivations());
        assertEquals(List.of(), written());
    }

    /**
     * A resolution stamps the held half and writes both, joined by one link.
     */
    @Test
    void aResolutionWritesBothHalvesUnderOneLink() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), false);

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
        assertEquals(1, collector.unresolvedActivations(), "the survivor is still held");
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

        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.getTargets().remove(bolt.getTargets().get(0));
        assertEquals(EffectRecord.OUTCOME_PARTIALLY_FIZZLED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.resetTargets();
        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));
    }

    /** An untargeted spell never reads as partially fizzled. */
    @Test
    void anUntargetedSpellIsSimplyResolved() {
        SpellAbility gain = AbilityFactory.getAbility(
                "SP$ GainLife | Cost$ W | Defined$ You | LifeAmount$ 3",
                TestCards.build("Healing Salve"));

        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(0, gain));
    }

    // ── zone changes ────────────────────────────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    /**
     * A card put onto the stack is a cast, not an outcome of one.
     *
     * <p>It has a record already — the activation half, with what was paid. As a
     * zone_change it lands in whichever bracket is open, so an opponent's
     * response reads as something the resolving ability did; it was 54% of the
     * whole channel.
     */
    @Test
    void aMoveOntoTheStackIsNotRecordedAsAnOutcome() {
        assertNull(BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Lightning Bolt"),
                zone(ZoneType.Hand), zone(ZoneType.Stack))));
    }

    /** A move off the stack is an outcome, and now says where it came from. */
    @Test
    void aMoveOffTheStackKeepsBothOfItsEnds() {
        EffectEvent moved = BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Grizzly Bears"),
                zone(ZoneType.Stack), zone(ZoneType.Battlefield)));

        assertNotNull(moved);
        String json = moved.toJson();
        assertTrue(json.contains("\"from_zone\":\"stack\""), json);
        assertTrue(json.contains("\"to_zone\":\"battlefield\""), json);
    }

    /**
     * The same transition told twice in one record is filed once.
     *
     * <p>A card is on the battlefield or it is not, so a second identical
     * {@code zone_change} carries nothing the first did not.
     */
    @Test
    void aRepeatedStateTransitionIsFiledOnce() {
        List<EffectEvent> into = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        assertTrue(BusBracketCollector.fileEvent(zoneChange(), into, seen));
        assertFalse(BusBracketCollector.fileEvent(zoneChange(), into, seen));

        assertEquals(1, into.size());
    }

    /**
     * A repeated quantity is not, because there the repeat is the information.
     *
     * <p>Two creatures each dealing one damage to the same blocker render two
     * identical events, and collapsing them would turn two damage into one.
     */
    @Test
    void aRepeatedQuantityIsFiledEveryTime() {
        List<EffectEvent> into = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        assertTrue(BusBracketCollector.fileEvent(oneDamage(), into, seen));
        assertTrue(BusBracketCollector.fileEvent(oneDamage(), into, seen));

        assertEquals(2, into.size());
    }

    private static EffectEvent zoneChange() {
        return new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E7").param("from_zone", "hand")
                .param("to_zone", "battlefield");
    }

    private static EffectEvent oneDamage() {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E7").param("amount", 1).param("combat", true);
    }

    // ── snapshot depth ──────────────────────────────────────────────────

    /**
     * The collector snapshots at the run's depth, not at one of its own.
     *
     * <p>Checked where the vector lands rather than where it was read. Hardcoded
     * here, it made {@code state.tiers} say which collector wrote a record —
     * {@code combat} and {@code resolution} at {@code [1,2]} against
     * {@code [1,2,3]} everywhere else — which is collection metadata the schema
     * keeps out of a model's inputs.
     */
    @Test
    void theSnapshotDepthIsTheRunsAndNotTheCollectorsOwn() {
        assertArrayEquals(
                new int[]{1, 2, 3},
                collector(CollectionCaps.defaults()).snapshotTiers());

        CollectionCaps deep = new CollectionCaps(
                1, 0.1, 2, 2, List.of(), List.of(1, 2, 3, 4), 0.1);
        assertArrayEquals(new int[]{1, 2, 3, 4}, collector(deep).snapshotTiers());
    }

    /** And a caller that passes no caps still gets the run's, not a default here. */
    @Test
    void aCollectorBuiltWithoutCapsStillReadsTheRunsDepth() {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        String previous = System.getProperty("effect.snapshot.tiers");
        System.setProperty("effect.snapshot.tiers", "1,2,3,4");
        try {
            assertArrayEquals(
                    new int[]{1, 2, 3, 4},
                    new BusBracketCollector(TestCards.game(), writer, "g")
                            .snapshotTiers());
        } finally {
            if (previous == null) {
                System.clearProperty("effect.snapshot.tiers");
            } else {
                System.setProperty("effect.snapshot.tiers", previous);
            }
        }
    }

    // ── resolution-time refs ────────────────────────────────────────────

    /**
     * The effect half's refs are read when the bracket closes, not at the cast.
     *
     * <p>A charm's chosen mode, an announced X and a named card are set by the
     * engine while the ability resolves. Read at the cast they are all absent,
     * which is why {@code refs.modes} was filled on 125 records out of 55,296;
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
     * <p>The source is on the bus event and was being dropped. It is what makes
     * two attackers each dealing 1 damage to the same blocker two events rather
     * than one line printed twice — and, on its own, the answer to "which
     * permanent caused this damage", which the corpus did not carry at all.
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

        String effectHalf = written().get(1);
        assertTrue(effectHalf.contains("\"cause\":\"P" + source.getId() + "\""),
                effectHalf);
        assertFalse(
                effectHalf.contains("\"cause\":\"E" + bolt.getHostCard().getId() + "\""),
                "the bus named a source, and it outranks the bracket");
    }

    /**
     * With nothing resolving and nothing named, the record says nothing.
     *
     * <p>An honest absence rather than a placeholder: Forge genuinely names no
     * causer for a turn-based action, and a reader can act on the difference.
     */
    @Test
    void anEventNothingCausedCarriesNoCause() {
        assertNull(EventAttribution.cause(null, null));

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.UNTAPPED).subject("E7"), null, null)
                .toJson();

        assertFalse(json.contains("cause"), json);
    }

    /**
     * The game's own outcome reaches the shard resolved to the winner's ref,
     * not the display name Forge names it by -- see {@code
     * BusEvents#gameOutcome}. Bound to its own two-player game rather than
     * {@code TestCards.game()} (which registers no players at all), so the
     * winning name has a real seat to resolve against.
     */
    @Test
    void theGameOutcomeNamesTheWinningPlayerAsARef() throws IOException {
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("p1", null)),
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("p2", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Match match = new Match(rules, players, "outcome-shard-test");
        Game outcomeGame = new Game(players, rules, match);
        Player p1 = outcomeGame.getPlayers().stream()
                .filter(p -> "p1".equals(p.getLobbyPlayer().getName()))
                .findFirst().orElseThrow();
        Card boltHost = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard("Lightning Bolt"),
                null, outcomeGame);
        SpellAbility bolt = AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose"
                        + " | NumDmg$ 3", boltHost);
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        BusBracketCollector outcomeCollector = new BusBracketCollector(
                outcomeGame, writer, "run.0-l1.0", CollectionCaps.defaults());

        outcomeCollector.beginBracket(bolt);
        outcomeCollector.onGameOutcome(new GameEventGameOutcome(
                4, List.of("p1 has won"), "p1", "p1: 1 p2: 0 "));
        outcomeCollector.endBracket(bolt.getId(), false);

        String effectHalf = written().get(1);
        assertTrue(effectHalf.contains("\"type\":\"player_won\""), effectHalf);
        assertTrue(effectHalf.contains(SnapshotBuilder.playerId(p1)), effectHalf);
    }

    /** A draw reaches the collector and writes nothing -- no bracket, no shard growth. */
    @Test
    void aDrawnGameOutcomeWritesNothing() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.onGameOutcome(new GameEventGameOutcome(4, List.of("Draw"), null, ""));
        collector.endBracket(bolt.getId(), false);

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
    void aCastStillOnTheStackAtTheEndIsCountedRatherThanGivenAnOutcome()
            throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        long abandoned = collector.finishGame(Set.of(bolt.getId()));

        assertEquals(1, abandoned);
        assertEquals(1, collector.abandonedActivations());
        assertEquals(0, collector.unresolvedActivations(), "and it is let go of");
        assertEquals(List.of(), written(), "no record claims one of the five");
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
        assertEquals(0, collector.abandonedActivations());
        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"countered\""), records.get(0));
    }

    /**
     * The combat that ended the game is written, where it used to be dropped.
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
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
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
     * A game in a damage step, the only state a combat bracket opens in.
     *
     * <p>Its own game rather than the shared one every other test here uses:
     * the phase is what routes an event into the combat bracket, and a shared
     * game left in a damage step would route every other test's events there
     * too.
     */
    private static Game gameInACombatDamageStep() {
        Deck deck = new Deck();
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("a", null)),
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(
                players, rules, new Match(rules, players, "BusBracketTest"));
        game.getPhaseHandler().devModeSet(
                PhaseType.COMBAT_DAMAGE, game.getPlayers().get(0));
        return game;
    }

    // ── an offer turned down ────────────────────────────────────────────

    /**
     * An optional effect the controller refused is {@code declined}.
     *
     * <p>The outcome the schema asked for and nothing ever wrote. There is no
     * bus event for it — an offer nobody took moves no card, deals no damage
     * and changes no life total — which is exactly why it needed saying: the
     * record was otherwise identical to one for an effect that was never
     * offered at all.
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
        // A spell cast in response takes the bracket over.
        collector.beginBracket(answer);
        collector.endBracket(answer.getId(), false);

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
        ConfirmListener listener = (ConfirmListener) Proxy.newProxyInstance(
                ConfirmListener.class.getClassLoader(),
                new Class<?>[]{ConfirmListener.class},
                collector.confirmHandler());

        collector.beginBracket(bolt);
        listener.onConfirm(bolt, true);
        collector.endBracket(bolt.getId(), false);

        assertTrue(written().get(0).contains("\"outcome\":\"resolved\""),
                "an offer taken is described by what it did, not by an outcome");
    }

    /** And a refusal delivered the same way reaches the record. */
    @Test
    void aRefusalDeliveredThroughTheHookReachesTheRecord() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        ConfirmListener listener = (ConfirmListener) Proxy.newProxyInstance(
                ConfirmListener.class.getClassLoader(),
                new Class<?>[]{ConfirmListener.class},
                collector.confirmHandler());

        collector.beginBracket(bolt);
        listener.onConfirm(bolt, false);
        collector.endBracket(bolt.getId(), false);

        assertTrue(written().get(0).contains("\"outcome\":\"declined\""));
    }

    // ── final-fix-3.md item 2: another game's confirmation ──────────────

    /**
     * final-fix-3.md item 2: a refusal reported for an ability belonging to a
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
     * <p>Companion to {@code PatchedCollectorTest
     * .anOutcomeFromAnotherGameDoesNotReachThisGamesBracket} and {@code
     * ClauseContractTest.aClauseFromAnotherGameDoesNotReachThisGamesBracket},
     * which cover the same fix on {@code PatchedCollectors}'s two hooks; this
     * one is on {@code BusBracketCollector} because {@code confirmHandler} is
     * declared there instead.
     */
    @Test
    void aDeclinedConfirmationFromAnotherGameDoesNotReachTheLiveBracket() throws IOException {
        GameRules forkRules = new GameRules(GameType.Constructed);
        Game fork = new Game(List.of(), forkRules, new Match(forkRules, List.of(), "fork"));
        Card forkHost = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard("Grizzly Bears"),
                null, TestCards.nextCardId(), fork);
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);

        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        ConfirmListener listener = (ConfirmListener) Proxy.newProxyInstance(
                ConfirmListener.class.getClassLoader(),
                new Class<?>[]{ConfirmListener.class},
                collector.confirmHandler());

        collector.beginBracket(bolt);
        // A fork's own confirmation, declined -- must not touch this bracket.
        listener.onConfirm(forkAbility, false);
        collector.endBracket(bolt.getId(), false);

        assertTrue(written().get(0).contains("\"outcome\":\"resolved\""),
                "a fork's own declined confirmation must not mark the live "
                        + "bracket's ability declined: " + written());
    }

    /**
     * The interface the patch declares, restated here for the proxy.
     *
     * <p>Not imported: this module compiles against stock Forge, where it does
     * not exist. That is the same reason the collector installs the real one
     * reflectively.
     */
    private interface ConfirmListener {
        void onConfirm(SpellAbility sa, boolean confirmed);
    }
}
