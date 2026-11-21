package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.GameEntityCounterTable;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.replacement.ReplacementType;
import forge.game.trigger.TriggerType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.scriptOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link RunParamRows}' per-mode tables, checked against Forge's own
 * mode names, and each mode that moves a life total or a counter count read
 * off a trigger or replacement the engine really runs: its amount signed as
 * the bus signs it (positive a gain, negative a loss), and a face change in
 * an entity's {@code face_state} spelling (V2 #1, V2 #2, C2 #1, C2 #3, C2 #4,
 * X1 #8).
 */
@ExtendWith(ForgeExtension.class)
class RunParamNormalizerTest {

    private static final String WHY = "for trigger and rewrite records";

    @TempDir
    Path tempDir;

    // ── the table ───────────────────────────────────────────────────────

    /** Every trigger row is a mode Forge's {@code TriggerType} declares. */
    @Test
    void everyTriggerRowIsAForgeTriggerMode() {
        assertRowsAreForgeModes(triggerModes());
    }

    /** Every replacement row is a mode Forge's {@code ReplacementType} declares. */
    @Test
    void everyReplacementRowIsAForgeReplacementMode() {
        assertRowsAreForgeModes(replacementModes());
    }

    /**
     * One family of Forge modes beside its table: {@code kind} names it in the
     * messages, {@code forgeValueOf} is the enum's lookup and {@code forgeNames}
     * the names the enum declares.
     */
    private record ModeFamily(String kind, Set<String> rows, Set<String> unread,
                              Function<String, ?> forgeValueOf, List<String> forgeNames) {
    }

    private static ModeFamily triggerModes() {
        return new ModeFamily("trigger", RunParamRows.TRIGGER_MODES.keySet(),
                RunParamRows.TRIGGER_MODES_UNREAD, TriggerType::valueOf,
                Arrays.stream(TriggerType.values()).map(Enum::name).toList());
    }

    private static ModeFamily replacementModes() {
        return new ModeFamily("replacement", RunParamRows.REPLACEMENT_MODES.keySet(),
                RunParamRows.REPLACEMENT_MODES_UNREAD, ReplacementType::valueOf,
                Arrays.stream(ReplacementType.values()).map(Enum::name).toList());
    }

    /** Fails unless Forge's {@code valueOf} answers to every row name of the family. */
    private static void assertRowsAreForgeModes(ModeFamily family) {
        List<String> unknown = new ArrayList<>();
        for (String mode : family.rows()) {
            try {
                family.forgeValueOf().apply(mode);
            } catch (IllegalArgumentException e) {
                unknown.add(mode);
            }
        }
        assertEquals(List.of(), unknown, "rows no Forge " + family.kind() + " mode answers to");
    }

    // ── reading the played records ──────────────────────────────────────

    /** The events of every fired trigger record acting through {@code script}. */
    private static List<Map<String, Object>> fired(List<String> records, String script) {
        List<Map<String, Object>> events = new ArrayList<>();
        for (String record : records) {
            if (isFiredTriggerThrough(record, script)) {
                events.add(asMap(payload(record).get("event")));
            }
        }
        assertFalse(events.isEmpty(), "no fired trigger through " + script + ": " + records);
        return events;
    }

    private static boolean isFiredTriggerThrough(String record, String script) {
        if (!"trigger".equals(parse(record).get("kind"))) {
            return false;
        }
        if (!script.equals(scriptOf(record))) {
            return false;
        }
        return Boolean.TRUE.equals(payload(record).get("fired"));
    }

    /** The one rewrite record acting through {@code script}'s payload. */
    private static Map<String, Object> rewrite(List<String> records, String script) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            if ("rewrite".equals(parse(record).get("kind")) && script.equals(scriptOf(record))) {
                found.add(payload(record));
            }
        }
        assertEquals(1, found.size(), "rewrites through " + script + ": " + records);
        return found.get(0);
    }

    private static Object delta(Map<String, Object> event) {
        return params(event).get("delta");
    }

    /**
     * Play {@code body} with every hook installed, the triggers of the cards
     * put on the board by hand made active first, as the engine's next
     * state check would make them.
     */
    private List<String> played(Game game, ShardRun.Played body) throws Throwable {
        game.getTriggerHandler().resetActiveTriggers();
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, body);
    }

    /** Put {@code cardName} in play, play {@code body}, and the first event of the trigger of {@code script}. */
    private Map<String, Object> firstFired(
            Game game, String cardName, ShardRun.Played body, String script) throws Throwable {
        TestBoard.inPlay(game, you(game), cardName);
        List<String> records = played(game, body);
        return fired(records, script).get(0);
    }

    /** Put {@code cardName} in play, play {@code body}, and the rewrite record of {@code script}. */
    private Map<String, Object> rewriteOf(
            Game game, String cardName, ShardRun.Played body, String script) throws Throwable {
        TestBoard.inPlay(game, you(game), cardName);
        List<String> records = played(game, body);
        return rewrite(records, script);
    }

    private static void assertDeltas(Map<String, Object> rewrite, long incoming, long outgoing) {
        assertEquals(incoming, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
        assertEquals(outgoing, delta(asMap(rewrite.get("outgoing"))), rewrite.toString());
    }

    private static Game game() {
        return TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    }

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }

    private static Player opponent(Game game) {
        return game.getPlayers().get(1);
    }

    // ── life ────────────────────────────────────────────────────────────

    @Test
    void lifeGainedIsAPositiveDelta() throws Throwable {
        Game game = game();
        Map<String, Object> event = firstFired(game, "Ajani's Pridemate",
                () -> you(game).gainLife(2, null, null), "cardsfolder/a/ajanis_pridemate.txt");
        assertEquals("life_change", event.get("type"), event.toString());
        assertEquals(2L, delta(event), event.toString());
    }

    @Test
    void lifeLostIsANegativeDelta() throws Throwable {
        Game game = game();
        Map<String, Object> event = firstFired(game, "Exquisite Blood",
                () -> opponent(game).loseLife(3, false, false, null),
                "cardsfolder/e/exquisite_blood.txt");
        assertEquals("LifeLost", params(event).get("mode"), event.toString());
        assertEquals(-3L, delta(event), event.toString());
    }

    @Test
    void payingLifeIsANegativeDelta() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Font of Agonies");
        Card bolt = TestBoard.inHand(game, you(game), "Lightning Bolt");
        List<String> records = played(game,
                () -> you(game).payLife(2, TestBoard.spellOf(bolt, you(game)), false));

        Map<String, Object> event = fired(records, "cardsfolder/f/font_of_agonies.txt").get(0);
        assertEquals("PayLife", params(event).get("mode"), event.toString());
        assertEquals(-2L, delta(event), event.toString());
    }

    @Test
    void aLifeGainReplacementWritesPositiveDeltas() throws Throwable {
        Game game = game();
        Map<String, Object> rewrite = rewriteOf(game, "Boon Reflection",
                () -> you(game).gainLife(3, null, null), "cardsfolder/b/boon_reflection.txt");
        assertDeltas(rewrite, 3L, 6L);
    }

    /** The case V2 #1 observed reading as a gain: an edited LifeReduced amount. */
    @Test
    void aLifeLossReplacementWritesNegativeDeltas() throws Throwable {
        Game game = game();
        Map<String, Object> rewrite = rewriteOf(game, "Bloodletter of Aclazotz",
                () -> opponent(game).loseLife(2, false, false, null),
                "cardsfolder/b/bloodletter_of_aclazotz.txt");
        assertDeltas(rewrite, -2L, -4L);
    }

    @Test
    void aPayLifeReplacementWritesANegativeDelta() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Ashiok, Wicked Manipulator");
        for (int i = 0; i < 4; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, you(game), "Forest"), null, null);
        }
        Card bolt = TestBoard.inHand(game, you(game), "Lightning Bolt");
        List<String> records = played(game,
                () -> you(game).payLife(2, TestBoard.spellOf(bolt, you(game)), false));

        Map<String, Object> rewrite = rewrite(
                records, "cardsfolder/a/ashiok_wicked_manipulator.txt");
        assertEquals(-2L, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
    }

    // ── counters ────────────────────────────────────────────────────────

    private static void putCounters(Card card, CounterType type, int n, Player by) {
        GameEntityCounterTable table = new GameEntityCounterTable();
        card.addCounter(type, n, by, table);
        table.replaceCounterEffect(card.getGame(), null);
    }

    /** CounterAdded fires once per counter: each is +1, never the running total. */
    @Test
    void eachCounterAddedIsOneCounter() throws Throwable {
        Game game = game();
        Card hoplite = TestBoard.inPlay(game, you(game), "Bloodcrazed Hoplite");
        List<String> records = played(game,
                () -> putCounters(hoplite, CounterType.getType("P1P1"), 2, you(game)));

        List<Map<String, Object>> events = fired(records, "cardsfolder/b/bloodcrazed_hoplite.txt");
        // Two firings, each one counter; two identical records are written once.
        assertFalse(events.isEmpty());
        for (Map<String, Object> event : events) {
            assertEquals("counter_change", event.get("type"), event.toString());
            assertEquals("+1/+1", params(event).get("counter_type"), event.toString());
            assertEquals(1L, delta(event), event.toString());
        }
    }

    @Test
    void countersAddedAtOnceAreTheirNumber() throws Throwable {
        Game game = game();
        Card biomancer = TestBoard.inPlay(game, you(game), "Benthic Biomancer");
        List<String> records = played(game,
                () -> putCounters(biomancer, CounterType.getType("P1P1"), 3, you(game)));

        Map<String, Object> event = fired(records, "cardsfolder/b/benthic_biomancer.txt").get(0);
        assertEquals(3L, delta(event), event.toString());
    }

    /** CounterRemoved fires once per counter: each is -1, never the count left. */
    @Test
    void eachCounterRemovedIsMinusOne() throws Throwable {
        Game game = game();
        Card chronicler = TestBoard.card(game, you(game), "Aeon Chronicler");
        chronicler = game.getAction().exile(chronicler, null, null);
        CounterType time = CounterType.getType("TIME");
        chronicler.setCounters(time, 4);
        Card exiled = chronicler;
        List<String> records = played(game,
                () -> exiled.subtractCounter(time, 2, you(game)));

        List<Map<String, Object>> events = fired(records, "cardsfolder/a/aeon_chronicler.txt");
        // Two firings, each one counter; two identical records are written once.
        assertFalse(events.isEmpty());
        for (Map<String, Object> event : events) {
            assertEquals(-1L, delta(event), event.toString());
        }
    }

    @Test
    void countersRemovedAtOnceAreMinusTheirNumber() throws Throwable {
        Game game = game();
        Card chandra = TestBoard.inPlay(game, you(game), "Chandra, Fire Artisan");
        CounterType loyalty = CounterType.getType("LOYALTY");
        chandra.setCounters(loyalty, 4);
        List<String> records = played(game,
                () -> chandra.subtractCounter(loyalty, 2, opponent(game)));

        Map<String, Object> event = fired(records, "cardsfolder/c/chandra_fire_artisan.txt").get(0);
        assertEquals("CounterRemovedOnce", params(event).get("mode"), event.toString());
        assertEquals(-2L, delta(event), event.toString());
    }

    @Test
    void aCounterAddingReplacementWritesPositiveDeltas() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Hardened Scales");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        List<String> records = played(game,
                () -> putCounters(bears, CounterType.getType("P1P1"), 1, you(game)));

        assertDeltas(rewrite(records, "cardsfolder/h/hardened_scales.txt"), 1L, 2L);
    }

    /**
     * Deification keeps a Chandra at one loyalty: the removal it received
     * (all four) and the one it let through (three) are both negative, and
     * the edit is visible — before, the map's remaining count was written as
     * the amount and an edit of it was no rewrite at all.
     */
    @Test
    void aCounterRemovingReplacementWritesNegativeDeltas() throws Throwable {
        Game game = game();
        Card deification = TestBoard.inPlay(game, you(game), "Deification");
        deification.setChosenType("Chandra");
        TestBoard.inPlay(game, you(game), "Grizzly Bears");
        Card chandra = TestBoard.inPlay(game, you(game), "Chandra, Fire Artisan");
        CounterType loyalty = CounterType.getType("LOYALTY");
        chandra.setCounters(loyalty, 4);
        List<String> records = played(game,
                () -> chandra.subtractCounter(loyalty, 5, opponent(game), true));

        assertEquals(1, chandra.getCounters(loyalty), "Deification kept one");
        assertDeltas(rewrite(records, "cardsfolder/d/deification.txt"), -4L, -3L);
    }

    // ── faces, attachments, prevention ──────────────────────────────────

    /** Transformed is a face change to the face the card shows now. */
    @Test
    void transformingNamesTheFaceTurnedTo() throws Throwable {
        Game game = game();
        Card avacyn = TestBoard.inPlay(game, you(game), "Archangel Avacyn");
        List<String> records = played(game,
                () -> avacyn.changeCardState("Transform", null, null));

        assertEquals(CardStateName.Backside, avacyn.getCurrentStateName());
        Map<String, Object> event = fired(
                records, "cardsfolder/a/archangel_avacyn_avacyn_the_purifier.txt").get(0);
        assertEquals("face_change", event.get("type"), event.toString());
        assertEquals("back", params(event).get("to_state"), event.toString());
    }

    /** TurnFaceUp is a face change to the front, as an entity's face_state spells it. */
    @Test
    void turningFaceUpNamesTheFront() throws Throwable {
        Game game = game();
        Card ainok = TestBoard.card(game, you(game), "Ainok Survivalist");
        ainok.turnFaceDownNoUpdate();
        game.getAction().moveToPlay(ainok, you(game), null, null);
        List<String> records = played(game,
                () -> ainok.turnFaceUp(null));

        Map<String, Object> event = fired(records, "cardsfolder/a/ainok_survivalist.txt").get(0);
        assertEquals("face_change", event.get("type"), event.toString());
        assertEquals("front", params(event).get("to_state"), event.toString());
    }

    /** Unattached: the equipment is the subject, what it left the detached_from. */
    @Test
    void anUnattachmentNamesTheEquipmentAndWhatItLeft() throws Throwable {
        Game game = game();
        Card wargear = TestBoard.inPlay(game, you(game), "Grafted Wargear");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        wargear.attachToEntity(bears, null);
        List<String> records = played(game,
                () -> wargear.unattachFromEntity(bears));

        Map<String, Object> event = fired(records, "cardsfolder/g/grafted_wargear.txt").get(0);
        assertEquals("unattached", event.get("type"), event.toString());
        assertEquals(List.of(Refs.of(wargear)), event.get("subjects"), event.toString());
        assertEquals(Refs.of(bears), params(event).get("detached_from"), event.toString());
    }

    /** DamagePreventedOnce is a prevention of the amount prevented. */
    @Test
    void preventedDamageIsAPreventionOfItsAmount() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Selfless Squire");
        Card bolt = TestBoard.inHand(game, opponent(game), "Lightning Bolt");
        List<String> records = played(game, () -> {
            // What GameAction.dealDamage does with the damage a shield
            // prevented once the damage event is over.
            forge.game.card.CardDamageTable prevented = new forge.game.card.CardDamageTable();
            prevented.put(bolt, you(game), 3);
            prevented.triggerPreventDamage(false);
        });

        Map<String, Object> event = fired(records, "cardsfolder/s/selfless_squire.txt").get(0);
        assertEquals("damage_prevented", event.get("type"), event.toString());
        assertEquals(3L, params(event).get("amount"), event.toString());
        assertEquals(List.of(Refs.of(you(game))), event.get("subjects"), event.toString());
    }

    // ── every Forge mode is read or listed (follow-up item 7) ───────────

    /** Every TriggerType is a row of the table or listed as read for no magnitude, never both. */
    @Test
    void everyForgeTriggerModeIsReadOrListed() {
        assertReadXorListed(triggerModes());
    }

    /** Every ReplacementType is a row of the table or listed as read for no magnitude, never both. */
    @Test
    void everyForgeReplacementModeIsReadOrListed() {
        assertReadXorListed(replacementModes());
    }

    /** Fails for each Forge mode name in both or neither of the family's rows and unread list. */
    private static void assertReadXorListed(ModeFamily family) {
        List<String> missing = new ArrayList<>();
        for (String mode : family.forgeNames()) {
            boolean read = family.rows().contains(mode);
            boolean listed = family.unread().contains(mode);
            if (read == listed) {
                missing.add(mode);
            }
        }
        assertEquals(List.of(), missing,
                family.kind() + " modes neither read nor listed, or both");
    }

    // ── the modes that carry a magnitude ────────────────────────────────

    /** DamageDoneOnce: the damage one target took at once, summed, and its one source. */
    @Test
    void damageDealtAtOnceIsOneDamageOfItsSum() throws Throwable {
        Game game = game();
        Card reckoner = TestBoard.inPlay(game, opponent(game), "Boros Reckoner");
        Card shock = TestBoard.inHand(game, you(game), "Shock");
        List<String> records = played(game, () -> {
            TestBoard.cast(game, shock, you(game), reckoner);
            game.getStack().resolveStack();
        });

        Map<String, Object> event = fired(records, "cardsfolder/b/boros_reckoner.txt").get(0);
        assertEquals("damage_dealt", event.get("type"), event.toString());
        assertEquals(2L, params(event).get("amount"), event.toString());
        assertEquals(List.of(Refs.of(reckoner)), event.get("subjects"), event.toString());
        assertEquals(Refs.of(game.getCardState(shock)), params(event).get("source"), event.toString());
    }

    /** ChangesZoneAll: the cards a batch moved, and the move they made. */
    @Test
    void aBatchOfMovesIsOneZoneChangeOfItsCards() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Dramatic Finale");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        Card shock = TestBoard.inHand(game, opponent(game), "Shock");
        List<String> records = played(game, () -> {
            TestBoard.cast(game, shock, opponent(game), bears);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(true);
        });

        Map<String, Object> event = fired(records, "cardsfolder/d/dramatic_finale.txt").get(0);
        assertEquals("zone_change", event.get("type"), event.toString());
        assertEquals(List.of(Refs.of(bears)), event.get("subjects"), event.toString());
        assertEquals("battlefield", params(event).get("from_zone"), event.toString());
        assertEquals("graveyard", params(event).get("to_zone"), event.toString());
    }

    /** Exploited: the creature sacrificed to the exploit is what was sacrificed. */
    @Test
    void anExploitedCreatureIsSacrificed() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Colonel Autumn");
        Card butcher = TestBoard.inPlay(game, you(game), "Silumgar Butcher");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        List<String> records = played(game, () -> {
            // What SacrificeEffect runs for an exploit (its Exploited key the
            // sacrificed creature's last-known copy).
            Card lastKnown = forge.game.card.CardCopyService.getLKICopy(bears);
            game.getAction().moveToGraveyard(bears, null);
            Map<forge.game.ability.AbilityKey, Object> run = forge.game.ability.AbilityKey.mapFromCard(butcher);
            run.put(forge.game.ability.AbilityKey.Exploited, lastKnown);
            game.getTriggerHandler().runTrigger(TriggerType.Exploited, run, false);
        });

        Map<String, Object> event = fired(records, "cardsfolder/c/colonel_autumn.txt").get(0);
        assertEquals("sacrificed", event.get("type"), event.toString());
        assertEquals(List.of(Refs.of(bears)), event.get("subjects"), event.toString());
    }

    /** DiscardedAll: the cards discarded at once, as subjects and count. */
    @Test
    void cardsDiscardedAtOnceAreOneDiscardOfTheirCount() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Arius, Flyby Trawler");
        TestBoard.inHand(game, you(game), "Forest");
        TestBoard.inHand(game, you(game), "Island");
        Card rot = TestBoard.inHand(game, opponent(game), "Mind Rot");
        List<String> records = played(game, () -> {
            TestBoard.cast(game, rot, opponent(game), you(game));
            game.getStack().resolveStack();
        });

        Map<String, Object> event = fired(records, "cardsfolder/a/arius_flyby_trawler.txt").get(0);
        assertEquals("card_discarded", event.get("type"), event.toString());
        assertEquals(2L, params(event).get("count"), event.toString());
        assertEquals(2, ((List<?>) event.get("subjects")).size(), event.toString());
    }

    /** RolledDie and RolledDieOnce: the dice, their sides and their results. */
    @Test
    void aRolledDieIsADiceRoll() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "As Luck Would Have It");
        TestBoard.inPlay(game, you(game), "Brazen Dwarf");
        Card host = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        forge.game.spellability.SpellAbility roller = forge.game.ability.AbilityFactory.getAbility(
                "AB$ RollDice | Cost$ 0 | Sides$ 6", host);
        roller.setActivatingPlayer(you(game));
        List<String> records = played(game, () -> roller.resolve());

        Map<String, Object> one = fired(records, "cardsfolder/a/as_luck_would_have_it.txt").get(0);
        assertEquals("dice_rolled", one.get("type"), one.toString());
        assertEquals(6L, params(one).get("sides"), one.toString());
        Map<String, Object> die = asMap(((List<?>) params(one).get("results")).get(0));
        assertTrue(die.containsKey("natural") && die.containsKey("final"), one.toString());
        Map<String, Object> once = fired(records, "cardsfolder/b/brazen_dwarf.txt").get(0);
        assertEquals("dice_rolled", once.get("type"), once.toString());
        assertEquals(1, ((List<?>) params(once).get("results")).size(), once.toString());
    }

    /** CounterPlayerAddedAll: one player's counters on one object, their total. */
    @Test
    void countersAPlayerPutAtOnceAreTheirTotal() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "All Will Be One");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        List<String> records = played(game,
                () -> putCounters(bears, CounterType.getType("P1P1"), 3, you(game)));

        Map<String, Object> event = fired(records, "cardsfolder/a/all_will_be_one.txt").get(0);
        assertEquals("counter_change", event.get("type"), event.toString());
        assertEquals(List.of(Refs.of(bears)), event.get("subjects"), event.toString());
        assertEquals(3L, delta(event), event.toString());
    }

    /** The Scry replacement: the cards a scry would look at. */
    @Test
    void aScryReplacementReadsTheScrysCount() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Eligeth, Crossroads Augur");
        for (int i = 0; i < 4; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, you(game), "Forest"), null, null);
        }
        List<String> records = played(game,
                () -> game.getAction().scry(List.of(you(game)), 2, null));

        Map<String, Object> rewrite = rewrite(records, "cardsfolder/e/eligeth_crossroads_augur.txt");
        Map<String, Object> incoming = asMap(rewrite.get("incoming"));
        assertEquals("card_looked_at", incoming.get("type"), rewrite.toString());
        assertEquals(2L, params(incoming).get("count"), rewrite.toString());
    }

    /** The CreateToken replacement: the tokens it would make, so a doubling is an edit. */
    @Test
    void aTokenReplacementReadsTheTokensItMakes() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Academy Manufactor");
        Card host = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        forge.game.spellability.SpellAbility clue = forge.game.ability.AbilityFactory.getAbility(
                "AB$ Token | Cost$ 0 | TokenScript$ c_a_clue_draw", host);
        clue.setActivatingPlayer(you(game));
        List<String> records = played(game, () -> clue.resolve());

        Map<String, Object> rewrite = rewrite(records, "cardsfolder/a/academy_manufactor.txt");
        assertEquals(1L, params(asMap(rewrite.get("incoming"))).get("count"), rewrite.toString());
    }
}
