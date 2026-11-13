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
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.scriptOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * {@link RunParamNormalizer}'s per-mode table, checked against Forge's own
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
        List<String> unknown = new ArrayList<>();
        for (String mode : RunParamNormalizer.TRIGGER_MODES.keySet()) {
            try {
                TriggerType.valueOf(mode);
            } catch (IllegalArgumentException e) {
                unknown.add(mode);
            }
        }
        assertEquals(List.of(), unknown, "rows no Forge trigger mode answers to");
    }

    /** Every replacement row is a mode Forge's {@code ReplacementType} declares. */
    @Test
    void everyReplacementRowIsAForgeReplacementMode() {
        List<String> unknown = new ArrayList<>();
        for (String mode : RunParamNormalizer.REPLACEMENT_MODES.keySet()) {
            try {
                ReplacementType.valueOf(mode);
            } catch (IllegalArgumentException e) {
                unknown.add(mode);
            }
        }
        assertEquals(List.of(), unknown, "rows no Forge replacement mode answers to");
    }

    // ── reading the played records ──────────────────────────────────────

    /** The events of every fired trigger record acting through {@code script}. */
    private static List<Map<String, Object>> fired(List<String> records, String script) {
        List<Map<String, Object>> events = new ArrayList<>();
        for (String record : records) {
            if ("trigger".equals(parse(record).get("kind")) && script.equals(scriptOf(record))
                    && Boolean.TRUE.equals(payload(record).get("fired"))) {
                events.add(asMap(payload(record).get("event")));
            }
        }
        assertFalse(events.isEmpty(), "no fired trigger through " + script + ": " + records);
        return events;
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
        TestBoard.inPlay(game, you(game), "Ajani's Pridemate");
        List<String> records = played(game,
                () -> you(game).gainLife(2, null, null));

        Map<String, Object> event = fired(records, "cardsfolder/a/ajanis_pridemate.txt").get(0);
        assertEquals("life_change", event.get("type"), event.toString());
        assertEquals(2L, delta(event), event.toString());
    }

    @Test
    void lifeLostIsANegativeDelta() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Exquisite Blood");
        List<String> records = played(game,
                () -> opponent(game).loseLife(3, false, false, null));

        Map<String, Object> event = fired(records, "cardsfolder/e/exquisite_blood.txt").get(0);
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
        TestBoard.inPlay(game, you(game), "Boon Reflection");
        List<String> records = played(game,
                () -> you(game).gainLife(3, null, null));

        Map<String, Object> rewrite = rewrite(records, "cardsfolder/b/boon_reflection.txt");
        assertEquals(3L, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
        assertEquals(6L, delta(asMap(rewrite.get("outgoing"))), rewrite.toString());
    }

    /** The case V2 #1 observed reading as a gain: an edited LifeReduced amount. */
    @Test
    void aLifeLossReplacementWritesNegativeDeltas() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Bloodletter of Aclazotz");
        List<String> records = played(game,
                () -> opponent(game).loseLife(2, false, false, null));

        Map<String, Object> rewrite = rewrite(records, "cardsfolder/b/bloodletter_of_aclazotz.txt");
        assertEquals(-2L, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
        assertEquals(-4L, delta(asMap(rewrite.get("outgoing"))), rewrite.toString());
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

    private static void putCounters(Game game, Card card, CounterType type, int n, Player by) {
        GameEntityCounterTable table = new GameEntityCounterTable();
        card.addCounter(type, n, by, table);
        table.replaceCounterEffect(game, null);
    }

    /** CounterAdded fires once per counter: each is +1, never the running total. */
    @Test
    void eachCounterAddedIsOneCounter() throws Throwable {
        Game game = game();
        Card hoplite = TestBoard.inPlay(game, you(game), "Bloodcrazed Hoplite");
        List<String> records = played(game,
                () -> putCounters(game, hoplite, CounterType.getType("P1P1"), 2, you(game)));

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
                () -> putCounters(game, biomancer, CounterType.getType("P1P1"), 3, you(game)));

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
                () -> putCounters(game, bears, CounterType.getType("P1P1"), 1, you(game)));

        Map<String, Object> rewrite = rewrite(records, "cardsfolder/h/hardened_scales.txt");
        assertEquals(1L, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
        assertEquals(2L, delta(asMap(rewrite.get("outgoing"))), rewrite.toString());
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
        Map<String, Object> rewrite = rewrite(records, "cardsfolder/d/deification.txt");
        assertEquals(-4L, delta(asMap(rewrite.get("incoming"))), rewrite.toString());
        assertEquals(-3L, delta(asMap(rewrite.get("outgoing"))), rewrite.toString());
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
}
