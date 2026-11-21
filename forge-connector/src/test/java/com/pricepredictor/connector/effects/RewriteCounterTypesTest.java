package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameEntityCounterTable;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
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
import static com.pricepredictor.connector.effects.RecordEvents.payload;
import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * An {@code AddCounter} replacement over a counter map naming several types,
 * played through the replacement hook: one rewrite record per type whose
 * count it changed, each with its own {@code counter_type}, its own {@code
 * incoming} and {@code outgoing}, and a board whose {@code pending_event} is
 * that {@code incoming}; a single-type map, and a map no type of which
 * changed, write the one record they always did.
 */
@ExtendWith(ForgeExtension.class)
class RewriteCounterTypesTest {

    private static final String WHY = "for rewrite records over a counter map";
    private static final String MARVEL = "Captain Marvel, Earth's Protector";

    @TempDir
    Path tempDir;

    // ── the split ───────────────────────────────────────────────────────

    /**
     * Hardened Scales edits only the +1/+1 counters of Captain Marvel's
     * power-up (a +1/+1 and an indestructible counter): one record, the
     * +1/+1 counters' alone; the indestructible counter it left alone is no
     * record's.
     */
    @Test
    void aReplacementEditingOneTypeOfTwoWritesThatTypesRecord() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Hardened Scales");
        Card marvel = TestBoard.inPlay(game, you(game), MARVEL);
        List<String> records = played(game, () -> powerUp(game, marvel));

        List<String> rewrites = rewrites(records, "cardsfolder/h/hardened_scales.txt");
        assertEquals(1, rewrites.size(), rewrites.toString());
        assertTyped(rewrites.get(0), "+1/+1", 1, 2);
        assertEquals(2, marvel.getCounters(CounterType.getType("P1P1")), "the scales applied");
    }

    /**
     * Doubling Season doubles both: two records, one per type, each with the
     * call's result and its board's pending event its own incoming.
     */
    @Test
    void aReplacementEditingEveryTypeWritesOneRecordPerType() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Doubling Season");
        Card marvel = TestBoard.inPlay(game, you(game), MARVEL);
        List<String> records = played(game, () -> powerUp(game, marvel));

        List<String> rewrites = rewrites(records, "cardsfolder/d/doubling_season.txt");
        assertEquals(2, rewrites.size(), rewrites.toString());
        assertTyped(rewrites.get(0), "+1/+1", 1, 2);
        assertTyped(rewrites.get(1), "INDESTRUCTIBLE", 1, 2);
        for (String rewrite : rewrites) {
            assertEquals("updated", payload(rewrite).get("result"), rewrite);
        }
        assertEquals(payload(rewrites.get(0)).get("replaced_by"),
                payload(rewrites.get(1)).get("replaced_by"), "one call's ability");
    }

    /**
     * Vizier of Remedies takes the one -1/-1 counter away: its type's record
     * goes from one to none, a delta of 0 on the outgoing side.
     */
    @Test
    void aReplacementRemovingATypeWritesItsRecordWithNoneLeft() throws Throwable {
        Game game = game();
        List<String> records = countersOnBears(
                game, "Vizier of Remedies", CounterType.getType("P1P1"), CounterType.getType("M1M1"));

        List<String> rewrites = rewrites(records, "cardsfolder/v/vizier_of_remedies.txt");
        assertEquals(1, rewrites.size(), rewrites.toString());
        assertTyped(rewrites.get(0), "-1/-1", 1, 0);
    }

    // ── no split ────────────────────────────────────────────────────────

    /** A single-type map writes its one record, typed, as it always did. */
    @Test
    void aSingleTypeMapWritesOneTypedRecord() throws Throwable {
        Game game = game();
        List<String> records = countersOnBears(
                game, "Hardened Scales", CounterType.getType("P1P1"), CounterType.getType("P1P1"));

        List<String> rewrites = rewrites(records, "cardsfolder/h/hardened_scales.txt");
        assertEquals(1, rewrites.size(), rewrites.toString());
        assertTyped(rewrites.get(0), "+1/+1", 2, 3);
    }

    /**
     * Melira keeps one poison counter at one: a replacement that changed no
     * type of a two-type map writes one record of the summed map, untyped,
     * with no outgoing event.
     */
    @Test
    void aReplacementChangingNoTypeWritesOneSummedRecord() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Melira, the Living Cure");
        List<String> records = played(game, () -> putCounters(game, you(game), opponent(game),
                CounterType.getType("POISON"), CounterType.getType("EXPERIENCE")));

        List<String> rewrites = rewrites(records, "cardsfolder/m/melira_the_living_cure.txt");
        assertEquals(1, rewrites.size(), rewrites.toString());
        Map<String, Object> payload = payload(rewrites.get(0));
        Map<String, Object> incoming = asMap(payload.get("incoming"));
        assertNull(params(incoming).get("counter_type"), incoming.toString());
        assertEquals(2L, params(incoming).get("delta"), incoming.toString());
        assertNull(payload.get("outgoing"), payload.toString());
    }

    // ── the counters of a card entering ─────────────────────────────────

    /**
     * Rhox Pummeler enters with a shield counter and, through Renata, a
     * +1/+1 counter: Hardened Scales, run on the move itself (an {@code
     * AddCounter} replacement over the move's {@code CounterMap}), writes the
     * +1/+1 counters' record alone.
     */
    @Test
    void aReplacementOnACardEnteringSplitsItsCountersTheSameWay() throws Throwable {
        Game game = game();
        TestBoard.inPlay(game, you(game), "Hardened Scales");
        TestBoard.inPlay(game, you(game), "Renata, Called to the Hunt");
        Card rhox = TestBoard.inHand(game, you(game), "Rhox Pummeler");
        List<String> records = played(game,
                () -> game.getAction().moveToPlay(rhox, you(game), null, null));

        Card entered = game.getCardState(rhox);
        assertEquals(2, entered.getCounters(CounterType.getType("P1P1")), "the scales applied");
        assertEquals(1, entered.getCounters(CounterType.getType("SHIELD")), "its shield");
        List<String> rewrites = rewrites(records, "cardsfolder/h/hardened_scales.txt");
        assertEquals(1, rewrites.size(), rewrites.toString());
        assertTyped(rewrites.get(0), "+1/+1", 1, 2);
    }

    // ── helpers ─────────────────────────────────────────────────────────

    /** One record of one type: its incoming and outgoing deltas, and its board's pending event. */
    private static void assertTyped(String record, String type, long incoming, long outgoing) {
        Map<String, Object> payload = payload(record);
        Map<String, Object> in = asMap(payload.get("incoming"));
        Map<String, Object> out = asMap(payload.get("outgoing"));
        assertEquals(type, params(in).get("counter_type"), record);
        assertEquals(incoming, params(in).get("delta"), record);
        assertEquals(type, params(out).get("counter_type"), record);
        assertEquals(outgoing, params(out).get("delta"), record);
        assertEquals(in, asMap(asMap(parse(record).get("state")).get("pending_event")),
                "the board's pending event is the record's own incoming: " + record);
    }

    /** Every rewrite record acting through {@code script}, in written order. */
    private static List<String> rewrites(List<String> records, String script) {
        List<String> found = new ArrayList<>();
        for (String record : records) {
            if ("rewrite".equals(parse(record).get("kind")) && script.equals(scriptOf(record))) {
                found.add(record);
            }
        }
        assertFalse(found.isEmpty(), "no rewrite through " + script + ": " + records);
        return found;
    }

    /** Captain Marvel's power-up, resolved: a +1/+1 and an indestructible counter at once. */
    private static void powerUp(Game game, Card marvel) {
        SpellAbility powerUp = TestBoard.ability(marvel, ApiType.PutCounter);
        powerUp.setActivatingPlayer(you(game));
        TestBoard.castAndResolve(game, powerUp);
    }

    /** The replacement card and Grizzly Bears in play, then one counter of each type put on the bears. */
    private List<String> countersOnBears(
            Game game, String replacement, CounterType first, CounterType second) throws Throwable {
        TestBoard.inPlay(game, you(game), replacement);
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        return played(game, () -> putCounters(game, bears, you(game), first, second));
    }

    /** One counter of each type (a type twice is two) put at once, with no effect as cause. */
    private static void putCounters(
            Game game, GameEntity on, Player by, CounterType... types) {
        GameEntityCounterTable table = new GameEntityCounterTable();
        for (CounterType type : types) {
            on.addCounter(type, 1, by, table);
        }
        table.replaceCounterEffect(game, null);
    }

    /**
     * Play {@code body} with every hook installed, the triggers of the cards
     * put on the board by hand made active first.
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
}
