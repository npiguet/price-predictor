package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardTypeView;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.Event;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.staticability.StaticAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * On a {@code continuous} record's board, the board without its static, a
 * static line's {@code active} is its condition on that board: a condition
 * the suppressed static made true reads false there, and one it has no part
 * in reads as it does live. The live game is unchanged by the read.
 */
@ExtendWith(ForgeExtension.class)
class SuppressedConditionTest {

    @TempDir
    Path tempDir;

    /**
     * Mycosynth Lattice makes Ardent Recruit, Kird Ape and the Forest
     * artifacts, so Recruit's metalcraft is on live; without the Lattice's
     * type static only the Lattice is an artifact and metalcraft is off.
     * Kird Ape's Forest is a Forest either way.
     */
    @Test
    void aConditionTheSuppressedStaticsTypesMetReadsOffOnItsRecord() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card lattice = board.battlefield("Mycosynth Lattice", board.you);
        Card recruit = board.battlefield("Ardent Recruit", board.you);
        Card ape = board.battlefield("Kird Ape", board.you);
        board.battlefield("Forest", board.you);
        board.game.getAction().checkStaticAbilities();
        assertEquals(3, recruit.getNetPower(), "metalcraft is not on live");
        Map<Card, CardTypeView> types = cachedTypes(board.game);

        assertEquals(true, active(live(board, recruit), recruit), "Recruit's metalcraft live");
        assertEquals(true, active(live(board, ape), ape), "Kird Ape's Forest live");
        String record = continuousRecordOf(board, lattice, 0);
        assertEquals(false, active(entity(record, recruit), recruit),
                "Recruit's metalcraft without the Lattice's types");
        assertEquals(true, active(entity(record, ape), ape),
                "Kird Ape's Forest without the Lattice's types");

        types.forEach((card, type) -> assertSame(type, card.getType(),
                "the snapshot changed " + card + "'s type"));
        assertEquals(3, recruit.getNetPower(), "the snapshot changed Recruit's power");
        assertTrue(recruit.isArtifact(), "the snapshot changed Recruit's type");
    }

    /**
     * Control Magic gives you the opponent's Ornithopter, your only
     * artifact, so Aeronaut Tinkerer's "as long as you control an artifact"
     * is on live and off without Control Magic.
     */
    @Test
    void aConditionTheSuppressedStaticsControlMetReadsOffOnItsRecord() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card tinkerer = board.battlefield("Aeronaut Tinkerer", board.you);
        Card ornithopter = board.battlefield("Ornithopter", board.opponent);
        Card magic = enchant(board, "Control Magic", ornithopter);
        assertEquals(board.you, ornithopter.getController(), "Control Magic did not take it");

        assertEquals(true, active(live(board, tinkerer), tinkerer), "Tinkerer's artifact live");
        String record = continuousRecordOf(board, magic, 0);
        assertEquals(false, active(entity(record, tinkerer), tinkerer),
                "Tinkerer's artifact without Control Magic");

        assertEquals(board.you, ornithopter.getController(), "the snapshot changed its controller");
        assertEquals(board.you.getView(), ornithopter.getView().getController(), "nor its view's");
    }

    /**
     * Reading a line's condition on the board without the static publishes
     * no game event, and leaves every card's type and controller as it was.
     */
    @Test
    void readingAConditionWithoutTheStaticChangesNothingLive() {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card lattice = board.battlefield("Mycosynth Lattice", board.you);
        Card recruit = board.battlefield("Ardent Recruit", board.you);
        board.battlefield("Ornithopter", board.you);
        board.game.getAction().checkStaticAbilities();
        Map<Card, CardTypeView> types = cachedTypes(board.game);
        Map<Card, Player> controllers = controllers(board.game);
        List<Object> heard = new ArrayList<>();
        board.game.subscribeToEvents(new Object() {
            @Subscribe
            public void on(Event event) {
                heard.add(event);
            }
        });

        String lines = CardLines.render(board.game, recruit,
                StaticSuppression.of(addTypeStatic(lattice)));

        assertEquals(false, active(Map.of("lines", parse("{\"l\":" + lines + "}").get("l")), recruit),
                "Recruit's metalcraft without the Lattice's types");
        assertEquals(List.of(), heard, "the read published events");
        types.forEach((card, type) -> assertSame(type, card.getType(),
                "the read changed " + card + "'s type"));
        assertEquals(controllers, controllers(board.game), "the read changed a controller");
    }

    /**
     * Only a static whose condition reads a card's type line or controller
     * is read on a lent board: Ardent Recruit's metalcraft
     * ({@code Condition$}) and Kird Ape's Forest ({@code IsPresent$}) are,
     * Glorious Anthem's unconditional static and Crusade's are not.
     */
    @Test
    void onlyAStaticWhoseConditionReadsTheBoardIsReadOnALentBoard() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        assertTrue(LentBoard.conditionReadsTheBoard(firstStatic(board, "Ardent Recruit")), "Ardent Recruit");
        assertTrue(LentBoard.conditionReadsTheBoard(firstStatic(board, "Kird Ape")), "Kird Ape");
        assertFalse(LentBoard.conditionReadsTheBoard(firstStatic(board, "Glorious Anthem")), "Glorious Anthem");
        assertFalse(LentBoard.conditionReadsTheBoard(firstStatic(board, "Crusade")), "Crusade");
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static void assumeSuppression() {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
    }

    /** Cast an aura of yours at {@code target} and resolve it; the aura on the battlefield. */
    private static Card enchant(CharacterizationBoard board, String auraName, Card target) {
        Card inHand = board.hand(auraName, board.you);
        board.cast(inHand, board.you, target);
        board.game.getStack().resolveStack();
        board.game.getAction().checkStaticAbilities();
        Card aura = board.game.getCardState(inHand);
        assertTrue(aura.isInPlay() && aura.getAttachedTo() != null,
                auraName + " did not resolve onto " + target);
        return aura;
    }

    /** The first static of a real card of yours, in no zone. */
    private static StaticAbility firstStatic(CharacterizationBoard board, String name) {
        return TestBoard.card(board.game, board.you, name).getStaticAbilities().get(0);
    }

    private static StaticAbility addTypeStatic(Card lattice) {
        return lattice.getStaticAbilities().stream().filter(st -> st.hasParam("AddType"))
                .findFirst().orElseThrow(() -> new AssertionError("no AddType$ on " + lattice));
    }

    /** Every card's type line as Forge reads it now, the object itself. */
    private static Map<Card, CardTypeView> cachedTypes(Game game) {
        Map<Card, CardTypeView> types = new IdentityHashMap<>();
        for (Card card : game.getCardsInGame()) {
            types.put(card, card.getType());
        }
        return types;
    }

    private static Map<Card, Player> controllers(Game game) {
        Map<Card, Player> controllers = new IdentityHashMap<>();
        for (Card card : game.getCardsInGame()) {
            controllers.put(card, card.getController());
        }
        return controllers;
    }

    /** The card's entity on the live board. */
    private static Map<String, Object> live(CharacterizationBoard board, Card card) {
        return parse(EntityJson.render(board.game, card, StaticSuppression.NONE));
    }

    /** The {@code active} of the card's own first static line in an entity. */
    private static Object active(Map<String, Object> entity, Card card) {
        String script = ScriptLocator.scriptFileOf(card);
        for (Object line : (List<?>) entity.get("lines")) {
            Map<String, Object> map = asMap(line);
            List<?> keys = (List<?>) map.getOrDefault("keys", List.of());
            if (keys.stream().anyMatch(key -> namesStatic(asMap(key), script, 0))) {
                return map.get("active");
            }
        }
        throw new AssertionError("no static line of " + script + " in " + entity);
    }

    /** Whether a provenance key names the static of this index in {@code script}. */
    private static boolean namesStatic(Map<String, Object> key, String script, long index) {
        return "static".equals(key.get("trait_kind")) && script.equals(key.get("script_file"))
                && Long.valueOf(index).equals(key.get("index_within_kind"));
    }

    /** The continuous record a boundary writes for {@code host}'s static of this index. */
    private String continuousRecordOf(CharacterizationBoard board, Card host, long index)
            throws Throwable {
        List<String> records = ShardRun.records(tempDir, board.game,
                (bracket, collectors) -> collectors.continuous().collect());
        String script = ScriptLocator.scriptFileOf(host);
        List<String> found = records.stream()
                .filter(record -> "continuous".equals(parse(record).get("kind")))
                .filter(record -> actsThrough(record, script, index))
                .toList();
        assertEquals(1, found.size(), "continuous records of " + script + ": " + records);
        return found.get(0);
    }

    /** Whether the record's first acting key is the static of this index in {@code script}. */
    private static boolean actsThrough(String record, String script, long index) {
        List<Map<String, Object>> keys = RecordAbilities.abilityKeys(record);
        return !keys.isEmpty() && namesStatic(keys.get(0), script, index);
    }

    private static Map<String, Object> entity(String record, Card card) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object entity : (List<?>) state.get("entities")) {
            Map<String, Object> map = asMap(entity);
            if (Refs.of(card).equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + Refs.of(card) + " in " + record);
    }
}
