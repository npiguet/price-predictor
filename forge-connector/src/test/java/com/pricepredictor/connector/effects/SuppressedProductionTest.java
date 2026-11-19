package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.google.common.collect.Table;
import com.google.common.eventbus.Subscribe;
import forge.card.CardTypeView;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.Event;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * On a {@code continuous} record's board, the board without its static, a
 * permanent's summoning sickness and a player's {@code untapped_production}
 * read that board: a stolen permanent is as sick as it was for the
 * controller it has there. The live cards are unchanged by the read.
 */
@ExtendWith(ForgeExtension.class)
class SuppressedProductionTest {

    private static final String NO_MANA = "{\"W\":0,\"U\":0,\"B\":0,\"R\":0,\"G\":0,\"C\":0}";
    private static final String ONE_GREEN = "{\"W\":0,\"U\":0,\"B\":0,\"R\":0,\"G\":1,\"C\":0}";
    private static final String EVERY_COLOR_AND_GREEN =
            "{\"W\":1,\"U\":1,\"B\":1,\"R\":1,\"G\":2,\"C\":0}";

    @TempDir
    Path tempDir;
    /** How many shards this test has written, each to a directory of its own. */
    private int shards;

    /**
     * The opponent's Llanowar Elves, on the battlefield since an earlier
     * turn, taken by Control Magic this turn: live it is summoning sick for
     * you (302.6, you have not controlled it since your turn began), and on
     * Control Magic's record, where the opponent still controls it, it is not
     * sick and taps for the opponent's green.
     */
    @Test
    void aStolenCreatureIsAsSickOnTheStaticsRecordAsItWasForItsOldController() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card elves = board.battlefield("Llanowar Elves", board.opponent);
        elves.setSickness(false); // the opponent's since an earlier turn
        Card magic = enchant(board, "Control Magic", elves);
        assertTrue(elves.isSick(), "the elves are not sick for you live");

        String record = continuousRecordOf(board, magic);
        assertEquals(false, entity(record, elves).get("sick"), "the elves' sickness without Control Magic");
        assertEquals(parse(ONE_GREEN), production(player(record, board.opponent)),
                "the opponent's production without Control Magic");
        assertEquals(parse(NO_MANA), production(player(record, board.you)),
                "your production without Control Magic");
        assertLiveUnchanged(board, magic, elves);
    }

    /**
     * Llanowar Elves that entered under the opponent this turn and were taken
     * by Control Magic are sick on its record until a turn of the
     * opponent's begins, and not after: had the opponent kept them, they
     * would have controlled them since that turn began.
     */
    @Test
    void aStolenCreatureIsSickOnTheStaticsRecordUntilItsOldControllersTurnBegins() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card elves = board.battlefield("Llanowar Elves", board.opponent);
        Card magic = enchant(board, "Control Magic", elves);

        String record = continuousRecordOf(board, magic);
        assertEquals(true, entity(record, elves).get("sick"), "sick before the opponent's turn");
        assertEquals(parse(NO_MANA), production(player(record, board.opponent)),
                "the opponent's production before their turn");
        board.opponent.incrementTurn();
        record = continuousRecordOf(board, magic);
        assertEquals(false, entity(record, elves).get("sick"), "sick after the opponent's turn began");
        assertEquals(parse(ONE_GREEN), production(player(record, board.opponent)),
                "the opponent's production after their turn began");
        assertLiveUnchanged(board, magic, elves);
    }

    /** The stolen card is the same object, yours and as sick as live, after the snapshots. */
    private static void assertLiveUnchanged(CharacterizationBoard board, Card magic, Card stolen) {
        assertSame(stolen, magic.getAttachedTo(), "the snapshot replaced the stolen card");
        assertEquals(board.you, stolen.getController(), "the snapshot changed its controller");
        assertEquals(board.you.getView(), stolen.getView().getController(), "nor its view's");
        assertTrue(stolen.isSick() && stolen.getView().isFirstTurnControlled(),
                "the snapshot changed its sickness");
    }

    /**
     * A mana ability's colours read from the permanents' colours on the
     * board without the static: on Painter's Servant's record, the Servant
     * naming blue, Bloom Tender sees your permanents green (itself, a
     * Grizzly Bears) and the Servant colourless, and makes green alone; live
     * every permanent is also blue and it makes green and blue. The live
     * cards keep their colours and the colour changes Forge keeps.
     */
    @Test
    void aPlayersProductionReadsThePermanentsColoursWithoutTheStatic() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card servant = board.battlefield("Painter's Servant", board.you);
        servant.setChosenColors(List.of("blue"));
        board.battlefield("Bloom Tender", board.you).setSickness(false);
        board.battlefield("Grizzly Bears", board.you);
        board.game.getAction().checkStaticAbilities();
        Map<Integer, List<Object>> colors = colorChanges(board.game);

        assertEquals(parse(greenAnd("U")), liveProduction(board), "the live board's production");
        String record = continuousRecordOf(board, servant);
        assertEquals(parse(ONE_GREEN), production(player(record, board.you)),
                "your production without Painter's Servant");
        assertEquals(colors, colorChanges(board.game), "the snapshot changed a card's colours");
    }

    /**
     * A mana ability's activation read from the types on the board without
     * the static: Mycosynth Lattice makes a Forest the third artifact you
     * control, so Mox Opal's metalcraft lets it tap live for any colour, and
     * on the Lattice's type static's record only the Lattice and the Mox are
     * artifacts and the Mox makes nothing. The live cards keep their type
     * lines, the same objects.
     */
    @Test
    void aPlayersProductionReadsThePermanentsTypesWithoutTheStatic() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card lattice = board.battlefield("Mycosynth Lattice", board.you);
        board.battlefield("Mox Opal", board.you);
        board.battlefield("Forest", board.you);
        board.game.getAction().checkStaticAbilities();
        Map<Card, CardTypeView> types = cachedTypes(board.game);

        assertEquals(parse(EVERY_COLOR_AND_GREEN), liveProduction(board), "the live board's production");
        String record = continuousRecordOf(board, lattice, 0);
        assertEquals(parse(ONE_GREEN), production(player(record, board.you)),
                "your production without the Lattice's types");
        types.forEach((card, type) -> assertSame(type, card.getType(),
                "the snapshot changed " + card + "'s type"));
    }

    /**
     * Reading a player's production on the board without a static publishes
     * no game event: Painter's Servant's colours and Control Magic's control
     * lent and put back.
     */
    @Test
    void readingProductionWithoutTheStaticPublishesNothing() throws Throwable {
        assumeSuppression();
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card owned = board.battlefield("Bloom Tender", board.opponent);
        owned.setSickness(false);
        Card magic = enchant(board, "Control Magic", owned);
        Card servant = board.battlefield("Painter's Servant", board.you);
        servant.setChosenColors(List.of("blue"));
        board.game.getAction().checkStaticAbilities();
        List<Object> heard = new ArrayList<>();
        board.game.subscribeToEvents(new Object() {
            @Subscribe
            public void on(Event event) {
                heard.add(event);
            }
        });

        for (Card host : List.of(magic, servant)) {
            PlayerJson.render(board.game, board.opponent,
                    StaticSuppression.of(host.getStaticAbilities().get(0)));
        }
        assertEquals(List.of(), heard, "the read published events");
        assertLiveUnchanged(board, magic, magic.getAttachedTo());
    }

    private static Map<String, Object> liveProduction(CharacterizationBoard board) {
        return production(parse(PlayerJson.render(board.game, board.you, StaticSuppression.NONE)));
    }

    /** One green and one of {@code color}, as an {@code untapped_production}. */
    private static String greenAnd(String color) {
        return NO_MANA.replace("\"G\":0", "\"G\":1")
                .replace("\"" + color + "\":0", "\"" + color + "\":1");
    }

    /** Every card's colours and the colour changes Forge keeps for it, each change the object itself. */
    private static Map<Integer, List<Object>> colorChanges(Game game) {
        // Keyed by the card's id: an IdentityHashMap compares its values by identity too.
        Map<Integer, List<Object>> changes = new LinkedHashMap<>();
        for (Card card : game.getCardsInGame()) {
            List<Object> kept = new ArrayList<>();
            kept.add(card.getColor());
            for (Table<Long, Long, Object> table : ForgeFields.colorLayers(card)) {
                table.cellSet().forEach(cell -> kept.add(
                        new SameCell(cell.getRowKey(), cell.getColumnKey(), cell.getValue())));
            }
            changes.put(card.getId(), kept);
        }
        return changes;
    }

    /** A layer-table cell, its value compared by identity. */
    private record SameCell(Object row, Object column, Object value) {
        @Override
        public boolean equals(Object other) {
            return other instanceof SameCell that && row.equals(that.row)
                    && column.equals(that.column) && value == that.value;
        }

        @Override
        public int hashCode() {
            return Objects.hash(row, column, System.identityHashCode(value));
        }
    }

    /** Every card's type line as Forge reads it now, the object itself. */
    private static Map<Card, CardTypeView> cachedTypes(Game game) {
        Map<Card, CardTypeView> types = new IdentityHashMap<>();
        for (Card card : game.getCardsInGame()) {
            types.put(card, card.getType());
        }
        return types;
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

    /** The one continuous record a boundary writes for {@code host}'s first static. */
    private String continuousRecordOf(CharacterizationBoard board, Card host) throws Throwable {
        return continuousRecordOf(board, host, 0);
    }

    /** The one continuous record a boundary writes for {@code host}'s static of this index. */
    private String continuousRecordOf(CharacterizationBoard board, Card host, long index)
            throws Throwable {
        Path dir = Files.createDirectories(tempDir.resolve("shard" + shards++));
        List<String> records = ShardRun.records(dir, board.game,
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
        List<Map<String, Object>> keys = RecordLines.abilityKeys(record);
        if (keys.isEmpty()) {
            return false;
        }
        Map<String, Object> key = keys.get(0);
        return "static".equals(key.get("trait_kind")) && script.equals(key.get("script_file"))
                && Long.valueOf(index).equals(key.get("index_within_kind"));
    }

    private static Map<String, Object> production(Map<String, Object> player) {
        return asMap(player.get("untapped_production"));
    }

    private static Map<String, Object> player(String record, Player player) {
        return member(record, "players", Refs.of(player));
    }

    private static Map<String, Object> entity(String record, Card card) {
        return member(record, "entities", Refs.of(card));
    }

    /** The member of the record's state list {@code list} whose id is {@code ref}. */
    private static Map<String, Object> member(String record, String list, String ref) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object each : (List<?>) state.get(list)) {
            Map<String, Object> map = asMap(each);
            if (ref.equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + ref + " in " + record);
    }
}
