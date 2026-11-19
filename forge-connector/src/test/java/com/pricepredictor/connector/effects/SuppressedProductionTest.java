package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    /** The one continuous record a boundary writes for {@code host}'s static. */
    private String continuousRecordOf(CharacterizationBoard board, Card host) throws Throwable {
        Path dir = java.nio.file.Files.createDirectories(tempDir.resolve("shard" + shards++));
        List<String> records = ShardRun.records(dir, board.game,
                (bracket, collectors) -> collectors.continuous().collect());
        String script = ScriptLocator.scriptFileOf(host);
        List<String> found = new ArrayList<>();
        for (String record : records) {
            if ("continuous".equals(parse(record).get("kind"))
                    && script.equals(RecordLines.scriptOf(record))) {
                found.add(record);
            }
        }
        assertEquals(1, found.size(), "continuous records of " + script + ": " + records);
        return found.get(0);
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
