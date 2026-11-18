package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * A player's {@code this_turn.creatures_died} counts the creatures that died
 * under that player's control this turn: put into a graveyard from the
 * battlefield (700.4, what morbid asks), and no other way of leaving it
 * (finding 9.11).
 */
@ExtendWith(ForgeExtension.class)
class CreaturesDiedTest {

    /** Unsummon returns a creature to its owner's hand: it left, it did not die. */
    @Test
    void aBouncedCreatureDidNotDie() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);

        board.cast(board.card("Unsummon", board.you), board.you, bears);
        board.game.getStack().resolveStack();

        assertEquals(ZoneType.Hand, bears.getZone().getZoneType());
        assertEquals(0L, creaturesDied(board, board.you));
    }

    /** An exiled creature left the battlefield without dying either. */
    @Test
    void anExiledCreatureDidNotDie() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().exile(board.battlefield("Grizzly Bears", board.you), null, null);

        assertEquals(0L, creaturesDied(board, board.you));
    }

    /** Lightning Bolt kills the opponent's Grizzly Bears: one death, theirs. */
    @Test
    void aCreatureKilledByDamageDiedUnderItsController() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);

        board.castAndResolve(board.cast(board.card("Lightning Bolt", board.you), board.you, bears));

        assertEquals(1L, creaturesDied(board, board.opponent));
        assertEquals(0L, creaturesDied(board, board.you));
    }

    /**
     * A token dies though it ceases to exist in the graveyard at once
     * (704.5d): the Soldier Lightning Bolt kills counts.
     */
    @Test
    void aTokenThatDiedCounts() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card soldier = board.token("w_1_1_soldier", board.you);

        board.castAndResolve(board.cast(board.card("Lightning Bolt", board.you), board.you, soldier));

        assertFalse(board.game.getCardsIn(ZoneType.Graveyard).contains(soldier),
                "the token has ceased to exist");
        assertEquals(1L, creaturesDied(board, board.you));
    }

    /**
     * A stolen creature dies under its controller, though its owner's
     * graveyard receives it: the death is the controller's.
     */
    @Test
    void aStolenCreatureDiedUnderItsController() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        bears.addTempController(board.you, board.game.getNextTimestamp());

        board.castAndResolve(board.cast(board.card("Lightning Bolt", board.you), board.you, bears));

        assertEquals(1L, creaturesDied(board, board.you));
        assertEquals(0L, creaturesDied(board, board.opponent));
    }

    /** A noncreature permanent put into a graveyard is no creature that died. */
    @Test
    void aNoncreaturePermanentPutIntoAGraveyardIsNoDeath() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().moveTo(ZoneType.Graveyard,
                board.battlefield("Glorious Anthem", board.you), null, null);

        assertEquals(0L, creaturesDied(board, board.you));
    }

    @SuppressWarnings("unchecked")
    private static Object creaturesDied(CharacterizationBoard board, Player player) {
        List<Map<String, Object>> players = (List<Map<String, Object>>) RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("players");
        Map<String, Object> entry = players.stream()
                .filter(p -> Refs.of(player).equals(p.get("id"))).findFirst().orElseThrow();
        return RecordLines.asMap(entry.get("this_turn")).get("creatures_died");
    }
}
