package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.GamePieceType;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * The key a dungeon's room trigger reaches a record with (finding R2-3).
 *
 * <p>Forge builds a dungeon from its token script ({@code VentureEffect}
 * makes it with {@code CardFactory.getCard} from the token's paper card)
 * and its {@code K:Dungeon} keyword builds one {@code RoomEntered} trigger
 * per room, in the keyword's room order. The script file and the room's
 * {@code built} index are the ones the converter declares on the room's
 * line.
 */
@ExtendWith(ForgeExtension.class)
class DungeonRoomKeyTest {

    private static final String LOST_MINE = "tokenscripts/lost_mine_of_phandelver.txt";

    /** Lost Mine of Phandelver in your command zone, as a venture with none puts it there. */
    private static Card lostMine(CharacterizationBoard board) {
        Card dungeon = forge.game.card.CardFactory.getCard(
                TestCards.paperToken("lost_mine_of_phandelver"), board.you, board.game);
        dungeon.setGamePieceType(GamePieceType.DUNGEON);
        board.game.getAction().moveToCommand(dungeon, null);
        return dungeon;
    }

    /** Venture into the dungeon in your command zone; the room's trigger goes on the stack. */
    private static SpellAbility ventureAndStackTheRoom(CharacterizationBoard board, Card host) {
        Game game = board.game;
        board.resolve(AbilityFactory.getAbility("DB$ Venture", host), board.you);
        TestBoard.priorityWithTriggers(game);
        return game.getStack().peekAbility();
    }

    private static KeywordInterface dungeonKeyword(Card dungeon) {
        return dungeon.getCurrentState().getIntrinsicKeywords().stream()
                .filter(each -> each.getOriginal().startsWith("Dungeon:"))
                .findFirst().orElseThrow();
    }

    /**
     * Venturing into Lost Mine of Phandelver enters Cave Entrance: the
     * resolving trigger is the first the dungeon keyword built, the
     * dungeon's script is its token script, and the dungeon keyword is the
     * face's keyword 0.
     */
    @Test
    void aRoomTriggerIsItsDungeonKeywordsBuiltTriggerInTheTokenScript() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card dungeon = lostMine(board);
        Card host = board.hand("Kick in the Door", board.you);

        SpellAbility room = ventureAndStackTheRoom(board, host);

        Trigger trigger = room.getTrigger();
        KeywordInterface keyword = dungeonKeyword(dungeon);
        assertEquals("Cave Entrance", dungeon.getCurrentRoom());
        assertSame(keyword.getTriggers().iterator().next(), trigger, "the first room's trigger");
        assertSame(dungeon, trigger.getHostCard());
        assertEquals(ZoneType.Command, dungeon.getZone().getZoneType());
        assertEquals(LOST_MINE, ScriptLocator.scriptFileOf(dungeon));
        assertEquals(0, FaceOrder.keywordIndex(dungeon.getCurrentState(), keyword));
        assertEquals(new ProvenanceKey.Built(TraitKind.TRIGGER, 0), ProvenanceResolver.builtOf(trigger));
    }

    /** Each room's trigger is built at its room's place in the keyword's list. */
    @Test
    void eachRoomsTriggerIsBuiltInRoomOrder() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card dungeon = lostMine(board);

        List<String> rooms = dungeonKeyword(dungeon).getTriggers().stream()
                .map(trigger -> trigger.getOverridingAbility().getParam("RoomName")).toList();

        assertEquals(List.of("Cave Entrance", "Goblin Lair", "Mine Tunnels", "Storeroom",
                "Dark Pool", "Fungi Cavern", "Temple of Dumathoin"), rooms);
    }

    /**
     * The key the collector gives the resolving room trigger today: none,
     * with the reason {@code engine_effect}. {@code ProvenanceResolver}
     * counts a {@code DUNGEON} game piece among the engine-built cards with
     * no script file, so a room's record names no line of the dungeon's
     * sidecar, though the dungeon has a token script and the room trigger a
     * {@code built} index (above). The key the converter can declare is
     * {@code (tokenscripts/lost_mine_of_phandelver.txt, 0, keyword, 0,
     * built = trigger 0)}; were the collector to give it, this test would
     * expect that key.
     */
    @Test
    void aResolvingRoomIsReportedAsAnEngineEffectWithNoKey() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        lostMine(board);
        Card host = board.hand("Kick in the Door", board.you);

        SpellAbility room = ventureAndStackTheRoom(board, host);

        ProvenanceKey.Resolved resolved = ProvenanceResolver.resolve(room);
        assertNull(resolved.key());
        assertEquals(UnresolvedReason.ENGINE_EFFECT, resolved.reason());
    }
}
