package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * The random seat does not equip a creature the Equipment is already on
 *: 701.3b makes it do nothing, and the seat never
 * passes while a non-mana ability is playable, so it would repeat it.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatEquipRepeatTest {

    private record Board(Game game, Card bonesplitter, Card bears, Card second, RandomSeatController seat) {
    }

    /** The probe's board: 4 Plains, Bonesplitter and Grizzly Bears, with a second creature if asked. */
    private static Board board(boolean secondCreature, long seed) {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(seed)), new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        for (int i = 0; i < 4; i++) {
            TestBoard.inPlay(game, a, "Plains");
        }
        Card bonesplitter = TestBoard.inPlay(game, a, "Bonesplitter");
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        Card second = secondCreature ? TestBoard.inPlay(game, a, "Grizzly Bears") : null;
        RandomSeatController seat = GamePlayer.randomSeatOf(game).orElseThrow();
        seat.attach(game, null, null);
        return new Board(game, bonesplitter, bears, second, seat);
    }

    private static void equipOnce(Board board) {
        List<SpellAbility> chosen = board.seat().chooseSpellAbilityToPlay();
        board.seat().playChosenSpellAbility(chosen.get(0));
        board.game().getStack().resolveStack();
    }

    @Test
    void aSeatWithOneCreatureDoesNotEquipItAgain() {
        for (long seed = 1; seed <= 5; seed++) {
            Board board = board(false, seed);
            equipOnce(board);
            assertSame(board.bears(), board.game().getCardState(board.bonesplitter()).getAttachedTo());

            assertNull(board.seat().chooseSpellAbilityToPlay(),
                    "seed " + seed + ": the only Equip left puts it on the creature it is on");
        }
    }

    @Test
    void anEquipToAnotherCreatureIsStillOffered() {
        Board board = board(true, 1L);
        equipOnce(board);
        Card equipped = board.game().getCardState(board.bonesplitter()).getAttachedTo();

        List<SpellAbility> chosen = board.seat().chooseSpellAbilityToPlay();

        assertEquals("Bonesplitter", chosen.get(0).getHostCard().getName());
        board.seat().playChosenSpellAbility(chosen.get(0));
        board.game().getStack().resolveStack();
        Card now = board.game().getCardState(board.bonesplitter()).getAttachedTo();
        assertSame(equipped == board.bears() ? board.second() : board.bears(), now);
    }
}
