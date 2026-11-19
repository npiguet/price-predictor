package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/** The random seat never draws a play that would pay it down to 0 life (decision D12, finding 8.6). */
@ExtendWith(ForgeExtension.class)
class RandomSeatLifeCostTest {

    private static RandomSeatController seatWithNecropotence(int life) {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        a.setLife(life, null);
        TestBoard.inPlay(game, a, "Necropotence");
        RandomSeatController seat = GamePlayer.randomSeatOf(game).orElseThrow();
        seat.attach(game, null, null);
        return seat;
    }

    @Test
    void aSeatAtOneLifeIsNeverOfferedPayOneLife() {
        assertNull(seatWithNecropotence(1).chooseSpellAbilityToPlay());
    }

    @Test
    void aSeatAtTwoLifeIsOfferedPayOneLife() {
        List<SpellAbility> chosen = seatWithNecropotence(2).chooseSpellAbilityToPlay();

        assertNotNull(chosen);
        assertEquals("Necropotence", chosen.get(0).getHostCard().getName());
    }
}
