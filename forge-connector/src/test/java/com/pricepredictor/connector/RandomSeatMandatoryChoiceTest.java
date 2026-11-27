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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The random seat makes a mandatory choice a resolving effect asks of it.
 *
 * <p>Forge's {@code AttachAi} answers what an Equipment attaches to with
 * nothing when the creature is "useless" to it (detained, unable to attack or
 * block, tapped for good), whether or not the choice is optional, so a
 * random-seat Equip aimed at such a creature resolved doing nothing.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatMandatoryChoiceTest {

    @Test
    void aRandomSeatEquipOntoAUselessCreatureAttaches() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Plains");
        Card bonesplitter = TestBoard.inPlay(game, a, "Bonesplitter");
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        bears.detain(game.getPlayers().get(1));
        RandomSeatController seat = GamePlayer.randomSeatOf(game).orElseThrow();
        seat.attach(game, null, null);

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
        assertNotNull(chosen, "the Equip is the seat's one play");
        assertEquals("Bonesplitter", chosen.get(0).getHostCard().getName());
        assertTrue(seat.playChosenSpellAbility(chosen.get(0)), "the Equip is activated");
        game.getStack().resolveStack();

        assertSame(bears, game.getCardState(bonesplitter).getAttachedTo(),
                "the Equipment attaches to the detained creature");
    }
}
