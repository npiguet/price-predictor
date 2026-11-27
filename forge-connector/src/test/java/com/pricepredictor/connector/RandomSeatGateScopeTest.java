package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.HookFailures;
import com.pricepredictor.connector.effects.LentActivator;
import com.pricepredictor.connector.effects.TestBoard;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The random seat's draw gates read a candidate with its activating player
 * lent: an Equip nobody has activated has none, and
 * the attach gate's target read needs one.
 */
@ExtendWith(ForgeExtension.class)
class RandomSeatGateScopeTest {

    @Test
    void anEquipNobodyActivatedIsReadWithItsActivatorLentAndLeftWithNone() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        for (int i = 0; i < 4; i++) {
            TestBoard.inPlay(game, a, "Plains");
        }
        Card bonesplitter = TestBoard.inPlay(game, a, "Bonesplitter");
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        bonesplitter.attachToEntity(bears, null, true);
        SpellAbility equip = bonesplitter.getSpellAbilities().stream()
                .filter(SpellAbility::isActivatedAbility).findFirst().orElseThrow();
        assertNull(equip.getActivatingPlayer(), "the board starts with no activator on the Equip");
        RandomSeatController seat = GamePlayer.randomSeatOf(game).orElseThrow();
        seat.attach(game, null, null);

        assertNull(seat.chooseSpellAbilityToPlay(), "the one Equip puts it on the creature it is on");

        assertNull(equip.getActivatingPlayer(), "the lend is put back");
    }

    @Test
    void aGateThatThrowsReadsAsNotDrawnIsCountedAndLeavesTheActivatorBack() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new RandomSeatLobbyPlayer("a", 1.0, new Random(1L)), new LobbyPlayerAi("b", null));
        Player a = game.getPlayers().get(0);
        Card bonesplitter = TestBoard.inPlay(game, a, "Bonesplitter");
        SpellAbility equip = bonesplitter.getSpellAbilities().stream()
                .filter(SpellAbility::isActivatedAbility).findFirst().orElseThrow();
        String key = "random seat gate unreadable";
        long before = HookFailures.counts().getOrDefault(key, 0L);

        boolean held = LentActivator.holds(equip, a, () -> {
            throw new IllegalStateException("a script the gate cannot read");
        });

        assertFalse(held);
        assertEquals(before + 1, HookFailures.counts().getOrDefault(key, 0L));
        assertNull(equip.getActivatingPlayer());
    }
}
