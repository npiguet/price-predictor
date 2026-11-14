package com.pricepredictor.connector.effects;

import forge.ai.LobbyPlayerAi;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;

/**
 * A seat whose controller refuses every "you may" it is asked, through the
 * confirm hook, in the live game and in every copy of it: {@code GameCopier}
 * keeps a seat's lobby player, which builds the copy's controller.
 */
final class DecliningSeat extends LobbyPlayerAi {

    DecliningSeat(String name) {
        super(name, null);
    }

    @Override
    public Player createIngamePlayer(Game game, int id) {
        Player player = new Player(getName(), game, id);
        player.setFirstController(new Declining(game, player, this));
        return player;
    }

    private static final class Declining extends PlayerControllerAi {
        Declining(Game game, Player player, LobbyPlayerAi lobby) {
            super(game, player, lobby);
        }

        @Override
        public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                     List<String> options, Card cardToShow, Map<String, Object> params) {
            try {
                java.lang.reflect.Method note = PlayerControllerAi.class.getDeclaredMethod(
                        "noteEffectRecordConfirm", SpellAbility.class, boolean.class);
                note.setAccessible(true);
                return (Boolean) note.invoke(null, sa, false);
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
    }
}
