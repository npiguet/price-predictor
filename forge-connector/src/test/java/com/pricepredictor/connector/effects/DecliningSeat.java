package com.pricepredictor.connector.effects;

import forge.ai.LobbyPlayerAi;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.WrappedAbility;

import java.util.List;
import java.util.Map;

/**
 * A seat whose controller refuses every "you may" it is asked, through the
 * confirm hook, in the live game and in every copy of it: {@code GameCopier}
 * keeps a seat's lobby player, which builds the copy's controller. {@link
 * #accepting} is the same seat answering yes to every one.
 */
final class DecliningSeat extends LobbyPlayerAi {

    private final boolean answer;

    DecliningSeat(String name) {
        this(name, false);
    }

    private DecliningSeat(String name, boolean answer) {
        super(name, null);
        this.answer = answer;
    }

    /** A seat that accepts every "you may" it is asked. */
    static DecliningSeat accepting(String name) {
        return new DecliningSeat(name, true);
    }

    @Override
    public Player createIngamePlayer(Game game, int id) {
        Player player = new Player(getName(), game, id);
        player.setFirstController(new Declining(game, player, this, answer));
        return player;
    }

    private static final class Declining extends PlayerControllerAi {
        private final boolean answer;

        Declining(Game game, Player player, LobbyPlayerAi lobby, boolean answer) {
            super(game, player, lobby);
            this.answer = answer;
        }

        @Override
        public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                     List<String> options, Card cardToShow, Map<String, Object> params) {
            return ForcedPlayRun.noted(sa, answer);
        }

        /** An accepting seat takes every optional trigger; a declining one leaves them to the AI. */
        @Override
        public boolean confirmTrigger(WrappedAbility wrapper) {
            return answer && !wrapper.isMandatory()
                    ? ForcedPlayRun.noted(wrapper.getWrappedAbility(), true) : super.confirmTrigger(wrapper);
        }
    }
}
