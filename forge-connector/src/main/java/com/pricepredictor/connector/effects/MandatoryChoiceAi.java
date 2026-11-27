package com.pricepredictor.connector.effects;

import forge.LobbyPlayer;
import forge.ai.AIOption;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.player.DelayedReveal;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import forge.game.spellability.SpellAbility;
import forge.util.collect.FCollectionView;

import java.util.Map;

/**
 * The Forge AI, except that a single choice a resolving effect may not
 * decline is made.
 *
 * <p>Forge's {@code AttachAi} answers the choice of what an Aura or an
 * Equipment attaches to with nothing when it finds the attachment useless
 * (a land already of the type an Aura would make it, a creature that cannot
 * attack or block), whether or not the choice is optional. The effect then
 * does nothing, and an Aura spell stays in the stack zone. Where the AI
 * answers a mandatory choice with nothing although options are offered, this
 * controller takes the option {@link #optionIndex} names, by default the
 * first; every other answer is the AI's own, so a choice the AI makes is
 * made as it would be live.
 *
 * <p>The controller a forced resolution runs its actor under
 * ({@link #deciding}), and the random seat's parent, which draws the option
 * instead.
 */
public class MandatoryChoiceAi extends PlayerControllerAi {

    /** A controller for {@code player} in {@code game}, seated by {@code lobbyPlayer}. */
    public MandatoryChoiceAi(Game game, Player player, LobbyPlayer lobbyPlayer) {
        super(game, player, lobbyPlayer);
    }

    /**
     * The controller {@code player} decides a forced resolution under: its
     * own when that is not the plain AI (a random seat's copy is already one
     * of these), else the plain AI it has, in the same simulation mode, with
     * its mandatory choices made.
     */
    static PlayerController deciding(Player player) {
        PlayerController current = player.getController();
        if (current.getClass() != PlayerControllerAi.class) {
            return current;
        }
        MandatoryChoiceAi controller = new MandatoryChoiceAi(player.getGame(), player, player.getLobbyPlayer());
        controller.getAi().setUseSimulation(simulationOf((PlayerControllerAi) current));
        return controller;
    }

    private static AIOption simulationOf(PlayerControllerAi controller) {
        if (controller.getAi().usesFullSimulation()) {
            return AIOption.USE_FULL_SIMULATION;
        }
        return controller.getAi().usesHybridSimulation() ? AIOption.USE_HYBRID_SIMULATION : null;
    }

    @Override
    public <T extends GameEntity> T chooseSingleEntityForEffect(
            FCollectionView<T> optionList, DelayedReveal delayedReveal, SpellAbility sa, String title,
            boolean isOptional, Player relatedPlayer, Map<String, Object> params) {
        T chosen = super.chooseSingleEntityForEffect(
                optionList, delayedReveal, sa, title, isOptional, relatedPlayer, params);
        if (chosen == null && mustChoose(optionList, isOptional)) {
            return optionList.get(optionIndex(optionList.size()));
        }
        return chosen;
    }

    private static boolean mustChoose(FCollectionView<?> optionList, boolean isOptional) {
        return !isOptional && !optionList.isEmpty();
    }

    /**
     * Which of {@code size} offered options a mandatory choice the AI
     * declined takes: the first.
     */
    protected int optionIndex(int size) {
        return 0;
    }
}
