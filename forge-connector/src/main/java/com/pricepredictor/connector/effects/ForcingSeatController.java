package com.pricepredictor.connector.effects;

import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.cost.Cost;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.WrappedAbility;
import forge.util.collect.FCollectionView;

import java.util.List;
import java.util.Map;

/**
 * The Forge AI with one play forced (FR-010b): the controller a directed
 * fork puts on its actor's player, in the copy only, when that seat is the
 * plain AI. At the decisions the play is made in it answers as its
 * {@link ForcedPlayRun} says; everywhere else it is the AI it replaces, built
 * from the same lobby player with the same simulation mode.
 */
final class ForcingSeatController extends PlayerControllerAi {

    private final ForcedPlayRun forced;

    ForcingSeatController(Game game, Player player, PlayerControllerAi replaced, ForcedPlayRun forced) {
        super(game, player, player.getLobbyPlayer());
        if (replaced.getAi().usesFullSimulation()) {
            getAi().setUseSimulation(forge.ai.AIOption.USE_FULL_SIMULATION);
        } else if (replaced.getAi().usesHybridSimulation()) {
            getAi().setUseSimulation(forge.ai.AIOption.USE_HYBRID_SIMULATION);
        }
        this.forced = forced;
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> line = forced.choose(player);
        return line != null ? line : super.chooseSpellAbilityToPlay();
    }

    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        Boolean played = forced.playChosen(player, sa);
        return played != null ? played : super.playChosenSpellAbility(sa);
    }

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        super.declareAttackers(attacker, combat);
        forced.declared(attacker, combat);
    }

    @Override
    public boolean confirmTrigger(WrappedAbility wrapper) {
        Boolean answer = wrapper.isMandatory() ? null : forced.answer(player, wrapper.getWrappedAbility());
        return answer != null ? answer : super.confirmTrigger(wrapper);
    }

    @Override
    public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                 List<String> options, Card cardToShow, Map<String, Object> params) {
        Boolean answer = forced.answer(player, sa);
        return answer != null ? answer : super.confirmAction(sa, mode, message, options, cardToShow, params);
    }

    /**
     * A "you may pay": a forced no declines; a forced yes pays whenever the
     * cost can be paid, whatever the AI would choose
     * ({@link ForcedPlayRun#pay}).
     */
    @Override
    public boolean payCostToPreventEffect(Cost cost, SpellAbility sa, boolean alreadyPaid,
                                          FCollectionView<Player> allPayers) {
        Boolean paid = forced.pay(player, cost, sa);
        return paid != null ? paid : super.payCostToPreventEffect(cost, sa, alreadyPaid, allPayers);
    }
}
