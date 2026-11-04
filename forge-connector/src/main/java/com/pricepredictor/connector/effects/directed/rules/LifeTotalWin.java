package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 104.2b, an effect that states a player wins the game ends it, on scenario
 * {@code 104.2b-1}: A has 40 life as its upkeep begins, so Felidar
 * Sovereign's trigger resolves and A wins.
 *
 * <p>The base board, read at a boundary: the upkeep begins and the active
 * player controls a permanent whose beginning-of-upkeep trigger wins it the
 * game if its life total is at least N (Felidar Sovereign, Test of
 * Endurance), and its life is at least N, so the trigger is about to fire. It
 * binds the permanent and the player.
 *
 * <p>Variant {@code one_short} ({@code life}): the player has N - 1 life, as
 * A has 39 instead of 40, so the trigger does not fire.
 */
public final class LifeTotalWin {

    public static final String RULE = "104.2b";
    static final String SCENARIO = "104.2b-1";
    static final String ONE_SHORT = "one_short";

    private LifeTotalWin() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, LifeTotalWin::bind, ONE_SHORT,
                new Application(ForkOperation.LIFE, LifeTotalWin::oneShort));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            Optional<Integer> needed = winningLife(permanent);
            if (needed.isPresent() && active.getLife() >= needed.get()) {
                return Optional.of(Binding.empty().with("permanent", permanent).with("player", active));
            }
        }
        return Optional.empty();
    }

    /** The life total at which the permanent's upkeep trigger wins its controller the game, if it has one. */
    static Optional<Integer> winningLife(Card permanent) {
        for (Trigger trigger : permanent.getTriggers()) {
            if (trigger.getMode() != TriggerType.Phase
                    || !"Upkeep".equals(trigger.getParam("Phase"))
                    || !"You".equals(trigger.getParam("ValidPlayer"))
                    || !"You".equals(trigger.getParam("LifeTotal"))
                    || !ApiType.WinsGame.name().equals(Boards.api(Boards.executed(trigger)))) {
                continue;
            }
            String amount = trigger.getParam("LifeAmount");
            if (amount != null && amount.startsWith("GE")) {
                Optional<Integer> needed = Boards.number(amount.substring(2));
                if (needed.isPresent()) {
                    return needed;
                }
            }
        }
        return Optional.empty();
    }

    private static BoardEdit oneShort(Binding fork) {
        int needed = winningLife(fork.card("permanent")).orElseThrow(
                () -> new IllegalStateException("the fork's permanent wins no game"));
        return new LifeOperation(fork.player("player"), needed - 1);
    }
}
