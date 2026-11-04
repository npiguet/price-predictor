package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.CounterEnumType;

import java.util.Optional;

/**
 * 104.3d, a player with ten or more poison counters loses the game, on
 * scenario {@code 104.3d-1}: B has 9 poison counters when A casts Ichor Rats,
 * whose enters trigger gives each player one more, and B loses.
 *
 * <p>The base board, read at a boundary: a main phase begins and a player has
 * nine poison counters, one short of losing, so the turn's next poison
 * counter given to that player ends the game. It binds that player.
 *
 * <p>Variant {@code eight} ({@code counters}): the player has eight poison
 * counters, so one more leaves it at nine and in the game.
 */
public final class PoisonLoss {

    public static final String RULE = "104.3d";
    static final String SCENARIO = "104.3d-1";
    static final String EIGHT = "eight";
    /** The poison counters one short of the ten that lose the game. */
    static final int ONE_SHORT = 9;

    private PoisonLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PoisonLoss::bind, EIGHT,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.player("player"), CounterEnumType.POISON, ONE_SHORT - 1)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        return live.getPlayers().stream()
                .filter(player -> player.getPoisonCounters() == ONE_SHORT)
                .findFirst()
                .map(player -> Binding.empty().with("player", player));
    }
}
