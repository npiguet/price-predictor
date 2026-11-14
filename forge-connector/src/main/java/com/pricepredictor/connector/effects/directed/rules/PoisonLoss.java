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
 * scenario {@code 104.3d-1}: B has 8 poison counters when A casts Ichor Rats,
 * whose enters trigger gives each player one more, and B stays in the game
 * at 9; at 9 B would lose.
 *
 * <p>The base board, read at a boundary: a main phase begins and a player has
 * eight poison counters, the common side: the turn's next poison counter
 * leaves that player at nine. It binds that player.
 *
 * <p>Variant {@code nine} ({@code counters}): the player has nine poison
 * counters, so the turn's next poison counter given to it ends the game.
 * What gives it is the seats' own play this turn (an Ichor Rats cast, a
 * poisonous trigger); a fork in which none comes writes no poison record.
 */
public final class PoisonLoss {

    public static final String RULE = "104.3d";
    static final String SCENARIO = "104.3d-1";
    static final String NINE = "nine";
    /** The poison counters one short of the ten that lose the game. */
    static final int ONE_SHORT = 9;

    private PoisonLoss() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PoisonLoss::bind, NINE,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.player("player"), CounterEnumType.POISON, ONE_SHORT)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        return live.getPlayers().stream()
                .filter(player -> player.getPoisonCounters() == ONE_SHORT - 1)
                .findFirst()
                .map(player -> Binding.empty().with("player", player));
    }
}
