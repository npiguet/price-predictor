package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.CounterEnumType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 122.1h, a permanent with a finality counter that would go to a graveyard
 * is exiled instead, on scenario {@code 122.1h-1}: A's Shock kills B's
 * Grizzly Bears, which has a finality counter, and it is exiled.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins and a creature on the battlefield has a finality counter, so if the
 * turn kills it, it is exiled. It binds that creature.
 *
 * <p>Variant {@code no_finality} ({@code counters}): the creature has no
 * finality counter, so a death takes it to its owner's graveyard.
 */
public final class FinalityCounter {

    public static final String RULE = "122.1h";
    static final String SCENARIO = "122.1h-1";
    static final String NO_FINALITY = "no_finality";

    private FinalityCounter() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, FinalityCounter::bind, NO_FINALITY,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.FINALITY, 0)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isCreature() && card.getCounters(CounterEnumType.FINALITY) > 0)
                .findFirst()
                .map(card -> Binding.empty().with("creature", card));
    }
}
