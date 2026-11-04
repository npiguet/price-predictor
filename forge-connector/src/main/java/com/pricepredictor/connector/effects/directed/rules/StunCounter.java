package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 122.1d, a stun counter replaces untapping with removing the counter, on
 * scenario {@code 122.1d-1}: a tapped Grizzly Bears with one stun counter is
 * told to untap, and loses the counter instead.
 *
 * <p>The base board, read at a boundary: the untap step begins and the
 * active player controls a tapped permanent with a stun counter, which the
 * step's untap is about to meet. It binds that permanent.
 *
 * <p>Variant {@code no_stun} ({@code counters}): the permanent has no stun
 * counter, so it untaps.
 */
public final class StunCounter {

    public static final String RULE = "122.1d";
    static final String SCENARIO = "122.1d-1";
    static final String NO_STUN = "no_stun";

    private StunCounter() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, StunCounter::bind, NO_STUN,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("permanent"), CounterEnumType.STUN, 0)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UNTAP)) {
            return Optional.empty();
        }
        return Boards.active(live).getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isTapped)
                .filter(card -> card.getCounters(CounterEnumType.STUN) > 0)
                .findFirst()
                .map(card -> Binding.empty().with("permanent", card));
    }
}
