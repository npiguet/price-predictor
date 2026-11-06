package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.Optional;

/**
 * 702.98a, a creature with unleash can't block as long as it has a +1/+1
 * counter on it, on scenario {@code 702.98a-2}: A's Rakdos Cackler with a
 * +1/+1 counter is not among the legal blockers of B's attacking Grizzly
 * Bears.
 *
 * <p>The base board, read at a boundary: the declare-blockers step begins,
 * before blockers are declared, and a defending player of an attacker
 * controls an untapped creature with unleash and a +1/+1 counter, so that
 * creature cannot block. It binds the attacker and that creature.
 *
 * <p>Variant {@code no_counter} ({@code counters}): the creature has no
 * +1/+1 counter, so it may block.
 */
public final class UnleashCantBlock {

    public static final String RULE = "702.98a";
    static final String SCENARIO = "702.98a-2";
    static final String NO_COUNTER = "no_counter";

    private UnleashCantBlock() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, UnleashCantBlock::bind, NO_COUNTER,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("blocker"), CounterEnumType.P1P1, 0)));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_BLOCKERS) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : combat.getAttackers()) {
            Player defender = combat.getDefenderPlayerByAttacker(attacker);
            if (defender == null) {
                continue;
            }
            for (Card creature : Boards.untappedCreatures(defender)) {
                if (creature.hasKeyword(Keyword.UNLEASH) && creature.getCounters(CounterEnumType.P1P1) > 0) {
                    return Optional.of(Binding.empty()
                            .with("attacker", attacker)
                            .with("blocker", creature));
                }
            }
        }
        return Optional.empty();
    }
}
