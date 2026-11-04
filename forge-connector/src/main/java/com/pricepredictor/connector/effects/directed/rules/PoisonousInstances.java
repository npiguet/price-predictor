package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.Collection;
import java.util.Optional;

/**
 * 702.70b, each instance of poisonous triggers separately, on scenario
 * {@code 702.70b-1}: an attacking Virulent Sliver has poisonous 1 twice, from
 * itself and from A's other Virulent Sliver, and B gets two poison counters.
 *
 * <p>The base board, read at a boundary: the declare-blockers step begins,
 * before blockers are declared, with a creature attacking a player that has
 * two or more instances of poisonous, one of them granted by a static ability
 * of another permanent that is neither attacking nor blocking (the second
 * Sliver). It binds that permanent.
 *
 * <p>Variant {@code one_instance} ({@code remove_permanent}): the permanent is
 * not on the battlefield, so the attacker has one instance fewer and one
 * poisonous trigger fewer fires.
 */
public final class PoisonousInstances {

    public static final String RULE = "702.70b";
    static final String SCENARIO = "702.70b-1";
    static final String ONE_INSTANCE = "one_instance";

    private PoisonousInstances() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PoisonousInstances::bind, ONE_INSTANCE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_BLOCKERS) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : combat.getAttackers()) {
            Collection<KeywordInterface> instances = attacker.getKeywords(Keyword.POISONOUS);
            if (!(combat.getDefenderByAttacker(attacker) instanceof Player) || instances.size() < 2) {
                continue;
            }
            for (KeywordInterface instance : instances) {
                Card granter = instance.getStatic() == null ? null : instance.getStatic().getHostCard();
                if (granter != null && granter != attacker && granter.isInPlay()
                        && !combat.isAttacking(granter) && !combat.isBlocking(granter)) {
                    return Optional.of(Binding.empty().with("permanent", granter));
                }
            }
        }
        return Optional.empty();
    }
}
