package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.Collection;
import java.util.Optional;

/**
 * 702.105b, each instance of dethrone triggers separately, on scenario
 * {@code 702.105b-1}: Enraged Revolutionary, with dethrone of its own and
 * another from Marchesa, the Black Rose, attacks the player with the most
 * life and gets two +1/+1 counters.
 *
 * <p>The base board, read at a boundary: the declare-attackers step begins,
 * before attackers are declared, the active player controls an untapped
 * creature that can attack this turn with two or more instances of dethrone,
 * one granted by another permanent's static ability, and an opponent has the
 * most life or is tied for it. It binds that other permanent.
 *
 * <p>Variant {@code single-instance} ({@code remove_permanent}): that
 * permanent is gone, so the creature has one instance fewer and one trigger
 * fewer fires.
 */
public final class DethroneInstances {

    public static final String RULE = "702.105b";
    static final String SCENARIO = "702.105b-1";
    static final String SINGLE_INSTANCE = "single-instance";

    private DethroneInstances() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DethroneInstances::bind, SINGLE_INSTANCE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_ATTACKERS)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        boolean mostLife = active.getOpponents().stream()
                .anyMatch(opponent -> opponent.getLife() >= Boards.mostLifeBesides(opponent));
        if (!mostLife) {
            return Optional.empty();
        }
        for (Card attacker : Boards.untappedCreatures(active)) {
            Collection<KeywordInterface> instances = attacker.getKeywords(Keyword.DETHRONE);
            if (attacker.isSick() || instances.size() < 2) {
                continue;
            }
            for (KeywordInterface instance : instances) {
                Card granter = instance.getStatic() == null ? null : instance.getStatic().getHostCard();
                if (granter != null && granter != attacker && granter.isInPlay()) {
                    return Optional.of(Binding.empty().with("permanent", granter));
                }
            }
        }
        return Optional.empty();
    }
}
