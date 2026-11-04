package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;

import java.util.List;
import java.util.Optional;

/**
 * 703.4m, combat damage is dealt simultaneously, on scenario
 * {@code 703.4m-1}: A's Grizzly Bears attacks and B's Grizzly Bears blocks
 * it, and each deals its 2 damage to the other at the same time.
 *
 * <p>The base board, read at a boundary: the first-strike damage step begins
 * (Forge begins it whenever blockers were declared, and skips its damage when
 * no combatant strikes first) with an attacker blocked by one creature that
 * blocks it alone, both dealing damage and neither with first strike or
 * double strike, so they will deal it to each other in the one regular damage
 * step. It binds the attacker and the blocker.
 *
 * <p>Variant {@code first_strike_blocker} ({@code replace_card}): the blocker
 * is Youthful Knight, a 2/1 with first strike, which deals its damage in the
 * first-strike step instead, carrying the old blocker's place in combat.
 */
public final class SimultaneousDamage {

    public static final String RULE = "703.4m";
    static final String SCENARIO = "703.4m-1";
    static final String FIRST_STRIKE_BLOCKER = "first_strike_blocker";
    static final String KNIGHT = "Youthful Knight";

    private SimultaneousDamage() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousDamage::bind, FIRST_STRIKE_BLOCKER,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("blocker"), KNIGHT)));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_FIRST_STRIKE_DAMAGE) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : combat.getAttackers()) {
            List<Card> blockers = combat.getBlockers(attacker);
            if (blockers.size() != 1) {
                continue;
            }
            Card blocker = blockers.get(0);
            if (combat.getAttackersBlockedBy(blocker).size() == 1
                    && dealsRegularDamage(attacker) && dealsRegularDamage(blocker)) {
                return Optional.of(Binding.empty()
                        .with("attacker", attacker)
                        .with("blocker", blocker));
            }
        }
        return Optional.empty();
    }

    /** Whether the creature deals combat damage, and only in the regular damage step. */
    private static boolean dealsRegularDamage(Card creature) {
        return creature.getNetCombatDamage() > 0
                && !creature.hasFirstStrike() && !creature.hasDoubleStrike();
    }
}
