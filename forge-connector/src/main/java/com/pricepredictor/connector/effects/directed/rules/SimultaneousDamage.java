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
 * 510.2, combat damage is dealt simultaneously, so a creature that dies
 * still deals its damage (703.4m restates it), on scenario {@code 510.2-1}:
 * A's Grizzly Bears attacks and B's Grizzly Bears blocks it, and each deals
 * its 2 damage to the other at the same time.
 *
 * <p>The base board, read at a boundary: the first-strike damage step begins
 * (Forge begins it whenever blockers were declared, and skips its damage when
 * no combatant strikes first) with an attacker blocked by one creature that
 * blocks it alone, both dealing damage and neither with first strike or
 * double strike, so they will deal it to each other in the one regular damage
 * step, the blocker carries nothing the Youthful Knight put in its place
 * would die of ({@link Boards#replacementSurvives}), and the Knight's power,
 * with the boosts and counters it carries, is at least the attacker's
 * toughness less its marked damage, so the attacker dies in the first-strike
 * step having dealt no damage: the detector's {@code fires} side. An attacker
 * the Knight leaves alive shows neither side. It binds the attacker and the
 * blocker.
 *
 * <p>Variant {@code knight_in_the_bears_place} ({@code replace_card}): the blocker
 * is Youthful Knight, a 2/1 with first strike, which deals its damage in the
 * first-strike step instead, carrying the old blocker's place in combat.
 */
public final class SimultaneousDamage {

    public static final String RULE = "510.2";
    static final String SCENARIO = "510.2-1";
    static final String KNIGHT_IN_PLACE = "knight_in_the_bears_place";
    static final String KNIGHT = "Youthful Knight";

    private SimultaneousDamage() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousDamage::bind, KNIGHT_IN_PLACE,
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
            if (trade(combat, attacker, blocker) && knightKills(attacker, blocker)) {
                return Optional.of(Binding.empty()
                        .with("attacker", attacker)
                        .with("blocker", blocker));
            }
        }
        return Optional.empty();
    }

    /**
     * Whether the blocker blocks the attacker alone and the two deal their
     * combat damage to each other in the regular damage step.
     */
    private static boolean trade(Combat combat, Card attacker, Card blocker) {
        return combat.getAttackersBlockedBy(blocker).size() == 1
                && dealsRegularDamage(attacker) && dealsRegularDamage(blocker);
    }

    /**
     * Whether Youthful Knight in the blocker's place survives the edit and
     * deals the attacker lethal damage in the first-strike step: its power,
     * with what it carries ({@link ReplaceCardOperation#replacementPower}),
     * is at least the attacker's toughness less the damage marked on it.
     */
    private static boolean knightKills(Card attacker, Card blocker) {
        return Boards.replacementSurvives(blocker, KNIGHT)
                && ReplaceCardOperation.replacementPower(blocker, KNIGHT)
                >= attacker.getNetToughness() - attacker.getDamage();
    }

    /** Whether the creature deals combat damage, and only in the regular damage step. */
    private static boolean dealsRegularDamage(Card creature) {
        return creature.getNetCombatDamage() > 0
                && !creature.hasFirstStrike() && !creature.hasDoubleStrike();
    }
}
