package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.Role;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;

import java.util.List;
import java.util.Optional;
import java.util.function.BiPredicate;

/** What a precondition reads off a live combat: who strikes when, who is blocked, what is attacked. Read only. */
final class Combats {

    /** The role of the attacker a combat rule binds. */
    static final Role<Card> ATTACKER = Role.card("attacker");
    /** The role of the creature its attacker keeps from blocking. */
    static final Role<Card> BLOCKER = Role.card("blocker");

    private Combats() {
    }

    /**
     * The attackers of the live combat that deal their combat damage in the
     * regular damage step and none blocks: neither first strike nor double
     * strike, power above 0, and no blocker declared against them.
     */
    static List<Card> unblockedRegularStrikers(Combat combat) {
        return combat.getAttackers().stream()
                .filter(attacker -> !attacker.hasFirstStrike() && !attacker.hasDoubleStrike())
                .filter(attacker -> attacker.getNetCombatDamage() > 0)
                .filter(attacker -> !combat.isBlocked(attacker) && combat.getBlockers(attacker).isEmpty())
                .toList();
    }

    /**
     * The unblocked regular-step attackers ({@link #unblockedRegularStrikers})
     * whose attacked planeswalker or battle has left combat: Forge keeps them
     * attacking a placeholder of no zone (506.4c, {@code Combat.isRemovedDefender}).
     */
    static List<Card> attackersOfAGoneDefender(Combat combat) {
        return unblockedRegularStrikers(combat).stream()
                .filter(attacker -> Combat.isRemovedDefender(combat.getDefenderByAttacker(attacker)))
                .toList();
    }

    /**
     * Whether a combatant of the live combat other than {@code attacker} deals
     * combat damage in the regular damage step: an attacker or blocker with
     * power above 0 that strikes in it (double strike, or no first strike)
     * and has something to hit (an attacker its blockers, or unblocked a
     * player or a planeswalker or battle still in combat; a blocker the
     * attackers it blocks). A damage step in which no combat damage is dealt
     * writes no combat record ({@code CombatStepBracket}), so an attacker
     * dealing none is recorded only beside another's damage.
     */
    static boolean anotherDealsRegularDamage(Combat combat, Card attacker) {
        for (Card other : combat.getAttackers()) {
            if (other != attacker && dealsRegularDamage(combat, other)) {
                return true;
            }
        }
        return combat.getAllBlockers().stream().anyMatch(blocker -> strikesInTheRegularStep(blocker)
                && !combat.getAttackersBlockedBy(blocker).isEmpty());
    }

    private static boolean strikesInTheRegularStep(Card combatant) {
        return combatant.getNetCombatDamage() > 0 && (combatant.hasDoubleStrike() || !combatant.hasFirstStrike());
    }

    private static boolean dealsRegularDamage(Combat combat, Card attacker) {
        if (!strikesInTheRegularStep(attacker)) {
            return false;
        }
        if (!combat.getBlockers(attacker).isEmpty()) {
            return true;
        }
        return !combat.isBlocked(attacker) && !Combat.isRemovedDefender(combat.getDefenderByAttacker(attacker));
    }

    /** The planeswalker on the battlefield {@code attacker} attacks, or empty when it attacks none. */
    static Optional<Card> attackedPlaneswalker(Combat combat, Card attacker) {
        return combat.getDefenderByAttacker(attacker) instanceof Card defender
                && defender.isPlaneswalker() && defender.isInPlay()
                ? Optional.of(defender) : Optional.empty();
    }

    /**
     * The first attacker of the live combat with the first untapped creature of
     * the player it attacks that {@code kept} accepts (the attacker and the
     * creature: the creature its attacker keeps from blocking), bound as
     * {@code attacker} and {@code blocker}; empty when no attacker has a
     * defending player or none of that player's creatures is accepted.
     * Attackers are tried in combat order, a defender's creatures in the order
     * it controls them.
     */
    static Optional<Binding> blockersKeptOut(Combat combat, BiPredicate<Card, Card> kept) {
        for (Card attacker : combat.getAttackers()) {
            Player defender = combat.getDefenderPlayerByAttacker(attacker);
            if (defender == null) {
                continue;
            }
            Optional<Card> blocker = Boards.untappedCreatures(defender).stream()
                    .filter(creature -> kept.test(attacker, creature))
                    .findFirst();
            if (blocker.isPresent()) {
                return Optional.of(Binding.empty().with(ATTACKER, attacker).with(BLOCKER, blocker.get()));
            }
        }
        return Optional.empty();
    }
}
