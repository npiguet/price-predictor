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

import java.util.Optional;

/**
 * 702.19f, an attacker without trample over planeswalkers deals no combat
 * damage to a player for attacking a planeswalker, on scenario
 * {@code 702.19f-1}: Craw Wurm attacks Jace Beleren unblocked, Jace leaves
 * the battlefield before damage, and the Wurm deals no combat damage.
 *
 * <p>The base board, read at a boundary: the regular combat damage step
 * begins, before its damage is dealt, with an unblocked attacker without
 * trample over planeswalkers that strikes in this step and attacks a
 * planeswalker or battle that has already left combat, whatever removed it
 * (first-strike damage, a removal spell, a sacrifice): Forge keeps it
 * attacking a placeholder (506.4c), and its damage reaches nothing; and
 * another combatant deals combat damage in the step
 * ({@link Combats#anotherDealsRegularDamage}), since a step dealing none
 * writes no combat record. Its damage is the step's turn-based action, which
 * no seat's choice decides beyond the attacker's assignment. It binds the
 * attacker.
 *
 * <p>Variant {@code trample_over_planeswalkers} ({@code replace_card}): the
 * attacker is Thrasta, Tempest's Roar, whose trample over planeswalkers
 * assigns its damage to the departed planeswalker's controller. The
 * precondition binds only an attacker whose Thrasta would survive what it
 * carries.
 */
public final class NoDamageToAGonePlaneswalker {

    public static final String RULE = "702.19f";
    static final String SCENARIO = "702.19f-1";
    static final String TRAMPLE_OVER_PLANESWALKERS = "trample_over_planeswalkers";
    static final String THRASTA = Cards.THRASTA;

    private NoDamageToAGonePlaneswalker() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, NoDamageToAGonePlaneswalker::bind, new DirectedRule.Variant(TRAMPLE_OVER_PLANESWALKERS,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card(Combats.ATTACKER), THRASTA))));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DAMAGE) || combat == null) {
            return Optional.empty();
        }
        return Combats.attackersOfAGoneDefender(combat).stream()
                .filter(attacker -> canTramplePlaneswalkersInstead(combat, attacker))
                .findFirst()
                .map(attacker -> Binding.empty().with(Combats.ATTACKER, attacker));
    }

    private static boolean canTramplePlaneswalkersInstead(Combat combat, Card attacker) {
        return !Boards.tramplesOverPlaneswalkers(attacker) && Combats.anotherDealsRegularDamage(combat, attacker)
                && Boards.replacementSurvives(attacker, THRASTA);
    }
}
