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
 * 702.19e, an attacker with trample over planeswalkers whose planeswalker
 * left combat assigns its damage to that planeswalker's controller, on
 * scenario {@code 702.19e-1}: Thrasta, Tempest's Roar attacks Jace Beleren
 * unblocked, Jace leaves the battlefield before damage, and Thrasta deals
 * all 7 to B.
 *
 * <p>The base board, read at a boundary: the regular combat damage step
 * begins, before its damage is dealt, with an unblocked attacker that has
 * trample over planeswalkers of its own text, strikes in this step, and
 * attacks a planeswalker or battle that has already left combat, whatever
 * removed it (first-strike damage, a removal spell, a sacrifice): Forge
 * keeps it attacking a placeholder (506.4c); and another combatant deals
 * combat damage in the step ({@link Boards#anotherDealsRegularDamage}),
 * since a step dealing none writes no combat record and the variant's
 * attacker deals none. Its damage is the step's turn-based action, which no
 * seat's choice decides beyond the attacker's assignment. It binds the
 * attacker.
 *
 * <p>Variant {@code plain_trample} ({@code replace_card}): the attacker is
 * Colossal Dreadmaw, which has trample but not trample over planeswalkers,
 * so it assigns its damage to the departed planeswalker alone and none
 * reaches B. The precondition binds only an attacker whose Dreadmaw would
 * survive what it carries.
 */
public final class TrampleOverAGonePlaneswalker {

    public static final String RULE = "702.19e";
    static final String SCENARIO = "702.19e-1";
    static final String PLAIN_TRAMPLE = "plain_trample";
    static final String DREADMAW = TrampleOverPlaneswalkers.DREADMAW;

    private TrampleOverAGonePlaneswalker() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TrampleOverAGonePlaneswalker::bind, PLAIN_TRAMPLE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("attacker"), DREADMAW)));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DAMAGE) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : Boards.attackersOfAGoneDefender(combat)) {
            if (Boards.tramplesOverPlaneswalkers(attacker) && Boards.ownTrampleOverPlaneswalkersOnly(attacker)
                    && Boards.anotherDealsRegularDamage(combat, attacker)
                    && Boards.replacementSurvives(attacker, DREADMAW)) {
                return Optional.of(Binding.empty().with("attacker", attacker));
            }
        }
        return Optional.empty();
    }
}
