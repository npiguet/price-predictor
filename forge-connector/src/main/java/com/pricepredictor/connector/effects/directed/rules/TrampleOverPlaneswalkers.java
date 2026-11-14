package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;

import java.util.Optional;

/**
 * 702.19c, trample over planeswalkers lets the damage beyond an attacked
 * planeswalker's loyalty reach its controller, on scenario
 * {@code 702.19c-1}: Thrasta, Tempest's Roar attacks Jace Beleren (3
 * loyalty) unblocked and assigns 3 damage to Jace and 4 to B.
 *
 * <p>The base board, read at a boundary: the regular combat damage step
 * begins, before its damage is dealt, with an unblocked attacker that has
 * trample over planeswalkers of its own text, strikes in this step and
 * attacks a planeswalker whose loyalty is below its power, so there is
 * excess to assign. Its damage is the step's turn-based action. It binds the
 * attacker.
 *
 * <p>Variant {@code plain_trample} ({@code replace_card}): the attacker is
 * Colossal Dreadmaw, which has trample but not trample over planeswalkers,
 * so all its damage stays on the planeswalker. The precondition binds only an
 * attacker whose Dreadmaw would keep a power above the loyalty, so the
 * fork's attack is still one the rule judges, and would survive what it
 * carries.
 */
public final class TrampleOverPlaneswalkers {

    public static final String RULE = "702.19c";
    static final String SCENARIO = "702.19c-1";
    static final String PLAIN_TRAMPLE = "plain_trample";
    static final String DREADMAW = "Colossal Dreadmaw";

    private TrampleOverPlaneswalkers() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TrampleOverPlaneswalkers::bind, PLAIN_TRAMPLE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("attacker"), DREADMAW)));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DAMAGE) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : Boards.unblockedRegularStrikers(combat)) {
            if (!Boards.tramplesOverPlaneswalkers(attacker) || !Boards.ownTrampleOverPlaneswalkersOnly(attacker)) {
                continue;
            }
            Optional<Card> planeswalker = Boards.attackedPlaneswalker(combat, attacker);
            if (planeswalker.isEmpty()) {
                continue;
            }
            int loyalty = planeswalker.get().getCounters(CounterEnumType.LOYALTY);
            if (attacker.getNetCombatDamage() > loyalty
                    && ReplaceCardOperation.replacementPower(attacker, DREADMAW) > loyalty
                    && Boards.replacementSurvives(attacker, DREADMAW)) {
                return Optional.of(Binding.empty().with("attacker", attacker));
            }
        }
        return Optional.empty();
    }
}
