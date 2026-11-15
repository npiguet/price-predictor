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
 * planeswalker's loyalty reach its controller, inverted (FR-014a) on scenario
 * {@code 702.19c-2}: Colossal Dreadmaw attacks Jace Beleren (3 loyalty)
 * unblocked and all its damage stays on Jace; Thrasta, Tempest's Roar in its
 * place assigns the excess to B.
 *
 * <p>The base board, read at a boundary: the regular combat damage step
 * begins, before its damage is dealt, with an unblocked attacker without
 * trample over planeswalkers that strikes in this step and attacks a
 * planeswalker whose loyalty is below the power Thrasta would have in its
 * place, so the fork's attack has excess to assign: the common board, whose
 * records are the rule's near miss where the attacker's own power is above
 * the loyalty too (the fork's side does not ask it). Its damage is the
 * step's turn-based action. It binds the attacker.
 *
 * <p>Variant {@code thrasta_in_the_dreadmaws_place} ({@code replace_card}):
 * the attacker is Thrasta, which has trample over planeswalkers, carrying the
 * old attacker's place in combat: the rule's {@code fires} side, which the
 * counts lack. The precondition binds only an attacker whose Thrasta would
 * survive what it carries.
 */
public final class TrampleOverPlaneswalkers {

    public static final String RULE = "702.19c";
    static final String SCENARIO = "702.19c-2";
    static final String THRASTA_IN_PLACE = "thrasta_in_the_dreadmaws_place";
    static final String THRASTA = "Thrasta, Tempest's Roar";

    private TrampleOverPlaneswalkers() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TrampleOverPlaneswalkers::bind, THRASTA_IN_PLACE,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("attacker"), THRASTA)));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DAMAGE) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : Boards.unblockedRegularStrikers(combat)) {
            if (Boards.tramplesOverPlaneswalkers(attacker)) {
                continue;
            }
            Optional<Card> planeswalker = Boards.attackedPlaneswalker(combat, attacker);
            if (planeswalker.isEmpty()) {
                continue;
            }
            int loyalty = planeswalker.get().getCounters(CounterEnumType.LOYALTY);
            if (ReplaceCardOperation.replacementPower(attacker, THRASTA) > loyalty
                    && Boards.replacementSurvives(attacker, THRASTA)) {
                return Optional.of(Binding.empty().with("attacker", attacker));
            }
        }
        return Optional.empty();
    }
}
