package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 505.5, the active player rolls to visit their Attractions as the precombat
 * main phase begins, on scenario {@code 505.5-1}: A controls Balloon Stand,
 * rolls, and the visit ability on the light rolled triggers.
 *
 * <p>The base board, read at a boundary: the precombat main phase begins
 * (before its turn-based actions, the roll among them) and the active player
 * controls an Attraction. It binds that Attraction.
 *
 * <p>Variant {@code no_attraction} ({@code remove_permanent}): the Attraction
 * is not on the battlefield, so the active player controls none and no roll
 * to visit is made.
 */
public final class AttractionVisit {

    public static final String RULE = "505.5";
    static final String SCENARIO = "505.5-1";
    static final String NO_ATTRACTION = "no_attraction";

    private AttractionVisit() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AttractionVisit::bind, NO_ATTRACTION,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("attraction"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        return Boards.active(live).getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isAttraction)
                .findFirst()
                .map(card -> Binding.empty().with("attraction", card));
    }
}
