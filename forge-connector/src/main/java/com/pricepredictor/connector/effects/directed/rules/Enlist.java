package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.List;
import java.util.Optional;

/**
 * 702.154a, an attacking creature with enlist may tap a nonattacking creature
 * without summoning sickness and gets its power, on scenario
 * {@code 702.154a-1}: Barkweave Crusher attacks and enlists Hill Giant, the
 * only other creature A controls.
 *
 * <p>The base board, read at a boundary: the declare-attackers step begins,
 * before attackers are declared, and the active player controls an untapped
 * creature with enlist that can attack this turn and exactly one other
 * untapped creature under its control since the turn began, the one creature
 * it could enlist. It binds both.
 *
 * <p>Variant {@code giant_already_tapped} ({@code tap}): that other creature
 * is already tapped, so nothing can be enlisted.
 */
public final class Enlist {

    public static final String RULE = "702.154a";
    static final String SCENARIO = "702.154a-1";
    static final String ALREADY_TAPPED = "giant_already_tapped";

    private Enlist() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Enlist::bind, ALREADY_TAPPED,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card("enlisted"), true)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_ATTACKERS)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> untapped = Boards.untappedCreatures(active);
        for (Card attacker : untapped) {
            if (!attacker.hasKeyword(Keyword.ENLIST) || attacker.isSick()) {
                continue;
            }
            List<Card> eligible = untapped.stream()
                    .filter(creature -> creature != attacker && !creature.isFirstTurnControlled())
                    .toList();
            if (eligible.size() == 1) {
                return Optional.of(Binding.empty()
                        .with("attacker", attacker)
                        .with("enlisted", eligible.get(0)));
            }
        }
        return Optional.empty();
    }
}
