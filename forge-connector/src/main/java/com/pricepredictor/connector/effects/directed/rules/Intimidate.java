package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ColourTypeOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.Optional;

/**
 * 702.13b, a creature with intimidate can be blocked only by artifact
 * creatures and creatures sharing a colour with it, on scenario
 * {@code 702.13b-1}: Bladetusk Boar (red, intimidate) attacks, and B's green
 * Grizzly Bears is not among its legal blockers.
 *
 * <p>The base board, read at a boundary: the declare-blockers step begins,
 * before blockers are declared, with an attacker that has intimidate and a
 * colour, and its defending player controls an untapped creature that is no
 * artifact and shares no colour with it, so that creature cannot block it.
 * It binds the attacker and that creature.
 *
 * <p>Variant {@code recolored_blocker} ({@code colour_type}): the creature is
 * the attacker's colours instead of its own, so it may block.
 */
public final class Intimidate {

    public static final String RULE = "702.13b";
    static final String SCENARIO = "702.13b-1";
    static final String RECOLORED_BLOCKER = "recolored_blocker";

    private Intimidate() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Intimidate::bind, RECOLORED_BLOCKER,
                new Application(ForkOperation.COLOUR_TYPE, fork -> ColourTypeOperation.colours(
                        fork.card("blocker"), fork.card("attacker").getColor())));
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_BLOCKERS) || combat == null) {
            return Optional.empty();
        }
        for (Card attacker : combat.getAttackers()) {
            if (!attacker.hasKeyword(Keyword.INTIMIDATE) || attacker.getColor().isColorless()) {
                continue;
            }
            Player defender = combat.getDefenderPlayerByAttacker(attacker);
            if (defender == null) {
                continue;
            }
            for (Card creature : Boards.untappedCreatures(defender)) {
                if (!creature.isArtifact() && !creature.getColor().sharesColorWith(attacker.getColor())) {
                    return Optional.of(Binding.empty()
                            .with("attacker", attacker)
                            .with("blocker", creature));
                }
            }
        }
        return Optional.empty();
    }
}
