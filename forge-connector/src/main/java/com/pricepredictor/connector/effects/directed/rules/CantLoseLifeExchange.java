package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 119.8, a player who can't lose life can't exchange a life total for a lower
 * one, on scenario {@code 119.8-2}: A's Tree of Perdition (0/13) targets B,
 * whose Platinum Emperion keeps B's life total from changing, and the
 * exchange, which would lower B's 20 life, does not happen.
 *
 * <p>Scenario {@code 119.8-1}, a life payment refused for the same reason, is
 * the other board of the rule; this one is the board gen-2's counts measured
 * the rule on, through the exchange detector it shares with 119.7.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an untapped permanent, under its control since the turn
 * began, with an activated ability that exchanges a target opponent's life
 * total with the permanent's toughness, and an opponent with more life than
 * that toughness cannot lose life because of exactly one permanent's static
 * ability (Platinum Emperion). It binds the exchanging permanent, the
 * opponent and the static's permanent.
 *
 * <p>Variant {@code no-emperion} ({@code remove_permanent}): the static's
 * permanent is not on the battlefield, so the exchange happens.
 */
public final class CantLoseLifeExchange {

    public static final String RULE = "119.8";
    static final String SCENARIO = "119.8-2";
    static final String NO_EMPERION = "no-emperion";

    private CantLoseLifeExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantLoseLifeExchange::bind, NO_EMPERION,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("source"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            if (permanent.isTapped() || permanent.isFirstTurnControlled()
                    || !LifeExchange.exchangesToughness(permanent)) {
                continue;
            }
            for (Player opponent : active.getOpponents()) {
                if (opponent.canLoseLife() || opponent.getLife() <= permanent.getNetToughness()) {
                    continue;
                }
                List<Card> sources = cantLoseLifeSources(live, opponent);
                if (sources.size() == 1) {
                    return Optional.of(Binding.empty()
                            .with("permanent", permanent)
                            .with("opponent", opponent)
                            .with("source", sources.get(0)));
                }
            }
        }
        return Optional.empty();
    }

    /** The permanents whose static ability, in force, keeps {@code player} from losing life. */
    static List<Card> cantLoseLifeSources(Game live, Player player) {
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getStaticAbilities().stream().anyMatch(stAb -> stopsLosing(stAb, player)))
                .toList();
    }

    private static boolean stopsLosing(StaticAbility stAb, Player player) {
        return (stAb.checkMode(StaticAbilityMode.CantLoseLife) || stAb.checkMode(StaticAbilityMode.CantChangeLife))
                && stAb.checkConditions()
                && stAb.matchesValidParam("ValidPlayer", player);
    }
}
