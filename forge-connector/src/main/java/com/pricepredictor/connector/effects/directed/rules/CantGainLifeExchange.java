package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 119.7, a player who can't gain life can't exchange a life total for a
 * higher one, on scenario {@code 119.7-1}: under Leyline of Punishment, A
 * (5 life) activates Tree of Redemption (0/13), and the exchange, which
 * would raise A's life, does not happen.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it cannot gain life, and it controls an untapped permanent, under
 * its control since the turn began, with an activated ability that exchanges
 * its own life total with the permanent's toughness, which is above its life.
 * It binds the permanent and the player.
 *
 * <p>Variant {@code exchange-lowers} ({@code life}): the player's life total
 * is above the toughness, the scenario's 20 where that is above it, so the
 * exchange would lower it and happens.
 */
public final class CantGainLifeExchange {

    public static final String RULE = "119.7";
    static final String SCENARIO = "119.7-1";
    static final String EXCHANGE_LOWERS = "exchange-lowers";
    /** The life total the scenario's variant gives the player. */
    static final int SCENARIO_LIFE = 20;

    private CantGainLifeExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantGainLifeExchange::bind, EXCHANGE_LOWERS,
                new Application(ForkOperation.LIFE, CantGainLifeExchange::exchangeLowers));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.canGainLife()) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(permanent -> !permanent.isTapped() && !permanent.isFirstTurnControlled())
                .filter(permanent -> exchangesOwnLife(permanent) && active.getLife() < permanent.getNetToughness())
                .findFirst()
                .map(permanent -> Binding.empty().with("permanent", permanent).with("player", active));
    }

    /** Whether the card has an activated ability exchanging its controller's own life total with its toughness. */
    static boolean exchangesOwnLife(Card card) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (ability.isActivatedAbility()
                    && ability.getApi() == ApiType.ExchangeLifeVariant
                    && "Toughness".equals(ability.getParam("Mode"))
                    && !ability.usesTargeting()
                    && "You".equals(ability.getParamOrDefault("Defined", "You"))) {
                return true;
            }
        }
        return false;
    }

    private static BoardEdit exchangeLowers(Binding fork) {
        int toughness = fork.card("permanent").getNetToughness();
        return new LifeOperation(fork.player("player"), Math.max(SCENARIO_LIFE, toughness + 1));
    }
}
