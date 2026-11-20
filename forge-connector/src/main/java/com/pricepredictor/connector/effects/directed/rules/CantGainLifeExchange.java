package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
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
 * begins, it can gain life, and it controls an untapped permanent, under its
 * control since the turn began, with an activated ability that exchanges its
 * own life total with the permanent's toughness, which is above its life, so
 * the exchange would raise it and happens. It binds the permanent and the
 * player.
 *
 * <p>Variant {@code leyline} ({@code add_permanent}): Leyline of Punishment
 * (players can't gain life) is on the battlefield under the player's
 * control, so the exchange, which would raise its life, does not happen: the
 * side of the rule the counts lack. The fork forces the activation (FR-010b),
 * which Forge's AI does not make when the exchange gives it nothing.
 */
public final class CantGainLifeExchange {

    public static final String RULE = "119.7";
    static final String SCENARIO = "119.7-1";
    static final String LEYLINE = "leyline";
    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> PERMANENT = Role.card("permanent");
    static final String LEYLINE_OF_PUNISHMENT = "Leyline of Punishment";

    private CantGainLifeExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantGainLifeExchange::bind, LEYLINE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(LEYLINE_OF_PUNISHMENT, fork.player(PLAYER), false))
                        .forcing(ForcedPlay.activate(PLAYER, PERMANENT,
                                ForcedPlay.Line.api("ExchangeLifeVariant"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!active.canGainLife()) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(permanent -> !permanent.isTapped() && !permanent.isFirstTurnControlled())
                .filter(permanent -> exchangesOwnLife(permanent) && active.getLife() < permanent.getNetToughness())
                .findFirst()
                .map(permanent -> Binding.empty().with(PERMANENT, permanent).with(PLAYER, active));
    }

    /** Whether the card has an activated ability exchanging its controller's own life total with its toughness. */
    static boolean exchangesOwnLife(Card card) {
        return card.getSpellAbilities().stream().anyMatch(CantGainLifeExchange::exchangesToughness);
    }

    private static boolean exchangesToughness(SpellAbility ability) {
        return ability.isActivatedAbility() && ability.getApi() == ApiType.ExchangeLifeVariant
                && "Toughness".equals(ability.getParam("Mode")) && !ability.usesTargeting()
                && "You".equals(ability.getParamOrDefault("Defined", "You"));
    }
}
