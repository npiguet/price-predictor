package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 701.12g, exchanging a life total with a toughness sets the toughness and
 * changes the life total, on scenario {@code 701.12g-1}: A's Tree of
 * Perdition (0/13) exchanges its toughness with B's 7 life, so B goes to 13.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an untapped permanent, under its control since the turn
 * began, with an activated ability that exchanges a target opponent's life
 * total with the permanent's toughness, and an opponent who can gain life has
 * less life than that toughness, so the exchange would raise it. It binds the
 * permanent and that opponent.
 *
 * <p>Variant {@code cannot_gain_life} ({@code add_permanent}): Everlasting
 * Torment (players can't gain life) is on the battlefield under the
 * opponent's control, so no part of the exchange happens.
 */
public final class LifeExchange {

    public static final String RULE = "701.12g";
    static final String SCENARIO = "701.12g-1";
    static final String CANNOT_GAIN_LIFE = "cannot_gain_life";
    static final String TORMENT = "Everlasting Torment";

    private LifeExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, LifeExchange::bind, CANNOT_GAIN_LIFE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(TORMENT, fork.player("opponent"), false)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            if (permanent.isTapped() || permanent.isFirstTurnControlled() || !exchangesToughness(permanent)) {
                continue;
            }
            for (Player opponent : active.getOpponents()) {
                if (opponent.canGainLife() && opponent.getLife() < permanent.getNetToughness()) {
                    return Optional.of(Binding.empty()
                            .with("permanent", permanent)
                            .with("opponent", opponent));
                }
            }
        }
        return Optional.empty();
    }

    /** Whether the card has an activated ability exchanging a targeted player's life with its toughness. */
    static boolean exchangesToughness(Card card) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (ability.isActivatedAbility()
                    && ability.getApi() == ApiType.ExchangeLifeVariant
                    && "Toughness".equals(ability.getParam("Mode"))
                    && ability.usesTargeting()) {
                return true;
            }
        }
        return false;
    }
}
