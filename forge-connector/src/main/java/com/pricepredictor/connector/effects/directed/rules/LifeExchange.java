package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
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
 * 701.12g, exchanging a life total with a toughness sets the toughness and
 * changes the life total, on scenario {@code 701.12g-1}: A's Tree of
 * Perdition (0/13) exchanges its toughness with B's 7 life, so B goes to 13.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an untapped permanent, under its control since the turn
 * began, with an activated ability that exchanges a target opponent's life
 * total with the permanent's toughness, and an opponent who can lose life has
 * no more life than that toughness, so the exchange would not lower it and
 * Forge's AI does not make it. It binds the active player, the permanent and
 * that opponent.
 *
 * <p>Variant {@code opponent_above} ({@code life}): the opponent's life is
 * above the toughness, the scenario's 20 where that is above it, so the
 * exchange lowers it and sets the toughness: the exchange happens, the side
 * of the rule the counts lack. The fork forces the activation at that
 * opponent (FR-010b).
 */
public final class LifeExchange {

    public static final String RULE = "701.12g";
    static final String SCENARIO = "701.12g-1";
    static final String OPPONENT_ABOVE = "opponent_above";
    /** The life total the scenario's variant gives the opponent. */
    static final int SCENARIO_LIFE = 20;
    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> PERMANENT = Role.card("permanent");
    private static final Role<Player> OPPONENT = Role.player("opponent");

    private LifeExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, LifeExchange::bind, new DirectedRule.Variant(OPPONENT_ABOVE,
                new Application(ForkOperation.LIFE, LifeExchange::opponentAbove)
                        .forcing(ForcedPlay.activate(PLAYER, PERMANENT,
                                ForcedPlay.Line.api("ExchangeLifeVariant")).targeting(OPPONENT))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(permanent -> !permanent.isTapped() && !permanent.isFirstTurnControlled())
                .filter(LifeExchange::exchangesToughness)
                .flatMap(permanent -> active.getOpponents().stream()
                        .filter(opponent -> opponent.canLoseLife() && opponent.getLife() <= permanent.getNetToughness())
                        .map(opponent -> Binding.empty()
                                .with(PLAYER, active)
                                .with(PERMANENT, permanent)
                                .with(OPPONENT, opponent)))
                .findFirst();
    }

    /** Whether the card has an activated ability exchanging a targeted player's life with its toughness. */
    static boolean exchangesToughness(Card card) {
        return card.getSpellAbilities().stream().anyMatch(LifeExchange::exchangesTargetedLife);
    }

    private static boolean exchangesTargetedLife(SpellAbility ability) {
        return ability.isActivatedAbility() && ability.getApi() == ApiType.ExchangeLifeVariant
                && "Toughness".equals(ability.getParam("Mode")) && ability.usesTargeting();
    }

    private static BoardEdit opponentAbove(Binding fork) {
        int toughness = fork.card(PERMANENT).getNetToughness();
        return new LifeOperation(fork.player(OPPONENT), Math.max(SCENARIO_LIFE, toughness + 1));
    }
}
