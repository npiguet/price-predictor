package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import forge.ai.ComputerUtilCard;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 704.5f, a creature with toughness 0 or less is put into its owner's
 * graveyard, on scenario {@code 704.5f-1}: A holds Disfigure in its main
 * phase and B's Drudge Skeletons is a 1/1, so the -2/-2 takes it to
 * toughness -1 and it dies whatever its regeneration shield.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player holds
 * a spell that gives a target creature -N/-N (a pump with a negative
 * toughness number) and has as many untapped lands as its mana value, and an
 * opponent of that player controls a creature of toughness 1 to N, which the
 * spell would take to 0 or less. It binds the creature Forge's AI rates best
 * among those, the one its removal would most likely aim at, and the spell's
 * card. Whether anyone casts the spell is the fork's and the live game's own
 * decision: the precondition only finds the board where the rule can happen.
 *
 * <p>Variant {@code tougher} ({@code pt}): the creature's base toughness rises
 * by N, so the spell leaves it at toughness 1 or more, as Drudge Skeletons is
 * a 1/3 in the scenario's variant.
 */
public final class ToughnessZero {

    public static final String RULE = "704.5f";
    static final String SCENARIO = "704.5f-1";
    static final String TOUGHER = "tougher";

    private ToughnessZero() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ToughnessZero::bind, TOUGHER,
                new Application(ForkOperation.PT, ToughnessZero::tougher));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Player holder : live.getPlayers()) {
            int lands = Boards.untappedLands(holder);
            for (Card spell : holder.getCardsIn(ZoneType.Hand)) {
                int loss = toughnessLoss(spell);
                if (loss == 0 || spell.getCMC() > lands) {
                    continue;
                }
                List<Card> killed = holder.getOpponents().stream()
                        .flatMap(opponent -> opponent.getCreaturesInPlay().stream())
                        .filter(creature -> creature.getNetToughness() >= 1
                                && creature.getNetToughness() <= loss)
                        .toList();
                if (!killed.isEmpty()) {
                    return Optional.of(Binding.empty()
                            .with("creature", ComputerUtilCard.getBestCreatureAI(killed))
                            .with("spell", spell));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The toughness a card's targeted -N/-N spell takes away, 0 when it has
     * none: only a number written into the script counts, since an X is
     * chosen on casting.
     */
    static int toughnessLoss(Card card) {
        int loss = 0;
        for (SpellAbility spell : card.getSpells()) {
            if (spell.getApi() != ApiType.Pump || !spell.usesTargeting() || !spell.hasParam("NumDef")) {
                continue;
            }
            try {
                loss = Math.max(loss, -Integer.parseInt(spell.getParam("NumDef").trim()));
            } catch (NumberFormatException e) {
                // An X or a count: not a fixed -N.
            }
        }
        return loss;
    }

    private static BoardEdit tougher(Binding fork) {
        Card creature = fork.card("creature");
        return new PtOperation(creature, null,
                creature.getCurrentToughness() + toughnessLoss(fork.card("spell")));
    }
}
