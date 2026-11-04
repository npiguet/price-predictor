package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 405.6d, a special action happens at once without using the stack, on
 * scenario {@code 405.6d-1}: A turns its face-down Exalted Angel face up by
 * paying its morph cost {2}{W}{W} with four Plains.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls a face-down permanent whose morph cost is mana alone,
 * and its untapped lands number exactly that cost's mana value with no other
 * permanent of its able to make mana, so the cost takes every land it has
 * untapped. It binds the face-down permanent and one of those lands.
 *
 * <p>Variant {@code cannot_pay} ({@code tap}): the land is already tapped, so
 * the morph cost cannot be paid and turning the permanent face up is not
 * legal.
 */
public final class MorphFaceUp {

    public static final String RULE = "405.6d";
    static final String SCENARIO = "405.6d-1";
    static final String CANNOT_PAY = "cannot_pay";

    private MorphFaceUp() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, MorphFaceUp::bind, CANNOT_PAY,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card("land"), true)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> sources = Boards.manaSources(active);
        if (sources.isEmpty() || !sources.stream().allMatch(Card::isLand)) {
            return Optional.empty();
        }
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            Optional<Integer> cost = morphCost(permanent);
            if (cost.isPresent() && cost.get() == sources.size()) {
                return Optional.of(Binding.empty().with("permanent", permanent).with("land", sources.get(0)));
            }
        }
        return Optional.empty();
    }

    /** The mana value of a face-down permanent's morph cost, when that cost is mana alone. */
    static Optional<Integer> morphCost(Card permanent) {
        if (!permanent.isFaceDown()) {
            return Optional.empty();
        }
        for (SpellAbility ability : permanent.getState(CardStateName.Original).getNonManaAbilities()) {
            if (ability.isTurnFaceUp() && ability.hasParam("MorphUp")
                    && "Morph".equals(ability.getParam("PrecostDesc"))
                    && ability.getPayCosts() != null && ability.getPayCosts().isOnlyManaCost()) {
                int value = Boards.manaValue(ability.getPayCosts());
                return value > 0 ? Optional.of(value) : Optional.empty();
            }
        }
        return Optional.empty();
    }
}
