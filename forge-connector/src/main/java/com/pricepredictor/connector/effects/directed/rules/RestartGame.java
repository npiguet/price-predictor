package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.cost.CostPart;
import forge.game.cost.CostRemoveCounter;
import forge.game.spellability.SpellAbility;

import java.util.Optional;

/**
 * 104.6, an effect that restarts the game, on scenario {@code 104.6-1}: Karn
 * Liberated has 14 loyalty counters in A's main phase, so A can activate its
 * minus-14 ability and restart the game.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and it controls a planeswalker that has activated no loyalty
 * ability this turn and has at least as many loyalty counters as the
 * restart-the-game ability removes, so the ability can be activated. It binds
 * the planeswalker.
 *
 * <p>Variant {@code loyalty_13} ({@code counters}): the planeswalker has one
 * loyalty counter fewer than the ability removes, as Karn has 13 instead of
 * 14, so the ability cannot be activated.
 */
public final class RestartGame {

    public static final String RULE = "104.6";
    static final String SCENARIO = "104.6-1";
    static final String LOYALTY_13 = "loyalty_13";

    private RestartGame() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, RestartGame::bind, LOYALTY_13,
                new Application(ForkOperation.COUNTERS, RestartGame::oneShort));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Card planeswalker : Boards.active(live).getPlaneswalkersInPlay()) {
            Optional<Integer> cost = restartCost(planeswalker);
            if (cost.isPresent() && planeswalker.getPlaneswalkerAbilityActivated() == 0
                    && planeswalker.getCounters(CounterEnumType.LOYALTY) >= cost.get()) {
                return Optional.of(Binding.empty().with("planeswalker", planeswalker));
            }
        }
        return Optional.empty();
    }

    /** How many loyalty counters the card's restart-the-game ability removes, if it has one. */
    static Optional<Integer> restartCost(Card card) {
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (ability.getApi() != ApiType.RestartGame || ability.getPayCosts() == null) {
                continue;
            }
            for (CostPart part : ability.getPayCosts().getCostParts()) {
                if (part instanceof CostRemoveCounter removed && removed.counter != null
                        && removed.counter.is(CounterEnumType.LOYALTY) && removed.convertAmount() != null) {
                    return Optional.of(removed.convertAmount());
                }
            }
        }
        return Optional.empty();
    }

    private static BoardEdit oneShort(Binding fork) {
        Card planeswalker = fork.card("planeswalker");
        int cost = restartCost(planeswalker).orElseThrow(
                () -> new IllegalStateException("the fork's planeswalker restarts no game"));
        return new CountersOperation(planeswalker, CounterEnumType.LOYALTY, cost - 1);
    }
}
