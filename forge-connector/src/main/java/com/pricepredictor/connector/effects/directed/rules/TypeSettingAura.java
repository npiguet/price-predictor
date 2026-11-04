package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.ai.ComputerUtilCard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.player.Player;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 205.1a, an effect that sets an object's card type replaces its types and
 * the subtypes that go with them, on scenario {@code 205.1a-1}: Song of the
 * Dryads makes B's Grizzly Bears a colourless Forest land, no longer a
 * creature or a Bear.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds an Aura it has the lands to cast whose static ability
 * sets the enchanted permanent's card type (removes its card types and gives
 * it another: Song of the Dryads, Imprisoned in the Moon), and an opponent
 * controls a creature with no +1/+1 counter. It binds the creature Forge's
 * AI rates best among those, the one such a curse most likely enchants, and
 * the Aura.
 *
 * <p>Variant {@code counter-kept} ({@code counters}): the creature has one
 * +1/+1 counter, which it keeps when the Aura makes it a land with no power
 * or toughness. The scenario's lacking polarity, {@code fires}, has no
 * classified variant; this is one of its two {@code other} ones, the one that
 * changes a counter rather than the card.
 */
public final class TypeSettingAura {

    public static final String RULE = "205.1a";
    static final String SCENARIO = "205.1a-1";
    static final String COUNTER_KEPT = "counter-kept";

    private TypeSettingAura() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TypeSettingAura::bind, COUNTER_KEPT,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.P1P1, 1)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> aura = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.isAura() && Boards.affordable(active, card) && setsEnchantedType(card))
                .findFirst();
        if (aura.isEmpty()) {
            return Optional.empty();
        }
        List<Card> creatures = active.getOpponents().stream()
                .flatMap(opponent -> opponent.getCreaturesInPlay().stream())
                .filter(creature -> creature.getCounters(CounterEnumType.P1P1) == 0)
                .toList();
        if (creatures.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty()
                .with("creature", ComputerUtilCard.getBestCreatureAI(creatures))
                .with("aura", aura.get()));
    }

    /** Whether one of the card's static abilities removes the enchanted permanent's card types and gives it a type. */
    static boolean setsEnchantedType(Card card) {
        for (StaticAbility stAb : card.getStaticAbilities()) {
            if (stAb.checkMode(StaticAbilityMode.Continuous)
                    && stAb.getParamOrDefault("Affected", "").contains("EnchantedBy")
                    && "True".equals(stAb.getParam("RemoveCardTypes"))
                    && stAb.hasParam("AddType")) {
                return true;
            }
        }
        return false;
    }
}
