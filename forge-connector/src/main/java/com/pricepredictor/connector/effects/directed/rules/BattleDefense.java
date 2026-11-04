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
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 122.1g, a battle with no defense counters is put into its owner's
 * graveyard, on scenario {@code 122.1g-1}: A's Lightning Bolt removes the
 * last 3 defense counters of Invasion of Zendikar.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a spell it has the lands to cast that deals a fixed N
 * damage to a target a battle can be (Lightning Bolt), and a battle on the
 * battlefield has 1 to N defense counters, which the spell would take to 0.
 * It binds the battle and the spell.
 *
 * <p>Variant {@code four_defense} ({@code counters}): the battle has N + 1
 * defense counters, as Invasion of Zendikar has 4 against Lightning Bolt's 3,
 * so the spell leaves it one and it stays.
 */
public final class BattleDefense {

    public static final String RULE = "122.1g";
    static final String SCENARIO = "122.1g-1";
    static final String FOUR_DEFENSE = "four_defense";

    private BattleDefense() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, BattleDefense::bind, FOUR_DEFENSE,
                new Application(ForkOperation.COUNTERS, BattleDefense::oneMore));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        List<Card> battles = live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isBattle() && card.getCounters(CounterEnumType.DEFENSE) > 0)
                .toList();
        if (battles.isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card spell : active.getCardsIn(ZoneType.Hand)) {
            int damage = battleDamage(spell);
            if (damage == 0 || !Boards.affordable(active, spell)) {
                continue;
            }
            for (Card battle : battles) {
                if (battle.getCounters(CounterEnumType.DEFENSE) <= damage) {
                    return Optional.of(Binding.empty().with("battle", battle).with("spell", spell));
                }
            }
        }
        return Optional.empty();
    }

    /** The fixed damage the card's spell deals to a target a battle can be, 0 when it has none. */
    static int battleDamage(Card card) {
        int damage = 0;
        for (SpellAbility spell : card.getSpells()) {
            String targets = spell.getParamOrDefault("ValidTgts", "");
            if (spell.getApi() == ApiType.DealDamage && spell.usesTargeting()
                    && (targets.contains("Any") || targets.contains("Battle"))) {
                damage = Math.max(damage, Boards.number(spell.getParam("NumDmg")).orElse(0));
            }
        }
        return damage;
    }

    private static BoardEdit oneMore(Binding fork) {
        return new CountersOperation(fork.card("battle"), CounterEnumType.DEFENSE,
                battleDamage(fork.card("spell")) + 1);
    }
}
