package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 205.1a, an effect that sets a card type replaces the existing card types,
 * on scenario {@code 205.1a-1}: A casts Song of the Dryads on B's Grizzly
 * Bears, which becomes a colorless Forest land and no creature.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and it holds an Aura it has the lands for whose static sets the
 * enchanted permanent's card type (it removes every card type and adds its
 * own: Song of the Dryads, Imprisoned in the Moon), beside a creature
 * without +1/+1 counters the Aura can enchant, an opponent's first. It binds
 * the active player, the Aura and the creature.
 *
 * <p>Variant {@code counter-kept} ({@code counters}): the creature has one
 * +1/+1 counter, which stays on it once it is a land. The fork forces the
 * Aura's cast at the creature (FR-010b), whose resolution applies the static
 * the rule is about: Forge's AI casts these Auras rarely, and the counts hold
 * no type set this way.
 */
public final class TypeSettingAura {

    public static final String RULE = "205.1a";
    static final String SCENARIO = "205.1a-1";
    static final String COUNTER_KEPT = "counter-kept";
    private static final Role<Player> CASTER = Role.player("caster");
    private static final Role<Card> AURA = Role.card("aura");
    private static final Role<Card> CREATURE = Role.card("creature");

    private TypeSettingAura() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TypeSettingAura::bind, new DirectedRule.Variant(COUNTER_KEPT,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card(CREATURE), CounterEnumType.P1P1, 1))
                        .forcing(ForcedPlay.cast(CASTER, AURA).targeting(CREATURE))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return Scan.handCard(active, aura -> targetFor(live, active, aura).isPresent())
                .flatMap(aura -> targetFor(live, active, aura).map(creature -> Binding.empty()
                        .with(CASTER, active).with(AURA, aura).with(CREATURE, creature)));
    }

    /**
     * The creature the type-setting Aura would enchant: one with no +1/+1
     * counter that the Aura's cast can target, an opponent's before the
     * active player's; empty for a card that is no such Aura.
     */
    private static Optional<Card> targetFor(Game live, Player active, Card aura) {
        SpellAbility spell = aura.getFirstSpellAbility();
        if (spell == null || !isTypeSettingAura(aura)) {
            return Optional.empty();
        }
        // Targets checked on a copy of the cast, never on the live ability.
        SpellAbility cast = spell.copy(aura, active, true);
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isCreature)
                .filter(card -> card.getCounters(CounterEnumType.P1P1) == 0)
                .filter(card -> cast.canTarget(card))
                .min((a, b) -> Boolean.compare(a.getController() == active, b.getController() == active));
    }

    private static boolean isTypeSettingAura(Card aura) {
        return aura.isAura() && setsCardType(aura);
    }

    /**
     * Whether the Aura's static sets its enchanted permanent's card type: it
     * applies to the enchanted object, removes every card type
     * ({@code RemoveCardTypes$ True}) and adds the ones it names.
     */
    static boolean setsCardType(Card aura) {
        return aura.getStaticAbilities().stream().anyMatch(TypeSettingAura::setsEnchantedCardType);
    }

    private static boolean setsEnchantedCardType(StaticAbility stAb) {
        return "Continuous".equals(stAb.getParam("Mode")) && "True".equals(stAb.getParam("RemoveCardTypes"))
                && stAb.hasParam("AddType") && Scripts.lists(stAb.getParam("Affected"), "Card.EnchantedBy");
    }
}
