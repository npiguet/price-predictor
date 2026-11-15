package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
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

    private TypeSettingAura() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TypeSettingAura::bind, COUNTER_KEPT,
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.P1P1, 1))
                        .forcing(ForcedPlay.cast("caster", "aura").targeting("creature")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card aura : active.getCardsIn(ZoneType.Hand)) {
            if (!aura.isAura() || !setsCardType(aura) || !Boards.affordable(active, aura)) {
                continue;
            }
            SpellAbility spell = aura.getFirstSpellAbility();
            if (spell == null) {
                continue;
            }
            // Targets checked on a copy of the cast, never on the live ability.
            SpellAbility cast = spell.copy(aura, active, true);
            Optional<Card> creature = live.getCardsIn(ZoneType.Battlefield).stream()
                    .filter(Card::isCreature)
                    .filter(card -> card.getCounters(CounterEnumType.P1P1) == 0)
                    .filter(card -> cast.canTarget(card))
                    .min((a, b) -> Boolean.compare(a.getController() == active, b.getController() == active));
            if (creature.isPresent()) {
                return Optional.of(Binding.empty().with("caster", active).with("aura", aura)
                        .with("creature", creature.get()));
            }
        }
        return Optional.empty();
    }

    /**
     * Whether the Aura's static sets its enchanted permanent's card type: it
     * applies to the enchanted object, removes every card type
     * ({@code RemoveCardTypes$ True}) and adds the ones it names.
     */
    static boolean setsCardType(Card aura) {
        for (StaticAbility stAb : aura.getStaticAbilities()) {
            if ("Continuous".equals(stAb.getParam("Mode")) && "True".equals(stAb.getParam("RemoveCardTypes"))
                    && stAb.hasParam("AddType") && Boards.lists(stAb.getParam("Affected"), "Card.EnchantedBy")) {
                return true;
            }
        }
        return false;
    }
}
