package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;

import java.util.Optional;

/**
 * 702.65b, the exchange fails when the Aura card in hand cannot enchant what
 * the swapped Aura enchants, on scenario {@code 702.65b-1}: A's Arcanum Wings
 * on Grizzly Bears cannot swap for Wild Growth (enchant land).
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an Aura with Aura swap whose cost it has the lands for,
 * attached to a creature, and its hand holds exactly one Aura card, which
 * cannot enchant that creature ({@link AuraSwap#find}). It binds that card and
 * the attached Aura.
 *
 * <p>Variant {@code can_enchant} ({@code move_card}): a Holy Strength (enchant
 * creature) is in that card's place in the hand, so the exchange succeeds.
 */
public final class AuraSwapCantEnchant {

    public static final String RULE = "702.65b";
    static final String SCENARIO = "702.65b-1";
    static final String CAN_ENCHANT = "can_enchant";
    static final String HOLY_STRENGTH = "Holy Strength";

    private AuraSwapCantEnchant() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AuraSwapCantEnchant::bind, CAN_ENCHANT,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), HOLY_STRENGTH)));
    }

    static Optional<Binding> bind(Game live) {
        return AuraSwap.find(live, false).filter(binding ->
                binding.card("aura").getEntityAttachedTo() instanceof Card host && host.isCreature());
    }
}
