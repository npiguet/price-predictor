package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.DirectedRule;

/**
 * 702.65a, Aura swap exchanges the Aura with an Aura card in its controller's
 * hand, on scenario {@code 702.65a-1}: A swaps Arcanum Wings on Grizzly Bears
 * for Holy Strength. The keyword's own rule for the exchange 701.12e defines,
 * on the same board ({@link AuraSwap#withoutAura}).
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an Aura with Aura swap whose cost it has the lands for,
 * attached to a creature without protection, and its hand holds cards but no
 * Aura card. It binds the hand's first card and the attached Aura.
 *
 * <p>Variant {@code aura_in_hand} ({@code move_card}): a Holy Strength is in
 * that card's place in the hand, so the swap moves it from the hand to the
 * battlefield: the side of the rule the counts lack. The fork forces the swap
 * ({@link AuraSwap#swap}).
 */
public final class AuraSwapExchange {

    public static final String RULE = "702.65a";
    static final String SCENARIO = "702.65a-1";
    static final String AURA_IN_HAND = "aura_in_hand";

    private AuraSwapExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AuraSwap::withoutAura, AURA_IN_HAND, AuraSwap.auraInHand());
    }
}
