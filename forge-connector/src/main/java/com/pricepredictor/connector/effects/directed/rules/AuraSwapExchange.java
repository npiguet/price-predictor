package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;

/**
 * 702.65a, Aura swap exchanges the Aura with an Aura card in its controller's
 * hand, on scenario {@code 702.65a-1}: A swaps Arcanum Wings on Grizzly Bears
 * for Holy Strength. The keyword's own rule for the exchange 701.12e defines,
 * on the same board ({@link AuraSwap#find}).
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an attached Aura with Aura swap whose cost it has the
 * lands for, and its hand holds exactly one Aura card, which can enchant what
 * the attached Aura enchants. It binds that card and the attached Aura.
 *
 * <p>Variant {@code no_aura_in_hand} ({@code move_card}): a Grizzly Bears is
 * in that card's place in the hand, so there is no Aura card to exchange with.
 */
public final class AuraSwapExchange {

    public static final String RULE = "702.65a";
    static final String SCENARIO = "702.65a-1";
    static final String NO_AURA_IN_HAND = "no_aura_in_hand";
    static final String BEARS = "Grizzly Bears";

    private AuraSwapExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, live -> AuraSwap.find(live, true), NO_AURA_IN_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), BEARS)));
    }
}
