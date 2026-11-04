package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 702.102c, the total cost of a fused split spell includes both halves' mana
 * costs, on scenario {@code 702.102c-1}: A can cast Give // Take fused for
 * {4}{G}{U} with its six lands.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a split card with fuse, and the permanents it can make mana
 * with are all lands, exactly as many as the fused spell's mana value. It
 * binds one of those lands.
 *
 * <p>Variant {@code one-land-short} ({@code remove_permanent}): that land is
 * gone, so the fused cast costs one mana more than A can make, while each half
 * alone stays castable.
 */
public final class FusedCost {

    public static final String RULE = "702.102c";
    static final String SCENARIO = "702.102c-1";
    static final String ONE_LAND_SHORT = "one-land-short";

    private FusedCost() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, FusedCost::bind, ONE_LAND_SHORT,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("land"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            for (SpellAbility spell : card.getSpells()) {
                if (spell.isKeyword(Keyword.FUSE)) {
                    Optional<Card> land = Boards.lastLandFor(active, Boards.manaValue(spell.getPayCosts()));
                    if (land.isPresent()) {
                        return Optional.of(Binding.empty().with("land", land.get()));
                    }
                }
            }
        }
        return Optional.empty();
    }
}
