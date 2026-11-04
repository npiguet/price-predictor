package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.CostDiscard;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.35a, a discarded madness card is exiled instead, and may be cast for its
 * madness cost, on scenario {@code 702.35a-1}: A discards Fiery Temper to
 * Tormenting Voice's cost, Fiery Temper goes to exile and A casts it for {R}.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, its hand holds exactly one card with madness (Fiery Temper) and
 * another card it has the lands for whose spell's cost discards a card
 * (Tormenting Voice). It binds the madness card.
 *
 * <p>Variant {@code no_madness} ({@code move_card}): a Shock is in that card's
 * place in the hand, so the card discarded goes to the graveyard and nothing
 * is exiled.
 */
public final class Madness {

    public static final String RULE = "702.35a";
    static final String SCENARIO = "702.35a-1";
    static final String NO_MADNESS = "no_madness";
    static final String SHOCK = "Shock";

    private Madness() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Madness::bind, NO_MADNESS,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), SHOCK)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> madness = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.hasKeyword(Keyword.MADNESS))
                .toList();
        if (madness.size() != 1) {
            return Optional.empty();
        }
        Card card = madness.get(0);
        boolean outlet = Boards.castableInHand(active, Madness::discardsForCost).stream()
                .anyMatch(other -> other != card);
        return outlet ? Optional.of(Binding.empty().with("card", card)) : Optional.empty();
    }

    /** Whether casting the spell discards a card as part of its cost. */
    static boolean discardsForCost(SpellAbility spell) {
        return spell.getPayCosts() != null && spell.getPayCosts().hasSpecificCostType(CostDiscard.class);
    }
}
