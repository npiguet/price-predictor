package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithCost;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 701.12e, Aura swap exchanges an attached Aura with an Aura card in hand,
 * the new one attached where the old one was, on scenario {@code 701.12e-1}:
 * A activates Aura swap on Arcanum Wings on Grizzly Bears and Holy Strength
 * from its hand enters attached to the Bears.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an attached Aura with Aura swap whose cost it has the
 * lands for, and its hand holds exactly one Aura card. It binds that card and
 * the attached Aura.
 *
 * <p>Variant {@code no_aura_in_hand} ({@code move_card}): a Giant Growth is in
 * that card's place in the hand, so there is nothing to exchange with and the
 * attached Aura stays where it is.
 */
public final class AuraSwap {

    public static final String RULE = "701.12e";
    static final String SCENARIO = "701.12e-1";
    static final String NO_AURA_IN_HAND = "no_aura_in_hand";
    static final String GIANT_GROWTH = "Giant Growth";

    private AuraSwap() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AuraSwap::bind, NO_AURA_IN_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), GIANT_GROWTH)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> aurasInHand = active.getCardsIn(ZoneType.Hand).stream().filter(Card::isAura).toList();
        if (aurasInHand.size() != 1) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isAura() && card.isAttachedToEntity() && swappable(card, active))
                .findFirst()
                .map(aura -> Binding.empty().with("card", aurasInHand.get(0)).with("aura", aura));
    }

    /** Whether the Aura has Aura swap and {@code player} has as many untapped lands as its cost's mana value. */
    static boolean swappable(Card aura, Player player) {
        for (KeywordInterface swap : aura.getKeywords(Keyword.AURA_SWAP)) {
            if (swap instanceof KeywordWithCost withCost
                    && Boards.manaValue(withCost.getCost()) <= Boards.untappedLands(player)) {
                return true;
            }
        }
        return false;
    }
}
