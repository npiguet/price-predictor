package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
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
 * begins, it controls an Aura with Aura swap whose cost it has the lands for,
 * attached to a creature without protection, and its hand holds cards but no
 * Aura card, so the swap has nothing to exchange with ({@link #withoutAura}).
 * It binds the active player, the hand's first card and the attached Aura.
 *
 * <p>Variant {@code aura_in_hand} ({@code move_card}): a Holy Strength (enchant
 * creature) is in that card's place in the hand, so the swap exchanges it for
 * the attached Aura and attaches it where that one was: the side of the rule
 * the counts lack. The fork forces the swap ({@link #swap}).
 */
public final class AuraSwap {

    public static final String RULE = "701.12e";
    static final String SCENARIO = "701.12e-1";
    static final String AURA_IN_HAND = "aura_in_hand";
    static final String HOLY_STRENGTH = "Holy Strength";

    private AuraSwap() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AuraSwap::bind, AURA_IN_HAND, auraInHand());
    }

    static Optional<Binding> bind(Game live) {
        return withoutAura(live);
    }

    /**
     * Holy Strength in the place of the hand's card the board binds, the swap
     * forced: the variant 701.12e and 702.65a share.
     */
    static Application auraInHand() {
        return new Application(ForkOperation.MOVE_CARD,
                fork -> MoveCardOperation.replace(fork.card("card"), HOLY_STRENGTH))
                .forcing(swap());
    }

    /**
     * The board 701.12e and 702.65a share, read at the start of the active
     * player's main phase: an Aura with Aura swap whose cost it has the lands
     * for, attached to a creature without protection (which Holy Strength,
     * enchant creature, can then enchant), and a hand of cards none of which
     * is an Aura. It binds the player, the hand's first card and the Aura.
     */
    static Optional<Binding> withoutAura(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> hand = List.copyOf(active.getCardsIn(ZoneType.Hand));
        if (hand.isEmpty() || hand.stream().anyMatch(Card::isAura)) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(aura -> aura.isAura() && swappable(aura, active)
                        && aura.getEntityAttachedTo() instanceof Card host && host.isCreature()
                        && !host.hasKeyword(Keyword.PROTECTION))
                .findFirst()
                .map(aura -> Binding.empty().with("player", active).with("card", hand.get(0)).with("aura", aura));
    }

    /**
     * The activation of the attached Aura's Aura swap by the active player,
     * which the fork forces (FR-010b): Forge's AI swaps only for an Aura it
     * prefers, and with nothing or a worse Aura in hand it never activates.
     */
    static ForcedPlay swap() {
        return ForcedPlay.activate("player", "aura", ForcedPlay.Line.keyword("AURA_SWAP"));
    }

    /**
     * The one Aura card in the active player's hand and an attached Aura it
     * could exchange with, read at the start of its main phase: an Aura with
     * Aura swap whose cost it has the lands for, whose host the hand's Aura
     * can enchant ({@code fits}) or cannot. 702.65b's board, with
     * {@code fits} false.
     */
    static Optional<Binding> find(Game live, boolean fits) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> aurasInHand = active.getCardsIn(ZoneType.Hand).stream().filter(Card::isAura).toList();
        if (aurasInHand.size() != 1) {
            return Optional.empty();
        }
        Card card = aurasInHand.get(0);
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(aura -> aura.isAura() && aura.isAttachedToEntity() && swappable(aura, active)
                        && aura.getEntityAttachedTo().canBeAttached(card, null) == fits)
                .findFirst()
                .map(aura -> Binding.empty().with("player", active).with("card", card).with("aura", aura));
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
