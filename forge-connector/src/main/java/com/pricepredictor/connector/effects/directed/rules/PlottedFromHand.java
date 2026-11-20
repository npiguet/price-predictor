package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.170c, a spell or ability can make a card in exile plotted, on scenario
 * {@code 702.170c-1}: Kellan Joins Up enters, A exiles Grizzly Bears (mana
 * value 2) from hand with its trigger, and the Bears become plotted.
 *
 * <p>The base board, read at a boundary: a main phase of the active
 * player's turn begins with the stack empty, and the active player holds
 * Kellan Joins Up, has the untapped lands for it, and holds no other nonland
 * card of mana value 3 or less, so the enters trigger would find nothing to
 * exile, but holds another nonland card, of mana value 4 or more; and it
 * could not cast Kellan beside a card of Grizzly Bears' mana value, having
 * fewer permanents that make mana now ({@link Mana#manaSources}) than the
 * two mana values: one that could would cast the Bears beside Kellan, and the
 * trigger would find nothing to exile. It binds the first such card.
 *
 * <p>Variant {@code small_card_in_hand} ({@code move_card}): Grizzly Bears, of
 * mana value 2, is in that card's place, so the trigger exiles it and it
 * becomes plotted: the side of the rule the counts lack.
 *
 * <p>It also binds the active player; the fork forces the cast of Kellan
 * Joins Up first (FR-010b), before the AI could cast the other card.
 */
public final class PlottedFromHand {

    public static final String RULE = "702.170c";
    static final String SCENARIO = "702.170c-1";
    static final String SMALL_CARD_IN_HAND = "small_card_in_hand";
    static final String KELLAN = "Kellan Joins Up";
    static final String BEARS = Cards.BEARS;
    /** Grizzly Bears' mana value. */
    static final int BEARS_MANA_VALUE = 2;
    static final int MOST_MANA_VALUE = 3;

    private PlottedFromHand() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PlottedFromHand::bind, SMALL_CARD_IN_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), BEARS))
                        .forcing(ForcedPlay.cast("player", "spell")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !live.getStack().isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        CardCollectionView hand = active.getCardsIn(ZoneType.Hand);
        Optional<Card> kellan = hand.stream()
                .filter(card -> KELLAN.equals(card.getName()) && Mana.affordable(active, card)).findFirst();
        if (kellan.isEmpty()) {
            return Optional.empty();
        }
        List<Card> others = hand.stream().filter(card -> card != kellan.get() && !card.isLand()).toList();
        if (others.isEmpty() || others.stream().anyMatch(card -> card.getCMC() <= MOST_MANA_VALUE)
                || Mana.manaSources(active).size() >= kellan.get().getCMC() + BEARS_MANA_VALUE) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("player", active)
                .with("card", others.get(0)).with("spell", kellan.get()));
    }
}
