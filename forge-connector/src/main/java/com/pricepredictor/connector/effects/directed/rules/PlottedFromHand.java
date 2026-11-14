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
 * Kellan Joins Up, has the untapped lands for it, and holds exactly one other
 * nonland card of mana value 3 or less, the one card the enters trigger could
 * exile; and it cannot cast both, having fewer permanents that make mana now
 * ({@link Boards#manaSources}) than their two mana values: one that could
 * would cast that card beside Kellan, the AI playing the card first, and the
 * trigger would find nothing to exile on either side. It binds that card.
 *
 * <p>Variant {@code card_too_expensive} ({@code move_card}): the card is Hill
 * Giant, of mana value 4, so the trigger finds no card to exile and nothing
 * becomes plotted.
 *
 * <p>It also binds the active player; the fork forces the cast of Kellan
 * Joins Up first (FR-010b), before the AI could cast the other card.
 */
public final class PlottedFromHand {

    public static final String RULE = "702.170c";
    static final String SCENARIO = "702.170c-1";
    static final String CARD_TOO_EXPENSIVE = "card_too_expensive";
    static final String KELLAN = "Kellan Joins Up";
    static final String HILL_GIANT = "Hill Giant";
    static final int MOST_MANA_VALUE = 3;

    private PlottedFromHand() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PlottedFromHand::bind, CARD_TOO_EXPENSIVE,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), HILL_GIANT))
                        .forcing(ForcedPlay.cast("player", "spell")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !live.getStack().isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        CardCollectionView hand = active.getCardsIn(ZoneType.Hand);
        Optional<Card> kellan = hand.stream()
                .filter(card -> KELLAN.equals(card.getName()) && Boards.affordable(active, card)).findFirst();
        if (kellan.isEmpty()) {
            return Optional.empty();
        }
        List<Card> exilable = hand.stream()
                .filter(card -> card != kellan.get() && !card.isLand() && card.getCMC() <= MOST_MANA_VALUE)
                .toList();
        if (exilable.size() != 1
                || Boards.manaSources(active).size() >= kellan.get().getCMC() + exilable.get(0).getCMC()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("player", active)
                .with("card", exilable.get(0)).with("spell", kellan.get()));
    }
}
