package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
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
    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> CARD = Role.card("card");
    private static final Role<Card> SPELL = Role.card("spell");

    private PlottedFromHand() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, PlottedFromHand::bind, new DirectedRule.Variant(SMALL_CARD_IN_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card(CARD), BEARS))
                        .forcing(ForcedPlay.cast(PLAYER, SPELL))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !live.getStack().isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> kellan = Scan.handCard(active, card -> KELLAN.equals(card.getName()));
        if (kellan.isEmpty()) {
            return Optional.empty();
        }
        List<Card> others = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card != kellan.get() && !card.isLand()).toList();
        if (!onlyBigOthers(others) || couldCastBeside(active, kellan.get())) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with(PLAYER, active)
                .with(CARD, others.get(0)).with(SPELL, kellan.get()));
    }

    /** Whether the player holds another nonland card and none of mana value {@link #MOST_MANA_VALUE} or less. */
    private static boolean onlyBigOthers(List<Card> others) {
        return !others.isEmpty() && others.stream().noneMatch(card -> card.getCMC() <= MOST_MANA_VALUE);
    }

    /** Whether the player has the permanents making mana to cast Kellan beside a card of the Bears' mana value. */
    private static boolean couldCastBeside(Player active, Card kellan) {
        return Mana.manaSources(active).size() >= kellan.getCMC() + BEARS_MANA_VALUE;
    }
}
