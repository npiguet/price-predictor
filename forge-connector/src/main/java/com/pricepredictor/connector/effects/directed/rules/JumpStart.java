package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.133a, jump-start casts a card from its owner's graveyard only with a
 * card to discard, on scenario {@code 702.133a-1}: A can cast Radical Idea
 * from its graveyard, discarding the Forest in its hand.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, its graveyard holds a card with jump-start it has the lands for,
 * and its hand holds exactly one card, the only one it could discard. It
 * binds that card.
 *
 * <p>Variant {@code empty_hand} ({@code move_card}): that card is at the
 * bottom of the library instead, so the hand is empty and the jump-start cast
 * cannot pay its discard.
 */
public final class JumpStart {

    public static final String RULE = "702.133a";
    static final String SCENARIO = "702.133a-1";
    static final String EMPTY_HAND = "empty_hand";

    private JumpStart() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, JumpStart::bind, EMPTY_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Library, null)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> hand = List.copyOf(active.getCardsIn(ZoneType.Hand));
        boolean jumps = active.getCardsIn(ZoneType.Graveyard).stream()
                .anyMatch(card -> card.hasKeyword(Keyword.JUMP_START) && Boards.affordable(active, card));
        return hand.size() == 1 && jumps ? Optional.of(Binding.empty().with("card", hand.get(0))) : Optional.empty();
    }
}
