package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 702.84a, unearth can be activated only from the graveyard and only as a
 * sorcery, on scenario {@code 702.84a-5}: in A's main phase with an empty
 * stack, A may unearth Dregscape Zombie from its graveyard.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins and its graveyard holds a card with unearth whose unearth cost it has
 * the lands for. It binds that card.
 *
 * <p>Variant {@code card_in_hand} ({@code move_card}): the card is in its
 * owner's hand instead, where unearth does not function: its unearth
 * candidate fails the zone check 702.84a-5's declaration reads.
 *
 * <p>The fork keeps the card's decision candidates whatever the decision
 * sample draws (FR-010c): the collector reads their verdicts at each of the
 * fork's priority stops itself, whether or not a seat plays them.
 */
public final class Unearth {

    public static final String RULE = "702.84a";
    static final String SCENARIO = "702.84a-5";
    static final String CARD_IN_HAND = "card_in_hand";

    private static final Role<Card> CARD = Role.card("card");

    private Unearth() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Unearth::bind, CARD_IN_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card(CARD), ZoneType.Hand, null))
                        .keepingCandidatesOf(CARD));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return active.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> Mana.canPayKeyword(active, card, Keyword.UNEARTH))
                .findFirst()
                .map(card -> Binding.empty().with(CARD, card));
    }
}
