package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 701.50f, a permanent connives even when its controller cannot draw or
 * discard, and connive triggers fire, on scenario {@code 701.50f-1}: under
 * Spirit of the Labyrinth A's Echo Inspector connives without drawing,
 * discards Grizzly Bears, and Glorious Purpose triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it cannot draw a card (Spirit of the Labyrinth after its draw), a
 * permanent on the battlefield triggers when a creature connives (Glorious
 * Purpose), and its hand holds exactly two cards: one it has the lands for
 * that connives (Echo Inspector) and one other. It binds the other card,
 * the first of the two that does not connive.
 *
 * <p>Variant {@code nothing_to_discard} ({@code move_card}): that card is at
 * the bottom of the library instead, so once the conniving card is cast the
 * hand is empty and the connive discards nothing; the connives trigger fires
 * as before.
 */
public final class ConniveWithoutDraw {

    public static final String RULE = "701.50f";
    static final String SCENARIO = "701.50f-1";
    static final String NOTHING_TO_DISCARD = "nothing_to_discard";

    private ConniveWithoutDraw() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ConniveWithoutDraw::bind, NOTHING_TO_DISCARD,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Library, null)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> hand = List.copyOf(active.getCardsIn(ZoneType.Hand));
        if (active.canDraw() || hand.size() != 2 || !Boards.watched(live, TriggerType.Connives)
                || !Boards.holdsActing(active, ApiType.Connive)) {
            return Optional.empty();
        }
        return hand.stream()
                .filter(card -> !Boards.acts(card, ApiType.Connive))
                .findFirst()
                .map(card -> Binding.empty().with("card", card));
    }
}
