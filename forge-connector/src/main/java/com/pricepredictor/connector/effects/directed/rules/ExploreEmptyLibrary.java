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

import java.util.Optional;
import java.util.stream.Stream;

/**
 * 701.44b, a permanent explores even when no card can be revealed, on
 * scenario {@code 701.44b-1}: A's Merfolk Branchwalker explores with A's
 * library empty, gets its counter, and Wildgrowth Walker still triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, its library is empty, it holds a card it has the lands for that
 * explores (Merfolk Branchwalker), a permanent on the battlefield triggers
 * when a creature explores (Wildgrowth Walker), and a land card is in its
 * graveyard or, failing that, its hand. It binds that land card.
 *
 * <p>Variant {@code land_on_top} ({@code move_card}): the land card is the
 * library's one card, so the explore reveals it and puts it into the hand;
 * the explores trigger fires as before.
 */
public final class ExploreEmptyLibrary {

    public static final String RULE = "701.44b";
    static final String SCENARIO = "701.44b-1";
    static final String LAND_ON_TOP = "land_on_top";

    private ExploreEmptyLibrary() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ExploreEmptyLibrary::bind, LAND_ON_TOP,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("land"), ZoneType.Library, 0)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!active.getCardsIn(ZoneType.Library).isEmpty() || !Boards.watched(live, TriggerType.Explores)
                || !Boards.holdsActing(active, ApiType.Explore)) {
            return Optional.empty();
        }
        return Stream.concat(active.getCardsIn(ZoneType.Graveyard).stream(), active.getCardsIn(ZoneType.Hand).stream())
                .filter(Card::isLand)
                .findFirst()
                .map(land -> Binding.empty().with("land", land));
    }
}
