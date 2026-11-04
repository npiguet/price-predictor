package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 118.12b, an "if you do" after an optional search asks whether the player
 * chose to search, not what the search found, on scenario
 * {@code 118.12b-1}: Elfhame Sanctuary's upkeep trigger lets A search for a
 * basic land, and A skips its draw step because it searched.
 *
 * <p>The base board, read at a boundary: the upkeep begins, the active player
 * controls a permanent whose optional beginning-of-upkeep trigger searches
 * its library and then, if it does, skips a phase (Elfhame Sanctuary), and
 * exactly one card in its library is of the type the search finds. It binds
 * that card.
 *
 * <p>Variant {@code search_finds_nothing} ({@code move_card}): a Lightning
 * Bolt is in that card's place in the library, so the search finds nothing,
 * and the phase is still skipped if the player chose to search.
 */
public final class OptionalSearch {

    public static final String RULE = "118.12b";
    static final String SCENARIO = "118.12b-1";
    static final String SEARCH_FINDS_NOTHING = "search_finds_nothing";
    static final String BOLT = "Lightning Bolt";

    private OptionalSearch() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, OptionalSearch::bind, SEARCH_FINDS_NOTHING,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), BOLT)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            for (Trigger trigger : permanent.getTriggers()) {
                Optional<String> found = searchedType(trigger);
                if (found.isEmpty()) {
                    continue;
                }
                List<Card> matching = CardLists.getValidCards(active.getCardsIn(ZoneType.Library),
                        found.get().split(","), active, permanent, null);
                if (matching.size() == 1) {
                    return Optional.of(Binding.empty().with("card", matching.get(0)));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The card type an optional upkeep trigger searches its controller's
     * library for, when the search is followed by a skipped phase.
     */
    static Optional<String> searchedType(Trigger trigger) {
        if (trigger.getMode() != TriggerType.Phase || !"Upkeep".equals(trigger.getParam("Phase"))
                || !"You".equals(trigger.getParam("ValidPlayer")) || !trigger.hasParam("OptionalDecider")) {
            return Optional.empty();
        }
        Map<String, String> search = Boards.executed(trigger);
        if (!ApiType.ChangeZone.name().equals(Boards.api(search)) || !"Library".equals(search.get("Origin"))
                || !search.containsKey("ChangeType")) {
            return Optional.empty();
        }
        Map<String, String> then = Boards.script(trigger.getHostCard(), search.get("SubAbility"));
        return ApiType.SkipPhase.name().equals(Boards.api(then))
                ? Optional.of(search.get("ChangeType"))
                : Optional.empty();
    }
}
