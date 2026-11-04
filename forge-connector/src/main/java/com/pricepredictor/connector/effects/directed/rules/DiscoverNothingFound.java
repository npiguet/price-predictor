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
import forge.game.spellability.SpellAbility;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * 701.57b, a player has discovered even when no card was found, and discover
 * triggers fire, on scenario {@code 701.57b-1}: A's Geological Appraiser
 * discovers 3 over a library with no nonland card of mana value 3 or less,
 * and Val, Marooned Surveyor still triggers.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the lands for that discovers N, N a number
 * of at least 2 (Geological Appraiser), a permanent on the battlefield
 * triggers when a player discovers (Val), and its library is not empty and
 * holds no nonland card of mana value N or less, N the largest such number
 * in its hand. It binds the library's
 * topmost nonland card, its top card when it holds none.
 *
 * <p>Variant {@code card_found} ({@code move_card}): a Grizzly Bears (mana
 * value 2) is in that card's place, so the discovery finds it; the discover
 * trigger fires as before.
 */
public final class DiscoverNothingFound {

    public static final String RULE = "701.57b";
    static final String SCENARIO = "701.57b-1";
    static final String CARD_FOUND = "card_found";
    static final String BEARS = "Grizzly Bears";
    /** Grizzly Bears' mana value: the least discover N it is found by. */
    private static final int BEARS_VALUE = 2;

    private DiscoverNothingFound() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DiscoverNothingFound::bind, CARD_FOUND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), BEARS)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !Boards.watched(live, TriggerType.Discover)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> library = List.copyOf(active.getCardsIn(ZoneType.Library));
        Optional<Integer> most = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> Boards.affordable(active, card))
                .flatMap(DiscoverNothingFound::discovers)
                .filter(n -> n >= BEARS_VALUE)
                .max(Integer::compare);
        if (library.isEmpty() || most.isEmpty() || library.stream()
                .anyMatch(card -> !card.isLand() && card.getCMC() <= most.get())) {
            return Optional.empty();
        }
        Card bound = library.stream().filter(card -> !card.isLand()).findFirst().orElse(library.get(0));
        return Optional.of(Binding.empty().with("card", bound));
    }

    /** The numbers N of the card's discover N, from its spells and the abilities its triggers execute. */
    static Stream<Integer> discovers(Card card) {
        Stream<Map<String, String>> spells = card.getSpells().stream()
                .filter(spell -> spell.getApi() == ApiType.Discover)
                .map(SpellAbility::getMapParams);
        Stream<Map<String, String>> triggered = card.getTriggers().stream()
                .map(Boards::executed)
                .filter(script -> ApiType.Discover.name().equals(Boards.api(script)));
        return Stream.concat(spells, triggered).flatMap(script -> Boards.number(script.get("Num")).stream());
    }
}
