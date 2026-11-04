package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 701.24e, shuffling a library of zero or one card still shuffles it, so
 * shuffle triggers fire, on scenario {@code 701.24e-1}: B's Demonic Tutor
 * takes the Forest from a two-card library and the one-card library left is
 * shuffled, which triggers A's Cosi's Trickster.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player
 * controls a permanent that triggers when an opponent shuffles their library
 * (Cosi's Trickster), and an opponent of theirs with exactly two cards in
 * their library holds a card they have the mana for whose spell searches
 * their library ({@link LimitedSearch#searches}). It binds the bottom card of
 * that library.
 *
 * <p>Variant {@code zero-cards} ({@code move_card}): that card is in its
 * owner's graveyard instead, so the library holds one card, which the search
 * takes, and the library shuffled after it is empty.
 */
public final class SmallLibraryShuffle {

    public static final String RULE = "701.24e";
    static final String SCENARIO = "701.24e-1";
    static final String ZERO_CARDS = "zero-cards";

    private SmallLibraryShuffle() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SmallLibraryShuffle::bind, ZERO_CARDS,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Graveyard, null)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Player watcher : live.getPlayers()) {
            if (!watchesOpponentsShuffles(watcher)) {
                continue;
            }
            for (Player searcher : watcher.getOpponents()) {
                List<Card> library = List.copyOf(searcher.getCardsIn(ZoneType.Library));
                if (library.size() == 2 && !Boards.castableInHand(searcher, LimitedSearch::searches).isEmpty()) {
                    return Optional.of(Binding.empty().with("card", library.get(1)));
                }
            }
        }
        return Optional.empty();
    }

    /** Whether {@code player} controls a permanent that triggers when an opponent shuffles their library. */
    static boolean watchesOpponentsShuffles(Player player) {
        return player.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(SmallLibraryShuffle::onOpponentsShuffle);
    }

    private static boolean onOpponentsShuffle(Trigger trigger) {
        String valid = trigger.getParam("ValidPlayer");
        return trigger.getMode() == TriggerType.Shuffled && valid != null && valid.contains("Opponent");
    }
}
