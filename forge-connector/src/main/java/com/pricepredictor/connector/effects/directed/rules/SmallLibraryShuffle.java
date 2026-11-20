package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.EmptyLibraryOperation;
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
 * shuffle triggers fire, inverted (FR-014a) on scenario {@code 701.24e-2}:
 * B's Demonic Tutor searches a library of cards and finds one, then the
 * library is shuffled, and A's Cosi's Trickster triggers; with B's library
 * empty the search finds nothing, the empty library is shuffled all the same
 * (Forge shuffles whatever the search found), and the Trickster still
 * triggers.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player
 * controls a permanent that triggers when an opponent shuffles their library
 * (Cosi's Trickster), and an opponent of theirs with a card in their library
 * holds a card they have the mana for whose spell searches their library and
 * that they can cast now ({@link Scan#searchersInHand}: a sorcery only as
 * the active player): the common board, whose library holds cards. It binds
 * the searcher and its spell, whose cast the fork forces (FR-010b).
 *
 * <p>Variant {@code empty_library} ({@code move_card}): the searcher's
 * library is empty, the rule's point, the one edit that makes a library of
 * zero cards out of any other. It moves no card to a graveyard, so no count
 * of one moves.
 */
public final class SmallLibraryShuffle {

    public static final String RULE = "701.24e";
    static final String SCENARIO = "701.24e-2";
    static final String EMPTY_LIBRARY = "empty_library";
    private static final Role<Player> SEARCHER = Role.player("searcher");
    private static final Role<Card> SPELL = Role.card("spell");

    private SmallLibraryShuffle() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SmallLibraryShuffle::bind, new DirectedRule.Variant(EMPTY_LIBRARY,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> new EmptyLibraryOperation(fork.player(SEARCHER)))
                        .forcing(ForcedPlay.cast(SEARCHER, SPELL).playing(Scan.LIBRARY_SEARCH_SPELL))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        return live.getPlayers().stream()
                .filter(SmallLibraryShuffle::watchesOpponentsShuffles)
                .flatMap(watcher -> watcher.getOpponents().stream())
                .map(SmallLibraryShuffle::searcherBinding)
                .flatMap(Optional::stream)
                .findFirst();
    }

    /** The searcher with a library and the first spell of its hand that searches it, or empty. */
    private static Optional<Binding> searcherBinding(Player searcher) {
        List<Card> spells = Scan.searchersInHand(searcher);
        if (searcher.getCardsIn(ZoneType.Library).isEmpty() || spells.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with(SEARCHER, searcher).with(SPELL, spells.get(0)));
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
