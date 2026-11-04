package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.List;
import java.util.Optional;

/**
 * 701.24g, a card put on top of a library at the same time the library is
 * shuffled stays on top, on scenario {@code 701.24g-1}: B's Barter in Blood
 * makes A sacrifice Darksteel Colossus, shuffled into A's library, and
 * Gravebane Zombie, put on top of it, and the Zombie is the top card after.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a spell it has the mana for that makes players sacrifice
 * two or more creatures at once ({@link SimultaneousShuffles#massSacrifice}),
 * and a player controls exactly two creatures, one shuffled into its owner's
 * library instead of going to a graveyard (Darksteel Colossus), the other put
 * on top of it instead (Gravebane Zombie). It binds the shuffled one.
 *
 * <p>Variant {@code zombie-only} ({@code replace_card}): Grizzly Bears is in
 * that creature's place, so the sacrifice puts the other on top and shuffles
 * nothing.
 */
public final class ShuffleAndPlaceOnTop {

    public static final String RULE = "701.24g";
    static final String SCENARIO = "701.24g-1";
    static final String ZOMBIE_ONLY = "zombie-only";

    private ShuffleAndPlaceOnTop() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ShuffleAndPlaceOnTop::bind, ZOMBIE_ONLY,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), SimultaneousShuffles.BEARS)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)
                || Boards.castableInHand(Boards.active(live), SimultaneousShuffles::massSacrifice).isEmpty()) {
            return Optional.empty();
        }
        for (Player player : live.getPlayers()) {
            List<Card> creatures = List.copyOf(player.getCreaturesInPlay());
            if (creatures.size() != 2) {
                continue;
            }
            Optional<Card> shuffled = creatures.stream().filter(SimultaneousShuffles::shufflesIntoLibrary).findFirst();
            boolean placed = creatures.stream().anyMatch(ShuffleAndPlaceOnTop::goesOnTop);
            if (shuffled.isPresent() && placed) {
                return Optional.of(Binding.empty().with("creature", shuffled.get()));
            }
        }
        return Optional.empty();
    }

    /** Whether the card is put on top of its owner's library instead of going to a graveyard (Gravebane Zombie). */
    static boolean goesOnTop(Card card) {
        return SimultaneousShuffles.ownGraveyardReplacement(card)
                .map(replacing -> "0".equals(replacing.get("LibraryPosition"))
                        && !"True".equals(replacing.get("Shuffle")))
                .orElse(false);
    }
}
