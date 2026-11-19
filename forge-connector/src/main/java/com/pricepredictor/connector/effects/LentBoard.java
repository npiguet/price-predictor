package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * A read Forge makes of the live board only, asked of the board a snapshot
 * shows ({@link StaticSuppression}): a static's condition on a
 * {@code continuous} record's board.
 *
 * <p>For the read every live card is lent its type line without the
 * suppressed static and every permanent its controller without it; each is
 * put back after, also on a throw, and lending publishes no game event. So a
 * condition that matches or counts types ("as long as you control an
 * artifact", metalcraft, under Mycosynth Lattice) or asks a card's controller
 * (an artifact you control only through Control Magic) reads that board. Not
 * lent: power and toughness, keywords, colours and abilities, and the zone a
 * permanent the static took control of sits in, which Forge moved to its new
 * controller's battlefield, so a count over one player's battlefield (Forge's
 * metalcraft) still counts it as the static's controller's.
 */
final class LentBoard {

    private LentBoard() {
    }

    /** {@code read} answered on {@code board}; on the live board, as it stands. */
    static <T> T read(Game game, StaticSuppression board, Supplier<T> read) {
        if (!board.active()) {
            return read.get();
        }
        List<Runnable> restore = new ArrayList<>();
        try {
            lend(game, board, restore);
            return read.get();
        } finally {
            for (int i = restore.size() - 1; i >= 0; i--) {
                restore.get(i).run();
            }
        }
    }

    /** Lends every live card what differs on {@code board}, adding what puts each back. */
    private static void lend(Game game, StaticSuppression board, List<Runnable> restore) {
        for (Card card : game.getCardsInGame()) {
            keep(restore, board.lendType(card));
        }
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            keep(restore, board.lendController(card));
        }
    }

    private static void keep(List<Runnable> restore, Runnable undo) {
        if (undo != null) {
            restore.add(undo);
        }
    }
}
