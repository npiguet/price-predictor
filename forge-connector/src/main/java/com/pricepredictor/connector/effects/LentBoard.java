package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * A read Forge makes of the live board only, asked of the board a snapshot
 * shows ({@link StaticSuppression}): a static's condition, and a player's
 * {@code untapped_production}, on a {@code continuous} record's board.
 *
 * <p>For the read every live card is lent its type line without the
 * suppressed static, its colours without it, and every permanent its
 * controller without it; each is put back after, also on a throw, and lending
 * publishes no game event. The type line and the colours are lent through
 * methods the {@code effect-record-hooks} branch adds for the purpose
 * ({@link PatchHooks#LEND_TYPE}, {@link PatchHooks#SET_ASIDE_COLORS}), which
 * {@link PatchHooks#report()} names when a checkout lacks them. So a condition or a mana ability that matches or
 * counts types ("as long as you control an artifact", metalcraft, under
 * Mycosynth Lattice), colours ("each colour among permanents you control",
 * under Painter's Servant) or asks a card's controller (an artifact you
 * control only through Control Magic) reads that board. Not lent: power and
 * toughness, keywords and abilities, and the zone a
 * permanent the static took control of sits in, which Forge moved to its new
 * controller's battlefield, so a count over one player's battlefield (Forge's
 * metalcraft) still counts it as the static's controller's.
 *
 * <p>A static whose condition reads neither (most have no condition) is
 * asked directly, as live: lending would change nothing it reads.
 */
final class LentBoard {

    /**
     * The parameters of Forge's {@code StaticAbility.checkConditions()}
     * whose answer can read a card's type line, colours or controller:
     * {@code Condition$} (the host's controller, metalcraft's artifacts),
     * {@code PlayerTurn$} (players defined from the host),
     * {@code TopCardOfLibraryIs$} and {@code IsPresent$} (cards matched by
     * type, colour and controller), and {@code CheckSVar$} (a count; the
     * {@code CheckSecondSVar$} and later ones are read only after it). Its
     * other reads ({@code Phases$}, {@code GameStage$}, {@code ClassLevel$},
     * the host's zone) read nothing lent.
     */
    private static final List<String> BOARD_CONDITIONS =
            List.of("Condition", "PlayerTurn", "TopCardOfLibraryIs", "IsPresent", "CheckSVar");

    private LentBoard() {
    }

    /**
     * Forge's {@code checkConditions()} of {@code stat} on {@code board},
     * lending the board only when the condition reads what is lent.
     */
    static boolean conditionsHold(Game game, StaticSuppression board, StaticAbility stat) {
        if (!conditionReadsTheBoard(stat)) {
            return stat.checkConditions();
        }
        return read(game, board, stat::checkConditions);
    }

    /**
     * Whether the static's condition reads a card's type line, colours or
     * controller: one that does not answers the same on every board.
     */
    static boolean conditionReadsTheBoard(StaticAbility stat) {
        return BOARD_CONDITIONS.stream().anyMatch(stat::hasParam);
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
            keep(restore, board.lendColors(card));
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
