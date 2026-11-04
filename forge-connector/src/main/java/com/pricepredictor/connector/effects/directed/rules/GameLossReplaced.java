package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 704.7, a single replacement effect replaces every game loss one
 * state-based check proposes, on scenario {@code 704.7-1}: A, at 1 life with
 * one card in its library, casts Night's Whisper, and Lich's Mirror replaces
 * both losses at once.
 *
 * <p>The base board, read at a boundary: a player controls a permanent with a
 * replacement effect for its own game loss, and that player is close to
 * losing: 5 life or less, or one card or none left in its library. It binds
 * that permanent.
 *
 * <p>Variant {@code no_mirror} ({@code remove_permanent}): that permanent is
 * gone, so nothing replaces the losses.
 */
public final class GameLossReplaced {

    public static final String RULE = "704.7";
    static final String SCENARIO = "704.7-1";
    static final String NO_MIRROR = "no_mirror";
    /** The life total at or below which a player is close to losing for life. */
    static final int LOW_LIFE = 5;

    private GameLossReplaced() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, GameLossReplaced::bind, NO_MIRROR,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        for (Card permanent : live.getCardsIn(ZoneType.Battlefield)) {
            Player controller = permanent.getController();
            if (closeToLosing(controller) && replacesOwnLoss(permanent)) {
                return Optional.of(Binding.empty().with("permanent", permanent));
            }
        }
        return Optional.empty();
    }

    private static boolean closeToLosing(Player player) {
        return player.getLife() <= LOW_LIFE || player.getCardsIn(ZoneType.Library).size() <= 1;
    }

    /** Whether one of the permanent's replacement effects replaces its controller's game loss. */
    private static boolean replacesOwnLoss(Card permanent) {
        for (ReplacementEffect effect : permanent.getReplacementEffects()) {
            if (effect.getMode() == ReplacementType.GameLoss && "You".equals(effect.getParam("ValidPlayer"))
                    && Boards.lists(effect.getParam("ActiveZones"), "Battlefield")) {
                return true;
            }
        }
        return false;
    }
}
