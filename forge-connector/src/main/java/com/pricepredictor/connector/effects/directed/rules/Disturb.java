package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 712.11d, a front-face ability letting a double-faced card be cast
 * transformed counts when judging whether it can be cast, on scenario
 * {@code 712.11d-1}: A can cast Beloved Beggar from its graveyard transformed
 * for its disturb cost with six Plains.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and its graveyard holds a card with disturb whose disturb cost it
 * has the lands for. It binds that card.
 *
 * <p>Variant {@code no_disturb} ({@code move_card}): Hanweir Watchkeep, a
 * double-faced card whose front face has no such ability, is in that card's
 * place, so nothing in the graveyard can be cast.
 */
public final class Disturb {

    public static final String RULE = "712.11d";
    static final String SCENARIO = "712.11d-1";
    static final String NO_DISTURB = "no_disturb";
    static final String WATCHKEEP = "Hanweir Watchkeep";

    private Disturb() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Disturb::bind, NO_DISTURB,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.replace(fork.card("card"), WATCHKEEP)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return active.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> Boards.canPayKeyword(active, card, Keyword.DISTURB))
                .findFirst()
                .map(card -> Binding.empty().with("card", card));
    }
}
