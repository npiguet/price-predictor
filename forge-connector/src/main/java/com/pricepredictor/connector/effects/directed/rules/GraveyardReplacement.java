package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 400.6, a replacement effect applies to a zone change before the object
 * moves, on scenario {@code 400.6-1}: with Rest in Peace on the battlefield,
 * a Grizzly Bears that Shock kills is exiled instead of going to the
 * graveyard, and so is the Shock.
 *
 * <p>The base board, read at a boundary: a precombat main phase begins with a
 * permanent on the battlefield whose replacement effect sends elsewhere what
 * would be put into a graveyard, from a permanent's static rather than one
 * watching only its own card (Rest in Peace, Leyline of the Void), so the
 * turn's next card to head for a graveyard meets it. It binds that
 * permanent.
 *
 * <p>Variant {@code no_replacement} ({@code remove_permanent}): the permanent
 * is not on the battlefield, so the zone change goes through unreplaced.
 */
public final class GraveyardReplacement {

    public static final String RULE = "400.6";
    static final String SCENARIO = "400.6-1";
    static final String NO_REPLACEMENT = "no_replacement";

    private GraveyardReplacement() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, GraveyardReplacement::bind, NO_REPLACEMENT,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(GraveyardReplacement::replacesGraveyardMoves)
                .findFirst()
                .map(card -> Binding.empty().with("permanent", card));
    }

    /** Whether one of the permanent's replacement effects replaces another card's move to a graveyard. */
    static boolean replacesGraveyardMoves(Card permanent) {
        for (ReplacementEffect replacement : permanent.getReplacementEffects()) {
            if (replacement.getMode() != ReplacementType.Moved
                    || !replacement.hasParam("ReplaceWith")
                    || !replacement.hasParam("Destination")
                    || !replacement.getParam("Destination").contains("Graveyard")) {
                continue;
            }
            String zones = replacement.hasParam("ActiveZones") ? replacement.getParam("ActiveZones") : "Battlefield";
            String valid = replacement.hasParam("ValidCard") ? replacement.getParam("ValidCard") : "Card";
            if (zones.contains("Battlefield") && !valid.startsWith("Card.Self")) {
                return true;
            }
        }
        return false;
    }
}
