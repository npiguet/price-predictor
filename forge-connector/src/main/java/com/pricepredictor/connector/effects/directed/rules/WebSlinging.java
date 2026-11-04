package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.188a, a spell with web-slinging can be cast for its web-slinging cost
 * by returning a tapped creature its caster controls to hand, on scenario
 * {@code 702.188a-1}: A casts Spider-Man, Web-Slinger for {W}, returning its
 * tapped Grizzly Bears.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with web-slinging whose web-slinging cost it has the
 * lands for, and it controls exactly one tapped creature, the only one it
 * could return. It binds that creature.
 *
 * <p>Variant {@code creature_untapped} ({@code tap}): that creature is
 * untapped, so nothing can be returned and the web-slinging cost cannot be
 * paid. The scenario's other variant taps the land paying the mana instead;
 * this one changes the fact web-slinging adds to the cost.
 */
public final class WebSlinging {

    public static final String RULE = "702.188a";
    static final String SCENARIO = "702.188a-1";
    static final String CREATURE_UNTAPPED = "creature_untapped";

    private WebSlinging() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, WebSlinging::bind, CREATURE_UNTAPPED,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card("creature"), false)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> tapped = active.getCreaturesInPlay().stream().filter(Card::isTapped).toList();
        if (tapped.size() != 1) {
            return Optional.empty();
        }
        boolean slings = active.getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> Boards.canPayKeyword(active, card, Keyword.WEB_SLINGING));
        return slings ? Optional.of(Binding.empty().with("creature", tapped.get(0))) : Optional.empty();
    }
}
