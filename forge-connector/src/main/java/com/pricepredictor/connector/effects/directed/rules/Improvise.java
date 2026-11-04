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
 * 702.126a, improvise lets untapped artifacts pay a spell's generic mana, on
 * scenario {@code 702.126a-1}: A casts Reverse Engineer ({3}{U}{U}) with two
 * Islands and three Ornithopters tapped for the {3}.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and it holds a card with improvise that its untapped lands alone
 * cannot pay for but its untapped lands and its untapped nonland artifacts
 * together pay exactly, the artifacts no more than the card's generic mana. It
 * binds one of those artifacts.
 *
 * <p>Variant {@code one_artifact_tapped} ({@code tap}): that artifact is
 * tapped, so the generic mana is one short and the spell cannot be cast. Of
 * the scenario's two classified variants this is the less invasive (the other
 * removes the artifact).
 */
public final class Improvise {

    public static final String RULE = "702.126a";
    static final String SCENARIO = "702.126a-1";
    static final String ONE_ARTIFACT_TAPPED = "one_artifact_tapped";

    private Improvise() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Improvise::bind, ONE_ARTIFACT_TAPPED,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card("artifact"), true)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> artifacts = active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isArtifact() && !card.isLand() && !card.isTapped())
                .toList();
        int lands = Boards.untappedLands(active);
        if (artifacts.isEmpty()) {
            return Optional.empty();
        }
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            int value = card.getCMC();
            if (card.hasKeyword(Keyword.IMPROVISE) && lands < value && lands + artifacts.size() == value
                    && card.getManaCost().getGenericCost() >= artifacts.size()) {
                return Optional.of(Binding.empty().with("artifact", artifacts.get(0)));
            }
        }
        return Optional.empty();
    }
}
