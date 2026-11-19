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
 * begins, and it holds a card with improvise whose mana value the mana its
 * other sources make ({@link #otherMana}) and its untapped nonland artifacts,
 * none of which makes mana itself, pay exactly, the artifacts no more than
 * the card's generic mana, and which the player can pay, colours included
 * ({@link Boards#affordable(Player, Card)}: every untapped artifact
 * improvises, as the rules allow, whatever the AI would tap). It binds one
 * of those artifacts and the card.
 *
 * <p>The active player cannot add a land to its mana this turn
 * ({@link Boards#noLandToPlay}): it holds no land, or its land drop is
 * used, so the turn's land drop cannot undo the mana the variant takes.
 *
 * <p>Variant {@code one_artifact_tapped} ({@code tap}): that artifact is
 * tapped, so the generic mana is one short and the spell cannot be cast. Of
 * the scenario's two classified variants this is the less invasive (the other
 * removes the artifact).
 *
 * <p>The fork keeps the card's decision candidates whatever the decision
 * sample draws (FR-010c): the collector reads their verdicts at each of the
 * fork's priority stops itself, whether or not a seat plays them.
 */
public final class Improvise {

    public static final String RULE = "702.126a";
    static final String SCENARIO = "702.126a-1";
    static final String ONE_ARTIFACT_TAPPED = "one_artifact_tapped";

    private Improvise() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Improvise::bind, ONE_ARTIFACT_TAPPED,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card("artifact"), true))
                        .keepingCandidatesOf("card"));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!Boards.noLandToPlay(active)) {
            return Optional.empty();
        }
        List<Card> artifacts = active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isArtifact() && !card.isLand() && !card.isTapped())
                .toList();
        if (artifacts.isEmpty() || artifacts.stream().anyMatch(artifact -> Boards.manaUnits(artifact) > 0)) {
            return Optional.empty();
        }
        int mana = otherMana(active, artifacts);
        return active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.hasKeyword(Keyword.IMPROVISE) && card.getCMC() == mana + artifacts.size())
                .filter(card -> card.getManaCost().getGenericCost() >= artifacts.size())
                .filter(card -> Boards.affordable(active, card))
                .findFirst()
                .map(card -> Binding.empty().with("artifact", artifacts.get(0)).with("card", card));
    }

    /**
     * The mana {@code player}'s sources other than {@code artifacts} make now
     * ({@link Boards#manaSources}), each one activation of its most
     * productive mana ability ({@link Boards#manaUnits}): a bounce land
     * counts two, a mana creature one.
     */
    static int otherMana(Player player, List<Card> artifacts) {
        return Boards.manaSources(player).stream()
                .filter(source -> !artifacts.contains(source))
                .mapToInt(Boards::manaUnits)
                .sum();
    }
}
