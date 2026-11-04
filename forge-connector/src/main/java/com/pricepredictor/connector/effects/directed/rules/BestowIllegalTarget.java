package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.103e, a bestowed Aura spell whose target became illegal resolves as a
 * creature spell, on scenario {@code 702.103e-1}: Hopeful Eidolon is cast
 * bestowed on Grizzly Bears, Shock kills the Bears in response, and the
 * Eidolon enters as a creature.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with bestow whose bestow cost it has the lands for,
 * and it controls exactly one creature, the one the Aura would enchant, with
 * toughness less than 3, small enough for a Shock to kill. It binds that
 * creature. A directed fork is taken at a phase boundary, where the stack is
 * empty, so the base is the board before the bestow cast rather than the
 * spell on the stack.
 *
 * <p>Variant {@code target-survives} ({@code replace_card}): Centaur Courser
 * (3/3) is in that creature's place, so the damage that would have made the
 * target illegal leaves it alive and the Aura attaches to it.
 */
public final class BestowIllegalTarget {

    public static final String RULE = "702.103e";
    static final String SCENARIO = "702.103e-1";
    static final String TARGET_SURVIVES = "target-survives";
    static final String COURSER = "Centaur Courser";
    /** Centaur Courser's toughness: a creature with less is the one it outlives a Shock in place of. */
    private static final int COURSER_TOUGHNESS = 3;

    private BestowIllegalTarget() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, BestowIllegalTarget::bind, TARGET_SURVIVES,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), COURSER)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> creatures = List.copyOf(active.getCreaturesInPlay());
        if (creatures.size() != 1 || creatures.get(0).getNetToughness() >= COURSER_TOUGHNESS) {
            return Optional.empty();
        }
        boolean bestows = active.getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> Boards.canPayKeyword(active, card, Keyword.BESTOW));
        return bestows ? Optional.of(Binding.empty().with("creature", creatures.get(0))) : Optional.empty();
    }
}
