package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.63c, each instance of vanishing removes its own time counter, on
 * scenario {@code 702.63c-1}: Maelstrom Djinn has gained vanishing twice,
 * so two upkeep triggers each take a time counter and the second sacrifices
 * it.
 *
 * <p>The base board, read at a boundary: the upkeep step begins and the
 * active player controls a permanent with two or more instances of vanishing,
 * whose triggers the step is about to put on the stack. It binds that
 * permanent.
 *
 * <p>Variant {@code one_instance} ({@code keyword}): the permanent has one
 * instance of vanishing, so one trigger removes one counter.
 */
public final class VanishingInstances {

    public static final String RULE = "702.63c";
    static final String SCENARIO = "702.63c-1";
    static final String ONE_INSTANCE = "one_instance";

    private VanishingInstances() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, VanishingInstances::bind, ONE_INSTANCE,
                new Application(ForkOperation.KEYWORD, fork -> new OneInstance(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP)) {
            return Optional.empty();
        }
        return Boards.active(live).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getAmountOfKeyword(Keyword.VANISHING) >= 2)
                .findFirst()
                .map(card -> Binding.empty().with("permanent", card));
    }

    /**
     * Every instance of vanishing taken away and one given back, through
     * {@link KeywordEdit} as the {@code keyword} operation does: a later
     * keyword layer adds after an earlier one removes, so the permanent ends
     * with exactly one instance, in the spelling its first one had.
     */
    record OneInstance(Card permanent) implements BoardEdit {

        @Override
        public ForkOperation operation() {
            return ForkOperation.KEYWORD;
        }

        @Override
        public void apply(Game fork) {
            BoardEdit.onBattlefield(permanent);
            if (permanent.getAmountOfKeyword(Keyword.VANISHING) < 2) {
                throw new IllegalStateException(permanent + " has fewer than two instances of vanishing");
            }
            List<String> carried = KeywordEdit.carried(permanent, "Vanishing");
            for (String original : carried) {
                KeywordEdit.remove(fork, permanent, original);
            }
            KeywordEdit.add(fork, permanent, carried.get(0));
            BoardEdit.settle(fork);
        }
    }
}
