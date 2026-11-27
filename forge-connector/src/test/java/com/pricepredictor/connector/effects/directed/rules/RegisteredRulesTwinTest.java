package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the twin itself catches: an edit the fork's first state-based
 * check undoes or ends the game with, and only a Keyword Effects card the
 * check tidies forgiven ({@link RegisteredRulesTest} sets out the twin).
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesTwinTest {

    // ── what the twin itself catches ────────────────────────────────────

    /** A rule of one variant on the active player, for the twin's own tests. */
    private static DirectedRule onePlayerRule(ForkOperation operation,
                                              Function<Binding, BoardEdit> edit) {
        return DirectedRule.of("0.0", "0.0-1",
                live -> Optional.of(Binding.empty().with("player", Boards.active(live))),
                new DirectedRule.Variant("v", new Application(operation, edit)));
    }

    /** The first check ending the fork's game moves no card, and the twin still sees it. */
    @Test
    void theTwinCatchesAnEditTheFirstCheckLosesAPlayerWith() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        DirectedRule zeroLife = onePlayerRule(ForkOperation.LIFE, fork -> new LifeOperation(fork.player("player"), 0));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(zeroLife));
        assertTrue(error.getMessage().contains("the first state-based check"), error.getMessage());
    }

    @Test
    void theTwinCatchesTenPoisonCounters() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        DirectedRule poison = onePlayerRule(ForkOperation.COUNTERS,
                fork -> new PoisonCounters(fork.player("player")));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(poison));
        assertTrue(error.getMessage().contains("the first state-based check"), error.getMessage());
    }

    /**
     * Only a player's Keyword Effects card is the engine's to tidy at the
     * first check: a change naming another effect card or an emblem, and a
     * floating effect gone that no Keyword Effects card made, stay changes.
     */
    @Test
    void theTwinForgivesOnlyAKeywordEffectsCardTheCheckTidies() {
        Map<String, Object> edited = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E3"), Map.of("source", "E9"))));
        Map<String, Object> keywordCardGone = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E9"))));
        Map<String, Object> otherGone = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E3"))));
        Set<String> tidied = Set.of("E3");
        assertEquals(Set.of(), RuleBoard.withoutTidied(
                Set.of("-E3", "E3.lines", "global.floating_effects"), tidied, edited, keywordCardGone));
        assertEquals(Set.of("-E9", "global.floating_effects"), RuleBoard.withoutTidied(
                Set.of("-E9", "global.floating_effects"), tidied, edited, otherGone));
    }

    /** A +1/+1 counter beside a -1/-1 one moves no card, but the check annihilates the pair. */
    @Test
    void theTwinCatchesCountersTheFirstCheckAnnihilates() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Hill Giant");
        board.subject.setCounters(CounterEnumType.M1M1, 1);
        Card giant = board.subject;
        DirectedRule counter = DirectedRule.of("0.0", "0.0-1",
                live -> Optional.of(Binding.empty().with("creature", giant)), new DirectedRule.Variant("v",
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.P1P1, 1))));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(counter));
        assertTrue(error.getMessage().contains("changed the edited board"), error.getMessage());
    }

    /** Ten poison counters on a player, as a {@code counters} edit for the twin's own test. */
    private record PoisonCounters(Player player) implements BoardEdit {
        @Override
        public ForkOperation operation() {
            return ForkOperation.COUNTERS;
        }

        @Override
        public void apply(Game fork) {
            player.setPoisonCounters(10, null);
            BoardEdit.settle(fork);
        }
    }
}
