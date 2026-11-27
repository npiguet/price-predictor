package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.EnumSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The registered directed rules: on a {@code TestBoard} set up
 * as each rule's scenario sets up its base variant, the precondition binds the
 * scenario's objects, and on a board missing the base it binds nothing; and
 * the fork, copied as a directed fork copies the live game and edited by the
 * rule's variant, differs from its real twin, the live board, by that
 * variant's one fact (Feature 025 story 3, acceptance 3).
 *
 * <p>The twin comparison reads the board-only snapshot a record would carry,
 * member by member ({@code BoardChanges}), and checks first that the unedited
 * copy reads as the live board does, so what differs afterwards is the edit's.
 *
 * <p>This class holds the registry's own test; the rules' tests are in the
 * {@code RegisteredRules*Test} family classes beside it, on the board
 * {@link RuleBoard} builds and the checks {@link RuleTwinChecks} makes.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesTest {

    // ── the registry ────────────────────────────────────────────────────

    /**
     * Every operation but {@code keyword} and {@code colour_type} is a
     * registered rule's: their rules (702.63c, 702.13b) were registered for
     * no listed run and left the registry. The protection forks edit through
     * {@code keyword}, and the {@code ForkOperation*Test} classes check each operation's
     * one fact.
     */
    @Test
    void theRegisteredRulesCoverEveryOperationButKeywordAndColourType() {
        Set<ForkOperation> covered = EnumSet.noneOf(ForkOperation.class);
        for (DirectedRule rule : DirectedRules.registered().rules()) {
            rule.variants().values().forEach(application -> covered.add(application.operation()));
        }
        assertEquals(EnumSet.complementOf(EnumSet.of(ForkOperation.KEYWORD, ForkOperation.COLOUR_TYPE)), covered);
    }
}
