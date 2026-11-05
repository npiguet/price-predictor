package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * Whether a line's text branches on whose turn it is (FR-057).
 *
 * <p>A clause carrying {@code ConditionPlayerTurn$} or
 * {@code ConditionOpponentTurn$} resolves one way on its controller's turn
 * and another on an opponent's ("If it's your turn, scry 2"; "If it's not
 * your turn, untap those creatures"). Forge's {@code SpellAbilityCondition}
 * reads exactly these two parameters for whose turn it is; the same names
 * `corpus_curation._CLAUSE_PARAMS` reads a resolved clause's condition by.
 */
final class TurnConditions {

    /** The clause parameters naming whose turn a clause resolves on. */
    static final List<String> WHOSE_TURN = List.of("ConditionPlayerTurn", "ConditionOpponentTurn");

    private TurnConditions() {
    }

    /**
     * Whether any clause the line offers, a charm's modes and their chains
     * included, resolves only on one player's turn.
     */
    static boolean branchesOnWhoseTurn(SpellAbility ability) {
        for (SpellAbility clause : TargetingClauses.offered(ability)) {
            for (String param : WHOSE_TURN) {
                if (clause.hasParam(param)) {
                    return true;
                }
            }
        }
        return false;
    }
}
