package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Whether a resolving clause's own condition held: one entry of a
 * resolution payload's {@code conditions_met} (FR-034).
 *
 * <p>Forge asks {@code sa.metConditions()} of every clause just after the
 * clause hook's "resolving" call ({@code AbilityUtils.resolveApiAbility}), and
 * a clause whose condition fails skips its own effect but not the clauses
 * below it. The answer is otherwise discarded, so the collector asks the same
 * question at the hook, on the same state: {@code metConditions} only reads
 * the game. A gift clause's GIFT trigger runs between the hook and Forge's
 * own question; it only queues triggered abilities (no "whenever you give a
 * gift" trigger is static), so it changes nothing a condition reads.
 *
 * @param segment the clause's chain index, spelled as the {@code attributed_to}
 *                of its events is ({@link EventAttribution#attributedTo}):
 *                {@code root}, or the index down the chain
 * @param met     what {@code metConditions()} answered
 */
record ConditionsMet(String segment, boolean met) {

    /**
     * The parameters {@code SpellAbilityCondition.setConditions} reads, which
     * are what make a clause conditional; {@code ConditionDescription$} is
     * display text and {@code ConditionSVar$} is read by no condition, so
     * neither is here.
     */
    static final Set<String> CONDITION_PARAMS = Set.of(
            "Condition", "ConditionActivationLimit", "ConditionCheckSVar",
            "ConditionChosenColor", "ConditionCompare", "ConditionCompare2",
            "ConditionDefined", "ConditionDefined2", "ConditionFirstCombat",
            "ConditionGameTypes", "ConditionLifeAmount", "ConditionLifeTotal",
            "ConditionManaNotSpent", "ConditionManaSpent", "ConditionNoDifferentColors",
            "ConditionNotPresent", "ConditionOpponentTurn", "ConditionOptionalPaid",
            "ConditionPhases", "ConditionPlayerContains", "ConditionPlayerDefined",
            "ConditionPlayerTurn", "ConditionPresent", "ConditionPresent2",
            "ConditionSVarCompare", "ConditionSorcerySpeed",
            "ConditionTargetValidTargeting", "ConditionTargetsSingleTarget",
            "ConditionZone", "OrConditionCheckSVar", "OrOtherConditionSVarCompare");

    /**
     * The clause's entry, or null for a clause with no condition or one whose
     * chain reaches no line ({@link EventAttribution#UNRESOLVED}).
     */
    static ConditionsMet of(SpellAbility clause) {
        if (clause == null || !isConditional(clause.getMapParams())) {
            return null;
        }
        String segment = EventAttribution.attributedTo(clause);
        if (EventAttribution.UNRESOLVED.equals(segment)) {
            return null;
        }
        return new ConditionsMet(segment, clause.metConditions());
    }

    static boolean isConditional(Map<String, String> params) {
        if (params == null) {
            return false;
        }
        for (String key : params.keySet()) {
            if (CONDITION_PARAMS.contains(key)) {
                return true;
            }
        }
        return false;
    }

    String toJson() {
        return Json.object().value("segment", segment).value("met", met).render();
    }

    static String toJsonArray(List<ConditionsMet> conditions) {
        return Json.array(conditions, ConditionsMet::toJson);
    }
}
