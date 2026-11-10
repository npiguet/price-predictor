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
 * <p>Forge asks {@code sa.metConditions()} of every clause once, in {@code
 * AbilityUtils.resolveClause}, and tells the clause listener the answer when
 * the clause's own effect is over ({@code ClauseRun}): {@code
 * CONDITIONS_UNMET} when the condition failed, {@code RAN} or {@code
 * UNLESS_COST_DECIDED} when it held. The collector takes that answer rather
 * than asking again: a second {@code metConditions()} is a second read of a
 * game the clause's own trigger (a gift's) may already have changed, and
 * reads nothing Forge acted on.
 *
 * @param segment the clause's chain index, spelled as the {@code attributed_to}
 *                of its events is ({@link EventAttribution#attributedTo}):
 *                {@code root}, or the index down the chain
 * @param met     whether Forge found the condition held
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
     * The clause's entry, or null for a clause with no condition, one whose
     * chain reaches no line ({@link EventAttribution#UNRESOLVED}), or one
     * Forge gave no answer for: the clause threw before its condition was
     * decided, or the jar's listener passes no {@code ClauseRun}.
     *
     * @param run the constant name of the {@code ClauseRun} Forge reported
     *            ({@link PatchHooks#clauseRunName}), or null
     */
    static ConditionsMet of(SpellAbility clause, String run) {
        if (clause == null || run == null || !isConditional(clause.getMapParams())) {
            return null;
        }
        Boolean met = switch (run) {
            case "RAN", "UNLESS_COST_DECIDED" -> Boolean.TRUE;
            case "CONDITIONS_UNMET" -> Boolean.FALSE;
            default -> null;
        };
        if (met == null) {
            return null;
        }
        String segment = EventAttribution.attributedTo(clause);
        if (EventAttribution.UNRESOLVED.equals(segment)) {
            return null;
        }
        return new ConditionsMet(segment, met);
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
