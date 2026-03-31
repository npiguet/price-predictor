package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility methods for walking SpellAbility chains.
 */
public final class SpellAbilityUtils {

    private SpellAbilityUtils() {}

    static List<String> collectParamInChain(SpellAbility sa, String param) {
        List<String> values = new ArrayList<>();
        if (sa == null) return values;
        String value = sa.getParam(param);
        if (value != null && !value.isEmpty()) {
            values.add(value);
        }
        SpellAbility sub = sa.getSubAbility();
        while (sub != null) {
            value = sub.getParam(param);
            if (value != null && !value.isEmpty()) {
                values.add(value);
            }
            sub = sub.getSubAbility();
        }
        return values;
    }

    /**
     * Walk the SubAbility (and optionally RepeatSubAbility) chain to find the SA that carries
     * {@code ResultSubAbilities}, and return its {@code SpellDescription}.
     * Returns null if no such SA is found.
     */
    public static String findDiceRollDescription(SpellAbility sa) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            if (cur.getParam("ResultSubAbilities") != null) {
                return cur.getParam("SpellDescription");
            }
            SpellAbility repeatSub = cur.getAdditionalAbility("RepeatSubAbility");
            if (repeatSub != null) {
                String desc = findDiceRollDescription(repeatSub);
                if (desc != null) return desc;
            }
        }
        return null;
    }

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into Ability nodes of {@code outputType}.
     * Walks the sub-ability chain starting from {@code sa}; if {@code chaseRepeatSub} is true,
     * also checks the {@code RepeatSubAbility} lateral branch.
     * Returns an empty list when no {@code ResultSubAbilities} param is found.
     */
    public static List<Ability> expandDiceOutcomes(
            SpellAbility sa, AbilityType outputType, boolean chaseRepeatSub) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            String resultSubAbilities = cur.getParam("ResultSubAbilities");
            if (resultSubAbilities != null && !resultSubAbilities.isEmpty()) {
                List<Ability> result = new ArrayList<>();
                for (String entry : resultSubAbilities.split(",")) {
                    String[] kv = entry.trim().split(":", 2);
                    if (kv.length < 2) continue;
                    String range = kv[0].trim();
                    SpellAbility sub = cur.getAdditionalAbility(range);
                    if (sub == null) continue;
                    String rawDesc = sub.getParam("SpellDescription");
                    if (rawDesc == null || rawDesc.isEmpty()) continue;
                    String normalized = AbilityDescription.normalize(AbilityDescription.replaceVert(rawDesc));
                    if (normalized != null) {
                        result.add(new TextAbility(outputType, AbilityDescription.applyCasing(normalized)));
                    }
                }
                return result;
            }
            if (chaseRepeatSub) {
                SpellAbility repeatSub = cur.getAdditionalAbility("RepeatSubAbility");
                if (repeatSub != null) {
                    List<Ability> fromRepeat = expandDiceOutcomes(repeatSub, outputType, true);
                    if (!fromRepeat.isEmpty()) return fromRepeat;
                }
            }
        }
        return List.of();
    }

    static String findParamInChain(SpellAbility sa, String param) {
        String value = sa.getParam(param);
        if (value != null && !value.isEmpty()) {
            return value;
        }
        SpellAbility sub = sa.getSubAbility();
        while (sub != null) {
            value = sub.getParam(param);
            if (value != null && !value.isEmpty()) {
                return value;
            }
            sub = sub.getSubAbility();
        }
        return null;
    }
}
