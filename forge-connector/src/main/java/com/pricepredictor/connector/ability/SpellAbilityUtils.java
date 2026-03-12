package com.pricepredictor.connector.ability;

import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility methods for walking SpellAbility chains.
 */
final class SpellAbilityUtils {

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
