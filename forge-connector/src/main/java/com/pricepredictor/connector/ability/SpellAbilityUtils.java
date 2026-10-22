package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * Utility methods for walking SpellAbility chains.
 */
final class SpellAbilityUtils {

    private SpellAbilityUtils() {}

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

    /**
     * The text of the effect tree {@code sa}'s chain renders, every node's
     * description joined by spaces in tree order, or empty when it renders
     * none. Only the chain's first tree counts.
     *
     * <p>For an ability whose printed line is one paragraph — an activated
     * ability, a trigger or a chapter whose own description leaves the effect
     * out — so the chain's text is appended to that line instead of being
     * split into lines of its own.
     */
    static String chainText(SpellAbility sa) {
        List<Ability> chain = SpellEffect.fromChain(sa);
        return chain.isEmpty() ? "" : treeText(chain.get(0));
    }

    private static String treeText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String ct = treeText(child);
            if (!ct.isEmpty()) sb.append(' ').append(ct);
        }
        return sb.toString();
    }
}
